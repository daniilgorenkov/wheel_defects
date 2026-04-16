import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from src.mixins.file_operator import FileOperatorMixin
from collections import Counter
from config import PreprocessorConfig


class PreprocessorMixin(FileOperatorMixin):

    def preprocess(self):
        preprocessed_data = {}
        dirs = self.get_all_dirs_in_data()
        for dirname in dirs:
            for fname in self.get_all_fnames(dirname):
                preprocessed_data[fname] = self.preprocess_file(fname, dirname)

        samples = self.build_samples(preprocessed_data)
        self.save_samples(self.build_samples(preprocessed_data), "prep_data")
        return samples

    def create_target(self, fname: str):
        marker = fname.split("_")[-2]
        if "17" in marker or "12" in marker:
            return 1
        else:
            return 0

    def build_samples(self, data: dict):
        samples = []

        for file_name, file_data in data.items():
            for sample_id, sample in file_data.items():
                samples.append(sample)

        c = Counter([i["target"] for i in samples])
        print(f"Class distribution in samples: {c}")
        return samples

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        enc = StandardScaler()
        return enc.fit_transform(signal.reshape(-1, 1)).flatten()

    def preprocess_file(self, fname: str, dirname: str):

        df = self.load(fname, dirname, use_cols=PreprocessorConfig.USE_COLS)
        df["datatime"] = pd.to_datetime(df["datatime"])
        df = self.filter_by_speed(df)
        df["accz"] = self.normalize_signal(df["accz"].values)
        revolutions = self.split_data_by_rotation(
            df,
            wheel_diameter=PreprocessorConfig.WHEEL_DIAMETER,
            gap_reset_sec=PreprocessorConfig.GAP_RESET_SEC,
            min_speed_kmh=PreprocessorConfig.MIN_SPEED_KMH,
            min_points_per_rev=PreprocessorConfig.MIN_POINTS_PER_REV,
        )

        filtered_revolutions = []
        for revolution_df in revolutions:
            clean_revolution = self.filter_by_acceleration(revolution_df)
            if len(clean_revolution) >= PreprocessorConfig.MIN_POINTS_PER_REV:
                filtered_revolutions.append(clean_revolution)

        return self.convert_dfs_to_arrays(filtered_revolutions, fname)

    def filter_by_speed(self, df: pd.DataFrame, bottom_edge: int = 11, top_edge: int = 30):
        return df[(df["spd"] >= bottom_edge) & (df["spd"] <= top_edge)]

    def filter_by_acceleration(
        self,
        df: pd.DataFrame,
        time_col="datatime",
        value_col="accz",
        min_abs_amplitude=0.5,
        max_abs_amplitude=24,
    ):
        signal = df.dropna().sort_values(time_col).copy()

        if min_abs_amplitude is not None:
            signal = signal[signal[value_col].abs() >= min_abs_amplitude].copy()
        if max_abs_amplitude is not None:
            signal = signal[signal[value_col].abs() <= max_abs_amplitude].copy()

        return signal.reset_index(drop=True)

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal using z-score (mean=0, std=1)."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-8:
            return signal
        return (signal - mean) / std

    def split_data_by_rotation(
        self,
        df: pd.DataFrame,
        wheel_diameter: float,
        gap_reset_sec: float = 0.5,
        min_speed_kmh: float = 0.1,
        min_points_per_rev: int = 8,
    ):
        boundaries = self.find_rotation_boundaries(
            frame=df,
            wheel_diameter=wheel_diameter,
            gap_reset_sec=gap_reset_sec,
            min_speed_kmh=min_speed_kmh,
            min_points_per_rev=min_points_per_rev,
        )
        print("Found rotation boundaries:", len(boundaries))
        return [df.iloc[start:end].reset_index(drop=True) for start, end in boundaries]

    def find_rotation_boundaries(
        self,
        frame: pd.DataFrame,
        wheel_diameter: float,
        gap_reset_sec: float = 0.5,
        min_speed_kmh: float = 0.1,
        min_points_per_rev: int = 8,
    ):
        if len(frame) < 2:
            return []

        wheel_circumference = np.pi * wheel_diameter

        work = frame.sort_values("datatime").reset_index(drop=True).copy()

        # Реальный шаг времени между измерениями (сек)
        dt = work["datatime"].diff().dt.total_seconds().fillna(0.0)
        dt = dt.clip(lower=0.0)

        # Скорость (м/с) и накопленная дистанция (м)
        speed_ms = work["spd"].to_numpy(dtype=float) / 3.6
        ds = speed_ms * dt.to_numpy(dtype=float)

        reset_mask = (dt <= 0.0) | (dt > gap_reset_sec) | (work["spd"] < min_speed_kmh)

        boundaries = []
        block_start = 0
        for i in range(1, len(work) + 1):
            is_reset = i == len(work) or bool(reset_mask.iloc[i])
            if not is_reset:
                continue

            if i - block_start >= 2:
                block_ds = ds[block_start:i]
                block_dist = np.cumsum(block_ds)
                total_distance = block_dist[-1]
                revolutions_count = int(total_distance // wheel_circumference)

                if revolutions_count > 0:
                    local_points = [0]
                    for turn in range(1, revolutions_count + 1):
                        target_dist = turn * wheel_circumference
                        idx_local = int(np.searchsorted(block_dist, target_dist, side="left"))
                        if idx_local > local_points[-1]:
                            local_points.append(idx_local)

                    for left, right in zip(local_points[:-1], local_points[1:]):
                        left_global = block_start + left
                        right_global = block_start + right
                        if right_global - left_global >= min_points_per_rev:
                            boundaries.append((left_global, right_global))

            block_start = i

        return boundaries

    def convert_dfs_to_arrays(self, dfs: list[pd.DataFrame], fname: str):
        arrays = {}
        for idx, df in enumerate(dfs):
            signal = df["accz"].values
            if PreprocessorConfig.NORMALIZE_SIGNAL:
                signal = self.normalize_signal(signal)
            arrays[idx] = {"X": signal, "speed": df["spd"].values, "target": self.create_target(fname)}
        return arrays
