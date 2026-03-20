import numpy as np
import pandas as pd
from src.mixins.file_operator import FileOperatorMixin
from collections import Counter


class PreprocessorConfig:
    USE_COLS: list = ["datatime", "accz", "spd"]
    WHEEL_DIAMETER = 0.960  # м
    CIRCUMFERENCE = np.pi * WHEEL_DIAMETER


class PreprocessorMixin(FileOperatorMixin):

    def preprocess(self):
        preprocessed_data = {}
        dirs = self.get_all_dirs_in_data()
        for dirname in dirs:
            for fname in self.get_all_fnames(dirname):
                preprocessed_data[fname] = self.preprocess_file(fname, dirname)

        return self.build_samples(preprocessed_data)

    def create_target(self, fname: str):
        marker = fname.split("_")[-2]
        if "17" in marker:
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

    def preprocess_file(self, fname: str, dirname: str):

        df = self.load(fname, dirname, use_cols=PreprocessorConfig.USE_COLS)
        df["datatime"] = pd.to_datetime(df["datatime"])

        df = self.filter_by_speed(df)
        df = self.filter_by_acceleration(df)
        df = self.split_data_by_rotation(df, wheel_circumference=PreprocessorConfig.CIRCUMFERENCE)

        return self.convert_dfs_to_arrays(df, fname)

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

    def split_data_by_rotation(self, df: pd.DataFrame, wheel_circumference: float, gap_factor: float = 1.5):
        boundaries = self.find_rotation_boundaries(df, wheel_circumference, gap_factor)
        print("Found rotation boundaries:", len(boundaries))
        return [df.iloc[start:end].reset_index(drop=True) for start, end in boundaries]

    # ...existing code...

    def split_data_by_rotation(self, df: pd.DataFrame, wheel_circumference: float):
        boundaries = self.find_rotation_boundaries(df, wheel_circumference)
        print("Found rotation boundaries:", len(boundaries))
        return [df.iloc[start:end].reset_index(drop=True) for start, end in boundaries]

    def find_rotation_boundaries(self, frame: pd.DataFrame, wheel_circumference: float):
        if len(frame) < 2:
            return []

        # Реальный шаг времени между измерениями (сек)
        dt = frame["datatime"].diff().dt.total_seconds().fillna(0.0)
        dt = dt.clip(lower=0.0).to_numpy(dtype=float)

        # Скорость (м/с) и накопленная дистанция (м)
        speed_ms = frame["spd"].to_numpy(dtype=float) / 3.6
        distance = np.cumsum(speed_ms * dt)

        total_distance = distance[-1]
        revolutions_count = int(total_distance // wheel_circumference)
        if revolutions_count <= 0:
            return []

        boundaries = [0]
        for turn in range(1, revolutions_count + 1):
            target_dist = turn * wheel_circumference
            idx = int(np.searchsorted(distance, target_dist, side="left"))
            if idx > boundaries[-1] and idx < len(frame):
                boundaries.append(idx)

        if len(boundaries) < 2:
            return []

        return [(left, right) for left, right in zip(boundaries[:-1], boundaries[1:])]

    def convert_dfs_to_arrays(self, dfs: list[pd.DataFrame], fname: str):
        arrays = {}
        for idx, df in enumerate(dfs):
            arrays[idx] = {"X": df["accz"].values, "speed": df["spd"].values, "target": self.create_target(fname)}
        return arrays
