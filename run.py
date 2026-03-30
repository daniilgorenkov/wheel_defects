from functools import partial

from src.mixins.dataset_builder import DataProcessor
from src.mixins.preprocessor import PreprocessorMixin
from src.mixins.trainer import Trainer
from src.models.baseline import Baseline
from src.mixins.metrics import FocalLoss
from src.models.lite_baseline import LiteBaseline
from src.models.transformer import CNNTransformerEncoder
from src.models.three_head_model import ThreeHeadModel
import config
from torch import nn
import torch
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--train", action="store_true", help="Train the model")
args = arg_parser.parse_args()


class ModelBuilder:

    def __init__(
        self,
        preprocessor: PreprocessorMixin,
        data_processor: DataProcessor,
        trainer: Trainer,
        model: nn.Module,
        model_config: dict,
    ):
        self.model = model(**model_config)
        self.preprocessor = preprocessor()
        self.data_processor = data_processor()

        self.device = config.TrainerConfig.device

        self.trainer = trainer(
            model=self.model,
            device=self.device,
        )

    def build(self):
        # Preprocess the data
        if args.train:
            data = self.preprocessor.load_samples("prep_data")
        else:
            data = self.preprocessor.preprocess()

        # Process the dataset
        train_loader, val_loader = self.data_processor.process(
            data=data,
            train_size=config.ProcessorConfig.train_size,
            batch_size=config.ProcessorConfig.batch_size,
        )

        # Train the model
        self.trainer.fit(
            epochs=config.TrainerConfig.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=partial(FocalLoss, gamma=2.0),
            optimizer=torch.optim.AdamW,
            lr=config.TrainerConfig.lr,
            patience=config.TrainerConfig.patience,
        )


if __name__ == "__main__":
    model_config = {
        "out_channels_signal": 64,
        # "kernel_size_signal": 7,
        "num_groups_signal": 8,
        "out_channels_speed": 2,
        "kernel_size_speed": 3,
        "nheads": 4,
        "enc_layers": 2,
        "dropout": 0.2,
    }

    model_builder = ModelBuilder(
        preprocessor=PreprocessorMixin,
        data_processor=DataProcessor,
        trainer=Trainer,
        model=CNNTransformerEncoder,
        model_config=model_config,
    )
    model_builder.build()
