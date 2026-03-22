from pyexpat import model

from src.mixins.dataset_builder import DataProcessor
from src.mixins.preprocessor import PreprocessorMixin
from src.mixins.trainer import Trainer
from src.models.baseline import Baseline
from src.models.lite_baseline import LiteBaseline
import config
from torch import nn
import torch


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
            loss_fn=nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            lr=config.TrainerConfig.lr,
        )


if __name__ == "__main__":
    model_config = {
        "out_channels": 32,
        "kernel_size": 5,
        "use_norm": True,
        "num_groups": 4,
        "dropout": 0.2,
    }

    model_builder = ModelBuilder(
        preprocessor=PreprocessorMixin,
        data_processor=DataProcessor,
        trainer=Trainer,
        model=LiteBaseline,
        model_config=model_config,
    )
    model_builder.build()
