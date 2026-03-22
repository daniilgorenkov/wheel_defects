from src.mixins.dataset_builder import DataProcessor
from src.mixins.preprocessor import PreprocessorMixin
from src.mixins.trainer import Trainer
from src.models.head_encoder import WheelBaseline
import config
from torch import nn
import torch


class ModelBuilder:

    def __init__(self, preprocessor: PreprocessorMixin, data_processor: DataProcessor, trainer: Trainer):
        self.model = WheelBaseline(
            n_classes=config.ModelConfig.n_classes,
            embedding_dim=config.ModelConfig.embedding_dim,
            base_channels=config.ModelConfig.base_channels,
            use_speed=config.ModelConfig.use_speed,
            dropout=config.ModelConfig.dropout,
        )
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
    model_builder = ModelBuilder(
        preprocessor=PreprocessorMixin,
        data_processor=DataProcessor,
        trainer=Trainer,
    )
    model_builder.build()
