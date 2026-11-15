"""
This module provides functionality for building the ML model.

It contains the ModelBuilderService class, which offers methods to
train a model and save it to a specified directory. 
"""

from pathlib import Path

from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelBuilderService:
    """
    A service class for building and saving the ML model.

    This class provides functionality to train the ML model and save it 
    to the specified path.

    Attributes:
        model_path (Path): The path to the model file.
        model_name (str): The name of the loaded model.

    Methods:
        __init__(self, model_path): Initializes the ModelBuilderService.
        train_model(self): Trains the model and saves it to the specified 
        directory.
    """

    def __init__(
        self,
        model_path: str = (
            f'{model_settings.model_output_path}/{model_settings.model_name}'
        ),
    ) -> None:
        """Initializes the ModelBuilderService instance."""
        self.model_path = Path(model_path)
        self.model_name = model_settings.model_name

    def train_model(self) -> None:
        """Train a model from a specified path and save the model to the model's dir ."""
        logger.info(
            f'Starting training {model_settings.model_name}',
            f'Saving the model at {self.model_path}',
            )
        build_model()
