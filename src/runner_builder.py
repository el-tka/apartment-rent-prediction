"""
Runner script for the application.

This script serves as the entry point for the application. 
It initializes the necessary components, trains the model, 
saves it to the specified directory and logs the results.
"""
from loguru import logger

from model.model_builder import ModelBuilderService


@logger.catch
def main() -> None:
    """
    Main function to run the application.

    It initializes the ModelBuilderService, trains the model,
    saves it to the specified directory and logs the results.

    Returns:
        None
    """
    logger.info("Building the model...")
    ml_svc = ModelBuilderService()
    ml_svc.train_model()


if __name__ == "__main__":
    main()
