from src.config.configuration import ConfigurationManager
from src.data.data_ingestion import DataIngestion
from src.data.data_validation import DataValidation
from src.data.data_transformation import DataTransformation
from src.models.model_trainer import ModelTrainer
from src.utils.logger import setup_logger
import sys

logger = setup_logger(__name__)


class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run_training_pipeline(self):
        """Execute the complete training pipeline"""
        try:
            logger.info("Starting training pipeline")

            # Data Ingestion
            logger.info("Step 1: Data Ingestion")
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            # Data Validation
            logger.info("Step 2: Data Validation")
            data_validation = DataValidation()
            import pandas as pd

            train_df = pd.read_csv(train_data_path)
            if not data_validation.validate_schema(train_df):
                raise Exception("Data validation failed")

            # Data Transformation
            logger.info("Step 3: Data Transformation")
            data_transformation = DataTransformation()
            X_train, y_train, X_test, y_test, preprocessor_path = (
                data_transformation.initiate_data_transformation(
                    train_data_path, test_data_path
                )
            )

            # Model Training
            logger.info("Step 4: Model Training")
            model_training_config = self.config_manager.get_model_training_config()
            model_trainer = ModelTrainer(config=model_training_config)
            model_path = model_trainer.initiate_model_trainer(
                X_train, y_train, X_test, y_test
            )

            logger.info("Training pipeline completed successfully")
            return model_path, preprocessor_path

        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise e


if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_training_pipeline()
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        sys.exit(1)
