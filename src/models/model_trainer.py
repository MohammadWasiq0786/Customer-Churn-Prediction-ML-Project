import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from pathlib import Path
from src.utils.logger import setup_logger
from src.config.configuration import ModelTrainingConfig
import numpy as np

logger = setup_logger(__name__)


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.models = {
            "RandomForestClassifier": RandomForestClassifier(**config.hyperparameters),
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
            "SVC": SVC(probability=True, random_state=42),
        }

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        return metrics

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        """Train and evaluate models"""
        try:
            logger.info("Starting model training")

            # Set MLflow experiment
            mlflow.set_experiment(
                self.config.experiment_name
                if hasattr(self.config, "experiment_name")
                else "churn_prediction"
            )

            best_model = None
            best_score = 0
            best_model_name = ""

            for model_name, model in self.models.items():
                with mlflow.start_run(run_name=f"{model_name}_training"):
                    logger.info(f"Training {model_name}")

                    # Train model
                    model.fit(X_train, y_train)

                    # Evaluate model
                    metrics = self.evaluate_model(model, X_test, y_test)

                    # Log metrics
                    mlflow.log_metrics(metrics)
                    mlflow.log_params(model.get_params())

                    # Log model
                    mlflow.sklearn.log_model(model, "model")

                    logger.info(
                        f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}"
                    )

                    # Track best model
                    if metrics["accuracy"] > best_score:
                        best_score = metrics["accuracy"]
                        best_model = model
                        best_model_name = model_name

            # Save best model
            model_path = Path("artifacts/model.pkl")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, model_path)

            logger.info(
                f"Best model ({best_model_name}) saved with accuracy: {best_score:.4f}"
            )
            return str(model_path)

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise e
