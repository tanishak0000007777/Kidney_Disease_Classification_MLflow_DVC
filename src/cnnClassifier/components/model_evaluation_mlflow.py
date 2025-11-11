import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src.cnnClassifier.utils.common import read_yaml, create_directories, save_json
from  src.cnnClassifier.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None

    # Create validation data generator
    def _valid_generator(self):
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.30)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    # Load saved model
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    # Evaluate the model
    def evaluation(self):
        print("🔍 Loading model...")
        self.model = self.load_model(self.config.path_of_model)
        print("✅ Model loaded successfully.")

        print("📦 Preparing validation data...")
        self._valid_generator()

        print("🧪 Evaluating model...")
        self.score = self.model.evaluate(self.valid_generator)
        print(f"✅ Evaluation complete — Loss: {self.score[0]:.4f}, Accuracy: {self.score[1]:.4f}")

        self.save_score()

    # Save scores locally
    def save_score(self):
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        save_json(path=Path("scores.json"), data=scores)
        print("💾 Scores saved to scores.json")

    # Log results to DagsHub (MLflow)
    def log_into_mlflow(self):
        print("🚀 Logging metrics & model to MLflow (DagsHub)...")
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # ✅ Create a custom run name (e.g. "VGG16 Evaluation Run")
        with mlflow.start_run(run_name="VGG16 Evaluation Run"):
            # Log parameters and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1]
            })

            # ✅ Save model locally first
            import os
            local_model_path = "artifacts/evaluated_model"
            os.makedirs(local_model_path, exist_ok=True)
            model_save_path = os.path.join(local_model_path, "VGG16Model.h5")
            self.model.save(model_save_path)

            # ✅ Log as artifact (model)
            mlflow.log_artifact(model_save_path, artifact_path="VGG16Model")

            # ✅ Add custom tags for better organization
            mlflow.set_tag("model_name", "VGG16")
            mlflow.set_tag("phase", "evaluation")
            mlflow.set_tag("framework", "TensorFlow")

        print("🎯 Metrics & 'VGG16Model' successfully logged to DagsHub!")

