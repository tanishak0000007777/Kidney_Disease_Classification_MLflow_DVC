import os
import tensorflow as tf
from pathlib import Path
import time

from src.cnnClassifier.entity.config_entity import TrainingConfig

tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        """Load the base model and compile it."""
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        """Prepare training and validation data generators."""
        datagenerator_kwargs = dict(rescale=1. / 255, validation_split=0.20)
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

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save trained model to disk."""
        model.save(path)

    def train(self):
        """Train the model with checkpointing and resume support."""
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # ✅ Define checkpoint directory and pattern
        checkpoint_dir = Path("artifacts/training/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(checkpoint_dir / "cp-{epoch:02d}.weights.h5")

        # ✅ Create checkpoint callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=False,
            verbose=1
        )

        # ✅ Resume from latest checkpoint if available
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"🔄 Resuming training from checkpoint: {latest_checkpoint}")
            self.model.load_weights(latest_checkpoint)
        else:
            print("🚀 Starting fresh training...")

        # ✅ Train model with checkpointing
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            callbacks=[checkpoint_callback],
            verbose=1
        )

        # ✅ Save final trained model
        self.save_model(path=self.config.trained_model_path, model=self.model)
        print(f"✅ Final model saved at: {self.config.trained_model_path}")
