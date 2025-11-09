import os
from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline


def is_stage_done(path: str) -> bool:
    """Check if the output of a stage already exists."""
    return os.path.exists(path) and os.path.getsize(path) > 0


# ===================== Stage 01: Data Ingestion ===================== #
STAGE_NAME = "Data Ingestion stage"

try:
    data_zip_path = "artifacts/data_ingestion/data.zip"
    if not is_stage_done(data_zip_path):
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    else:
        logger.info(f"Skipping {STAGE_NAME} — already completed.")
except Exception as e:
    logger.exception(e)
    raise e


# ===================== Stage 02: Prepare Base Model ===================== #
STAGE_NAME = "Prepare Base Model"

try:
    model_path = "artifacts/prepare_base_model/base_model_updated.h5"
    if not is_stage_done(model_path):
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    else:
        logger.info(f"Skipping {STAGE_NAME} — already completed.")
except Exception as e:
    logger.exception(e)
    raise e
