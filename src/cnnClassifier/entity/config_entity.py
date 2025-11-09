from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)

class DataIngestionConfig:
    rootdir:Path
    source_url:str
    local_data_file:Path

    unzip_dir:Path

@dataclass(frozen=True)
# frozen=true as we don't wnat to do any changes in it
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int