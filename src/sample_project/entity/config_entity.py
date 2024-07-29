from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL: str
    local_data_file : Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreprocessConfig:
    root_dir: Path
    data_dir : Path
    train_loader_dir: Path
    valid_loader_dir: Path
    test_loader_dir: Path
    params_batch_size : int
    params_valid_size : float
    params_image_dim : int

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_batch_size : int
    params_valid_size : float
    params_Loss_function : str
    params_learning_rate : float
    params_momentum: float
    params_image_dim : int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path : Path
    base_model_path: Path
    train_loader_dir : Path
    valid_loader_dir :  Path
    params_batch_size : int
    params_valid_size : float
    params_learning_rate : float
    params_momentum: float
    params_image_dim : int
    params_random_seed: int
    params_epochs: int

@dataclass(frozen=True)
class TestingConfig:
    root_dir: Path
    trained_model_path : Path
    test_loader_dir :  Path
    params_random_seed: int
