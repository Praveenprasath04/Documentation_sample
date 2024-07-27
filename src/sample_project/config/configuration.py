from sample_project.constants import *
from sample_project.utils.common import read_yaml, create_directories
from sample_project.entity.config_entity import DataIngestionConfig
from sample_project.entity.config_entity import PrepareBaseModelConfig
from sample_project.entity.config_entity import DataPreprocessConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config


    def get_data_preprocess_config(self) -> DataPreprocessConfig:
        config = self.config.data_preprocessing
        params = self.params

        create_directories([config.root_dir])

        data_preprocess_config = DataPreprocessConfig(
            root_dir=config.root_dir,
            data_dir = config.data_dir,
            train_loader_dir = config.train_loader_dir,
            valid_loader_dir = config.valid_loader_dir,
            test_loader_dir = config.test_loader_dir,
            params_image_dim = params.IMAGE_DIM,
            params_batch_size= params.BATCH_SIZE,
            params_valid_size = params.VALID_SIZE

        )

        return data_preprocess_config
    
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_batch_size = self.params.BATCH_SIZE,
            params_valid_size =  self.params.VALID_SIZE,  
            params_Loss_function = self.params.LOSS_FUNCTION,
            params_learning_rate = self.params.LEARNING_RATE,
            params_momentum = self.params.MOMENTUM ,
            params_image_dim = self.params.IMAGE_DIM 
        )

        return prepare_base_model_config