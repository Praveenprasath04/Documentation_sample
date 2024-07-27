from sample_project.config.configuration import ConfigurationManager
from sample_project.components.data_preprocessing import DataPreprocess 
from sample_project import logger

STAGE_NAME = "Data Preprocessing"

class DataPreprocessTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocess__config = config.get_data_preprocess_config()
        data_preprocess = DataPreprocess(config=data_preprocess__config)
        data_preprocess.transform_data()
        data_preprocess.split_valid_set()
        data_preprocess.data_loaders()





if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e