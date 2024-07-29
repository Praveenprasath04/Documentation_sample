from sample_project.config.configuration import ConfigurationManager
from sample_project.components.training import Training 
from sample_project import logger

STAGE_NAME = "Training"

class FinalTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training__config = config.get_training_config()
        training_step = Training(config=training__config)
        training_step.initializing_model()
        training_step.loading_iterators()
        training_step.train()





if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FinalTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e