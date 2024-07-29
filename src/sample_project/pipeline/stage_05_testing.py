from sample_project.config.configuration import ConfigurationManager
from sample_project.components.testing import Testing 
from sample_project import logger

STAGE_NAME = "Testing"

class FinalTestingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        testing__config = config.get_testing_config()
        testing_step = Testing(config=testing__config)
        testing_step.initializing_model()
        testing_step.loading_iterators()
        testing_step.test()





if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FinalTestingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e