import logging

from src.training.data_preparation import save_data_train_test

logger = logging.getLogger(__name__)

def main():
    "Save data train and test on disk as a csv file"
    logger.info("Saving data train and test...")
    save_data_train_test()
    logger.info("Data train and test saved. ✅")


if __name__ == '__main__':
    main()