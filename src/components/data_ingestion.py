import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

# use dataclass to define class variables only
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # the 3 class variables of DataIngestionConfig class are saved to class variable ingestion_config

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/ENB2012_data.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True) # create the path

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Initiate train test set split")

            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    data_ingestion1 = DataIngestion()
    traint_set, test_set = data_ingestion1.initiate_data_ingestion()

    data_transformation1 = DataTransformation()
    data_transformation1.initiate_data_transformation(traint_set, test_set)
