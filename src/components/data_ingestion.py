from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import os


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion method')
        try:
            logging.info("Reading the data")
            df = pd.read_csv('src\data\heart_disease_data.csv')
            
            logging.info("Creating the directory for the data")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header= True)

            logging.info("Train, test split intiated")

            train_data, test_data = train_test_split(df, test_size = 0.2, random_state = 42)

            logging.info("Loading the train data into the corresponding files")
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header= True)
            logging.info("Loading the test data into the corresponding files")
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header= True)

            logging.info("Ingestion of the data completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path  
                )
        
        except Exception as e:
            raise CustomException(e, sys)
        







