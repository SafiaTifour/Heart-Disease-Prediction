import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig


class DataTransformation:
    def __init__(self):
        pass
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data is completed")


            target_column_name = "target"

            input_train = train_df.drop(columns = [target_column_name], axis = 1)
            target_train = train_df[target_column_name]

            input_test = test_df.drop(columns = [target_column_name], axis = 1)
            target_test = test_df[target_column_name]

            train_arr_x = np.array(input_train)
            test_arr_x = np.array(input_test)

            train_arr_y = np.array(target_train)
            test_arr_y = np.array(target_test)

            return (
                train_arr_x,
                test_arr_x,
                train_arr_y,
                test_arr_y,
            )

            

        except Exception as e:
            raise CustomException(e, sys)




    


            