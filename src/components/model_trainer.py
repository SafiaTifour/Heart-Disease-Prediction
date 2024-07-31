import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import load_object

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr_x, train_arr_y, test_arr_x, test_arr_y):
        try:
            logging.info("Starting model trainer...")

            X_train, y_train, X_test, y_test = (
                train_arr_x,
                train_arr_y,
                test_arr_x,
                test_arr_y
            )

        

            model = LogisticRegression()
            model.fit(X_train, y_train)

            
            X_train_prediction = model.predict(X_train)
            training_data_accuracy = accuracy_score(X_train_prediction, y_train)
        
            X_test_prediction = model.predict(X_test)
            test_data_accuracy = accuracy_score(X_test_prediction, y_test)

            print(test_data_accuracy)
            
            if test_data_accuracy < 0.6:
                raise CustomException("The model is performing BAD")
            
            logging.info("Best model found")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= model
            )


        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    data_transformation_obj = DataTransformation()
    model_trainer = ModelTrainer()

    train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()

    train_arr_x, test_arr_x, train_arr_y, test_arr_y = data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)
    
    model_trainer.initiate_model_trainer(train_arr_x, train_arr_y, test_arr_x, test_arr_y)



