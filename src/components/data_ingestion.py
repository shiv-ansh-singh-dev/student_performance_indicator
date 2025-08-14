import os
#os is used to create directories and files
#os.path.join is used to join the path of the file

import sys
#sys is used to get the path of the file and handle exceptions

#take input from logger.py and exception.py
#logger.py is used to log the information and exception.py is used to handle the exceptions
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

#decorator(dataclass), as we use init to define a class we can directly define the attributes of the class
#os.path.join is used to join the path of the file
#here we are defining the configuration for data ingestion, we are using dataclass to define the attributes of the class
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() #has three inputs

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/stud.csv') 
            #reading from dataset also we can read from mongodb or any other database
            logging.info('Read the dataset as dataframe') #use logging to log the information so that we can track the information and errors

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 
            #convert raw data into train and test set
            #os.makedirs is used to create the directory if it does not exist, exist_ok=True
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) 

            logging.info("Train test split initiated") 
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            #logging.info("Train test split completed") #train_test_split is used to split the data into train and test set
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            #save the train and test set into the respective paths defined in the DataIngestionConfig class
            logging.info("Inmgestion of the data iss completed")
            #return the train and test set paths
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))