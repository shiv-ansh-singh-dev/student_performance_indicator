import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer #use to create pipelines for numerical and categorical columns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

#exception handling and logging
from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass #dataclass is used to create a class with attributes and methods
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl") #pickle file to save the preprocessor object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig() 
        #create my pickle file and convertiing the data into numerical and categorical vice versa 

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), #hadling missing values
                ("scaler",StandardScaler()) #performing standard scaling

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), #handling missing values
                ("one_hot_encoder",OneHotEncoder()), #performing one hot encoding
                ("scaler",StandardScaler(with_mean=False)) #performing standard scaling
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            #combining pipelines numerical and categorical columns
            #ColumnTransformer is used to apply different transformations to different columns
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        '''This function is responsible for data transformation'''
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            #dropping the target column from the train dataframe and storing it in a separate variable
            #input_feature_train_df is the train dataframe without the target column

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            #dropping the target column from the test dataframe and storing it in a separate variable
            #input_feature_test_df is the test dataframe without the target column
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            #difference between fit transform and transform is that fit_transform is used to fit the data and transform it at the same time
            #transform is used to transform the data after it has been fitted
            #fit_transform is used to fit the data and transform it at the same time

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            #c__ is used to concatenate the input feature and target feature arrays
            #np.array is used to convert the target feature dataframe into an array
            #train_arr and test_arr are the final arrays that will be used to train the model

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            #save the preprocessing object to a pickle file
            #save_object is a utility function to save the object to a file
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj #save the preprocessing object to a file

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
