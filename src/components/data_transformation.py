import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join("artifacts", "preprocessing.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """
        this function is used to perform data transformation based on different types of features
        """
        try:
            # preprocessing pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("normalize", StandardScaler()),
                ]
            )

            # preprocessing pipeline for categorical features
            # cat_pipeline = Pipeline(
            #     steps=[
            #         ('impute', SimpleImputer(strategy="most_frequent")),
            #         ("normalize", OneHotEncoder()),
            #         ("normalize", StandardScaler())
            #     ]
            # )

            logging.info("Numerical and/or categorical features are imputed, (encoded) and scaled")

            # preprocessing = ColumnTransformer(
            #     ("num_pipeline", num_pipeline, make_column_selector(dtype_include=np.number)),
            #     ("cat_pipeline", cat_pipeline, make_column_selector(dtype_include=object)),
            # )

            # the data set has only numerical features
            preprocessing = make_column_transformer(
                (num_pipeline, make_column_selector(dtype_include=np.number))
            )

            return preprocessing

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test datasets completed")

            logging.info("Obtaining preprocessing object file")

            preprocessing_obj = self.get_data_transformer_obj()

            target_columns = ["Y1", "Y2"]
            input_feature_train_df = train_df.drop(columns=target_columns, axis=1)
            target_feature_train_df = train_df[target_columns]

            input_feature_test_df = test_df.drop(columns=target_columns, axis=1)
            target_feature_test_df = test_df[target_columns]

            logging.info(
                "Applying preprocessing object on training dataframe and testing datasets"
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df) # no fit() on test data!

            # two sets of train/test for heating and cooling loads, respectively
            train_arr1 = np.c_[input_feature_train_arr, np.array(target_feature_train_df[target_columns[0]])]
            test_arr1 = np.c_[input_feature_test_arr, np.array(target_feature_test_df[target_columns[0]])]

            train_arr2 = np.c_[input_feature_train_arr, np.array(target_feature_train_df[target_columns[1]])]
            test_arr2 = np.c_[input_feature_test_arr, np.array(target_feature_test_df[target_columns[1]])]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr1,
                test_arr1,
                train_arr2,
                test_arr2,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)