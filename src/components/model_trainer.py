# Basic Import
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
# Modelling
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    train_model_file_path1:str = os.path.join("artifacts", "model1.pkl")
    train_model_file_path2:str = os.path.join("artifacts", "model2.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_df1, test_df1, train_df2, test_df2, preprocessing_path):

        try:
            logging.info("Load the preprocessing.pkl file")
            
            preprocessing = load_object(preprocessing_path)

            logging.info("Get input and output variables")

            # target y1
            X_train1, y_train1, X_test1, y_test1 = (
                train_df1.iloc[:, :-1], 
                train_df1.iloc[:, -1], 
                test_df1.iloc[:, :-1], 
                test_df1.iloc[:, -1],
            )

            # target y2
            X_train2, y_train2, X_test2, y_test2 = (
                train_df2.iloc[:, :-1], 
                train_df2.iloc[:, -1], 
                test_df2.iloc[:, :-1], 
                test_df2.iloc[:, -1],
            )
            
            # create a dictionary of different models
            models = {
                "linear_regression": LinearRegression(),
                "lasso": Lasso(),
                "ridge": Ridge(),
                "elastic_net": ElasticNet(),
                "support_vector_regressor": SVR(),
                "random_forest_regressor": RandomForestRegressor(random_state=42),
            }

            # create a list that stores all the hyper-parameters for all the models
            param_grids = [
                {},
                {"lasso__alpha": [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]},
                {"ridge__alpha": [0.1, 0.3, 1, 2, 5, 10, 20, 50, 100, 200]},
                {"elastic_net__alpha": [0.1, 0.3, 1, 3, 10],
                "elastic_net__l1_ratio": [1, 3, 10, 30, 100]},
                {"svr__kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                "svr__C": [0.1, 1, 10, 100],
                "svr__epsilon": [0.01, 0.1, 1]},
                {"random_forest__n_estimators": [10, 30, 100, 300],
                "random_forest__max_depth": [3, 4, 5, 6],
                "random_forest__min_samples_split": [2, 3, 4, 5],
                "random_forest__max_features": ['sqrt', 'log2']},
            ]


            # model training
            model_list = []
            final_model1_list = []
            final_model2_list = []
            mape1_list =[]
            mape2_list = []

            for i in range(len(list(models))):
                model = list(models.values())[i]
                param_grid = param_grids[i]
                if len(param_grid.keys())==0:
                    model_name = "reg"
                else:
                    first_hyperparam = list(param_grid.keys())[0]
                    model_name = first_hyperparam.split("__")[0] # get the model name so that scikit-learn pipeline can recognize the hyperparams

                # build the full pipeline fropm preprocessing to model training
                full_pipeline = Pipeline([
                    ('preprocessing', preprocessing),
                    (model_name, model)
                ])

                # hyperparameter tuning using GridSearch cross-validation
                grid_search = GridSearchCV(full_pipeline, param_grid=param_grid, cv=5)

                # Train model and make predictions for y1
                grid_search.fit(X_train1, y_train1) # Train model
                final_model1 = grid_search.best_estimator_ # include pre-processing
                final_model1_list.append(final_model1) 
                y_train1_pred = grid_search.predict(X_train1)
                y_test1_pred = grid_search.predict(X_test1)

                # Train model and make predictions for y2
                grid_search.fit(X_train2, y_train2) # Train model
                final_model2 = grid_search.best_estimator_ # include pre-processing
                final_model2_list.append(final_model2) 
                y_train2_pred = grid_search.predict(X_train2)
                y_test2_pred = grid_search.predict(X_test2)
                
                # Evaluate Train and Test dataset
                model_train1_mae, model_train1_rmse, model_train1_mape = evaluate_model(y_train1, y_train1_pred)

                model_test1_mae, model_test1_rmse, model_test1_mape = evaluate_model(y_test1, y_test1_pred)

                model_train2_mae, model_train2_rmse, model_train2_mape = evaluate_model(y_train2, y_train2_pred)

                model_test2_mae, model_test2_rmse, model_test2_mape = evaluate_model(y_test2, y_test2_pred)
                
                print(list(models.keys())[i])
                model_list.append(list(models.keys())[i])
                
                print('Model performance (heating load) for Training set')
                print("- Root Mean Squared Error: {:.4f}".format(model_train1_rmse))
                print("- Mean Absolute Error: {:.4f}".format(model_train1_mae))
                print("- Mean Absolute Percentage Error: {:.4f}".format(model_train1_mape))

                print('----------------------------------')
                
                print('Model performance (heating load) for Test set')
                print("- Root Mean Squared Error: {:.4f}".format(model_test1_rmse))
                print("- Mean Absolute Error: {:.4f}".format(model_test1_mae))
                print("- Mean Absolute Percentage Error: {:.4f}".format(model_test1_mape))
                mape1_list.append(model_test1_mape)

                print('----------------------------------')

                print('Model performance (cooling load) for Training set')
                print("- Root Mean Squared Error: {:.4f}".format(model_train2_rmse))
                print("- Mean Absolute Error: {:.4f}".format(model_train2_mae))
                print("- Mean Absolute Percentage Error: {:.4f}".format(model_train2_mape))

                print('----------------------------------')
                
                print('Model performance (cooling load) for Test set')
                print("- Root Mean Squared Error: {:.4f}".format(model_test2_rmse))
                print("- Mean Absolute Error: {:.4f}".format(model_test2_mae))
                print("- Mean Absolute Percentage Error: {:.4f}".format(model_test2_mape))
                mape2_list.append(model_test2_mape)

                print('='*35)
                print('\n')
                
                
            # save the model that produces the smallest MAPE in the test set
            final_model_y1 = final_model1_list[mape1_list.index(min(mape1_list))]
            print("The best model for heating load prediction: ", list(models.values())[mape1_list.index(min(mape1_list))])
            final_model_y2 = final_model2_list[mape2_list.index(min(mape2_list))]
            print("The best model for cooling load prediction: ", list(models.values())[mape2_list.index(min(mape2_list))])

            logging.info("Best models for the testing dadaset are found")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path1,
                obj=final_model_y1,
            )

            save_object(
                file_path=self.model_trainer_config.train_model_file_path2,
                obj=final_model_y2,
            )

            logging.info("Final models (including preprocessing) are saved")

            return (
                min(mape1_list), 
                min(mape2_list),
            )

        except Exception as e:
            raise CustomException(e, sys)