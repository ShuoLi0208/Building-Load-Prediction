# End-to-end Machine Learning Project on Building Load Prediction
The task is to perform building heating and cooling load prediction using Machine Learning algorithms including linear regression, support vector machine and random forest. All of the data pre-processing and model training and prediction tools are imported from scikit-learn. The design of the project structure follows the github repo [mlproject](https://github.com/krishnaik06/mlproject) by Krish Naik. If you are interested in implementing this end-to-end machine learning project step by step, please folllow the youtude videos (https://www.youtube.com/watch?v=S_F_c9e2bz4&list=PLZoTAELRMXVPS-dOaVbAux22vzqdgoGhG) by Krish Naik.

## Data source
The dataset has 768 examples and 8 features, which can be downloaded from the UCI machine learning repository (https://archive.ics.uci.edu/ml/datasets/Energy+efficiency). Note that the heating and cooling load data are obtained by simulation using software Ecotect, in contrast to measurement by sensors. The 8 features (denoted by X1 to X8) are given as follows:
* X1 Relative Compactness
* X2 Surface Area
* X3 Wall Area
* X4 Roof Area
* X5 Overall Height
* X6 Orientation
* X7 Glazing Area
* X8 Glazing Area Distribution

## Jupyter notebook code
In the folder named **notebook** (here-and-after folder names are in bold), we performed exploratory data analysis and model training, which are given respectively in the EDA.ipynb and model_training.ipynb.

## Modular code
The application.py file at the top-level directory and the predict_pipeline.py file in teh folder **src/pipeline** are used to generate a website using Flask, aiming for performing heating and cooling load prediction with custom user inputs of the 8 features. The major source codes for implementing the machine learning pipeline are in the folder **src/components**, including data_ingestion.py, data_transformation.py and model_trainer.py files. The train and test datasets (in .csv format), and the saved preprocessing and model objects (in .pkl format) are stored in the folder named **artifacts**. Python files logger.py and exception.py files within the source folder (i.e. **src**) are used for logging exception handling, respectively. 

## Deployment using AWS Elastic Beanstalk
See the configuration file i.e. python.config in the folder named **.ebextensions**.
