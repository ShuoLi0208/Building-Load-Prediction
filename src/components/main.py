
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


if __name__=="__main__":
    data_ingestion1 = DataIngestion()
    traint_set, test_set = data_ingestion1.initiate_data_ingestion()

    data_transformation1 = DataTransformation()
    train_df1, test_df1, train_df2, test_df2, preprocessing_path = data_transformation1.initiate_data_transformation(traint_set, test_set)

    model_trainer1 = ModelTrainer()
    model_trainer1.initiate_model_trainer(train_df1, test_df1, train_df2, test_df2, preprocessing_path)