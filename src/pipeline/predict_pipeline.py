import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path1 = "artifacts/model1.pkl" # include preprocessing (1 denotes heating load)
            model_path2 = "artifacts/model2.pkl" # include preprocessing (2 denotes cooling load)
            model1 = load_object(file_path=model_path1)
            model2 = load_object(file_path=model_path2)

            pred1 = model1.predict(features)
            pred2 = model2.predict(features)

            return pred1, pred2
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(self,
        relative_compactness: float,
        surface_area: float,
        wall_area: float,
        roof_area: float,
        overall_height: float,
        orientation: int,
        glazing_area: float,
        glazing_area_distribution: int):

        self.relative_compactness = relative_compactness
        self.surface_area = surface_area
        self.wall_area = wall_area
        self.roof_area = roof_area
        self.overall_height = overall_height
        self.orientation = orientation
        self.glazing_area = glazing_area
        self.glazing_area_distribution = glazing_area_distribution

    def get_data_as_dataframe(self):

        try:
            custom_data_input_dict = {
                "X1": [self.relative_compactness],
                "X2": [self.surface_area],
                "X3": [self.wall_area],
                "X4": [self.roof_area],
                "X5": [self.overall_height],
                "X6": [self.orientation],
                "X7": [self.glazing_area],
                "X8": [self.glazing_area_distribution],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
