# copy from app.py but remove debug=True
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

"""
Route for a home page
"""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
    else:
        data = CustomData(
            relative_compactness=float(request.form.get("relative_compactness")),
            surface_area=float(request.form.get("surface_area")),
            wall_area=float(request.form.get("wall_area")),
            roof_area=float(request.form.get("roof_area")),
            overall_height=float(request.form.get("overall_height")),
            orientation=int(request.form.get("orientation")),
            glazing_area=float(request.form.get("glazing_area")),
            glazing_area_distribution=int(request.form.get("glazing_area_distribution")),
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        pred1, pred2 = predict_pipeline.predict(pred_df)

        return render_template("home.html", heating_load=pred1[0], cooling_load=pred2[0])


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)