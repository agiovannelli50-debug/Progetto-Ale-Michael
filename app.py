from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="IRIS PREDICTION")

# Load models from app directory
iris_logistic_model = joblib.load("/app/iris_model.joblib")
iris_KNN_model = joblib.load("/app/iris_model_knn.joblib")


@app.get("/")
def home():
    return {"message": "IRIS API is running"}


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


#logistic_model
@app.post("/predict/iris_logistic")
def predict_iris_logistic(data: IrisFeatures):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = iris_logistic_model.predict(features)[0]

    return {
        "model": "iris_logistic",
        "iris_prediction": int(prediction)
    }


#KNN model
@app.post("/predict/iris_KNN")
def predict_iris_KNN(data: IrisFeatures):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = iris_KNN_model.predict(features)[0]

    return {
        "model": "iris_KNN",
        "iris_prediction": int(prediction)
    }