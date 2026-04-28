#imports
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import HTMLResponse

app = FastAPI(title="IRIS PREDICTION")

#carica i modelli
iris_logistic_model = joblib.load("/app/iris_model.joblib")
iris_KNN_model = joblib.load("/app/iris_model_knn.joblib")


#carica pagina html(scelta opzionale fatto per visualizzare meglio la pagina web
@app.get("/", response_class=HTMLResponse)
def main_page():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Errore: file index.html non trovato nel container</h1>"


#classe per la validazione dei dati - contiene features del nostro modello
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


#modello logistico -
@app.post("/predict/iris_logistic")
def predict_iris_logistic(data: IrisFeatures):
    #creazione matrice con le features
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    #prende il primo elemento dell'array numpy che viene creato dal metodo predict
    prediction = iris_logistic_model.predict(features)[0]

    return {
        "model": "iris_logistic",
        "iris_prediction": int(prediction)
    }


#modello KNN
@app.post("/predict/iris_KNN")
def predict_iris_KNN(data: IrisFeatures):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    #invece che usare logistic regression qua usiamo KNN
    prediction = iris_KNN_model.predict(features)[0]

    return {
        "model": "iris_KNN",
        "iris_prediction": int(prediction)
    }