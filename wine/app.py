#imports
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import HTMLResponse

app = FastAPI(title="WINE PREDICTION")

#carica i modelli
wine_logistic_model = joblib.load("/Users/mykael/PycharmProjects/wine_pred/logistic_regression_model.pkl")
wine_KNN_model = joblib.load("/Users/mykael/PycharmProjects/wine_pred/logistic_regression_model.pkl")

'''
#carica pagina html(scelta opzionale fatto per visualizzare meglio la pagina web
@app.get("/", response_class=HTMLResponse)
def main_page():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Errore: file index.html non trovato nel container</h1>"
'''


@app.get("/")
def home():
    return {"message": "Wine Prediction API is running"}


#classe per la validazione dei dati - contiene features del nostro modello
class WineFeatures(BaseModel):
    alcohol = float
    malic_acid = float
    ash = float
    alcalinity_of_ash = float
    magnesium = int
    total_phenols = float
    flavanoids = float
    nonflavanoid_phenols = float
    proanthocyanins = float
    color_intensity = float
    hue = float
    od280_od315_of_diluted_wines = float
    proline = float



#modello logistico
@app.post("/predict/wine_logistic")
def predict_wine_logistic(data: WineFeatures):
    #creazione matrice con le features
    features = np.array([[data.alcohol, data.malic_acid, data.ash, data.alcalinity_of_ash,
                          data.magnesium, data.total_phenols, data.flavanoids, data.nonflavanoid_phenols,
                          data.proline, data.color_intensity, data.hue, data.od280_od315_of_diluted_wines,data.proanthocyanins]])
    #prende il primo elemento dell'array numpy che viene creato dal metodo predict
    prediction = wine_logistic_model.predict(features)[0]

    return {
        "model": "wine_logistic",
        "wine_prediction": int(prediction)
    }


#modello KNN
@app.post("/predict/iris_KNN")
def predict_wine_KNN(data: WineFeatures):
    features = np.array([[data.alcohol, data.malic_acid, data.ash, data.alcalinity_of_ash,
                          data.magnesium, data.total_phenols, data.flavanoids, data.nonflavanoid_phenols,
                          data.proline, data.color_intensity, data.hue, data.od280_od315_of_diluted_wines,
                          data.proanthocyanins]])
    #invece che usare logistic regression qua usiamo KNN
    prediction = wine_KNN_model.predict(features)[0]

    return {
        "model": "wine_KNN",
        "wine_prediction": int(prediction)
    }