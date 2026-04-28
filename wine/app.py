#imports
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import HTMLResponse

app = FastAPI(title="WINE PREDICTION")

# Carica i Modelli
wine_logistic_model = joblib.load("logistic_regression_model.pkl")
wine_KNN_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")           ## necessario perché usato nel training

# Carica pagina html(scelta opzionale fatto per visualizzare meglio la pagina web
@app.get("/", response_class=HTMLResponse)
def main_page():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Errore: file index.html non trovato nel container</h1>"


# Classe per la validazione dei dati - Contiene Features Modello
## Importante: L'ordine DEVE essere lo stesso nel dataset
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float



# Modello Logistico
@app.post("/predict/wine_logistic")
def predict_wine_logistic(data: WineFeatures):
    # Creazione Matrice con Features -- Anche qui DEVE essere in ordine con il dataset
    features = np.array([[
        data.alcohol,
        data.malic_acid,
        data.ash,
        data.alcalinity_of_ash,
        data.magnesium,
        data.total_phenols,
        data.flavanoids,
        data.nonflavanoid_phenols,
        data.proanthocyanins,
        data.color_intensity,
        data.hue,
        data.od280_od315,
        data.proline
    ]])

    features_scaled = scaler.transform(features)    ## applica lo scaling delle features ricevute

    # Prende il primo elemento dell'array numpy che viene creato dal metodo predict
    prediction = wine_logistic_model.predict(features_scaled)[0]

    return {
        "model": "wine_logistic",
        "wine_prediction": int(prediction)
    }


# Modello KNN
@app.post("/predict/wine_KNN")
def predict_wine_KNN(data: WineFeatures):
    features = np.array([[
        data.alcohol,
        data.malic_acid,
        data.ash,
        data.alcalinity_of_ash,
        data.magnesium,
        data.total_phenols,
        data.flavanoids,
        data.nonflavanoid_phenols,
        data.proanthocyanins,
        data.color_intensity,
        data.hue,
        data.od280_od315,
        data.proline
    ]])

    features_scaled = scaler.transform(features)
    # Invece che usare logistic regression qua usiamo KNN
    prediction = wine_KNN_model.predict(features_scaled)[0]

    return {
        "model": "wine_KNN",
        "wine_prediction": int(prediction)
    }