from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import HTMLResponse
from tensorflow import keras

app = FastAPI(title="MUSHROOM CLASSIFICATION API")

# Carica i Modelli
xgb_model = joblib.load("xgboost_model.pkl")
log_reg_model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
nn_model = keras.models.load_model("neural_network_model.keras")


# Carica pagina html(scelta opzionale fatto per visualizzare meglio la pagina web
@app.get("/", response_class=HTMLResponse)
def main_page():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html non trovato</h1>"


# Classe per la validazione dei dati - Contiene Features Modello
## Importante: L'ordine DEVE essere lo stesso nel dataset
class MushroomFeatures(BaseModel):
    profilo_cappello: int
    profilo_lamelle: int
    profilo_gambo: int
    profilo_anello: int
    profilo_ambientale: int
    profilo_odore_e_colore_spore: int


# Creazione Matrice con Features -- Anche qui DEVE essere in ordine con il dataset
def to_array(data: MushroomFeatures):
    return np.array([[
        data.profilo_cappello,
        data.profilo_lamelle,
        data.profilo_gambo,
        data.profilo_anello,
        data.profilo_ambientale,
        data.profilo_odore_e_colore_spore
    ]])


#Modello XGBoost
@app.post("/predict/xgboost")
def predict_xgboost(data: MushroomFeatures):

    features = to_array(data)
    prediction = xgb_model.predict(features)[0]

    return {
        "model": "xgboost",
        "prediction": int(prediction)
    }


#Modello Logistic Regression
@app.post("/predict/logistic")
def predict_logistic(data: MushroomFeatures):

    features = to_array(data)
    features_scaled = scaler.transform(features)

    prediction = log_reg_model.predict(features_scaled)[0]

    return {
        "model": "logistic_regression",
        "prediction": int(prediction)
    }


#Modello Deep Neural Network
@app.post("/predict/neural_network")
def predict_nn(data: MushroomFeatures):

    features = to_array(data)

    prediction = nn_model.predict(features, verbose=0)[0][0]
    prediction = int(prediction > 0.5)

    return {
        "model": "neural_network",
        "prediction": prediction
    }