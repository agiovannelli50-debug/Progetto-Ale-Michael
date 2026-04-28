# 🌸 Iris Prediction API

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

REST API ad alte prestazioni sviluppata con **FastAPI** per la classificazione del dataset Iris. Il sistema integra due diverse metodologie di Machine Learning e fornisce un'interfaccia web intuitiva per l'interazione manuale.

## 🚀 Accesso Rapido

Una volta avviato il servizio, le risorse saranno disponibili ai seguenti indirizzi:

* **Interactive API Docs (Swagger UI):** [http://localhost:8079/docs](http://localhost:8079/docs)
* **Web User Interface:** [http://localhost:8079/](http://localhost:8079/)

---

## 🧠 Modelli Implementati

L'API permette di confrontare i risultati tra un approccio parametrico e uno non parametrico.

### 1. Logistic Regression (Multinomiale)
Modello probabilistico basato sulla massimizzazione della verosimiglianza.

* **Logica Matematica:** Trasformazione lineare delle feature seguita dall'applicazione della funzione **Sigmoide**:
    $$\sigma(z)=\frac{1}{1+e^{-z}}$$
* **Ottimizzazione:** Il modello minimizza la **Log-Loss** (Negative Log-Likelihood) per allineare le probabilità stimate alle etichette reali:
    $$L = -\frac{1}{m}\sum_{i=1}^{m} [y^{(i)}\log(p^{(i)}) + (1-y^{(i)})\log(1-p^{(i)})]$$
* **Output:** Probabilità stimata per ogni classe con classificazione basata su soglia (*threshold*).

### 2. K-Nearest Neighbors (KNN)
Algoritmo di apprendimento supervisionato **non parametrico** che esegue previsioni basate sulla similarità locale.

* **Principio:** Classifica un punto $x$ analizzando i $k$ campioni più vicini nel training set.
* **Metriche di Distanza:**
    * **Euclidea:** $d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}$ (Standard).
    * **Manhattan:** $d(x,y)=\sum_{i=1}^{n}|x_i-y_i|$ (Robusta agli outlier).
* **Ottimizzazione del Parametro $k$:**
    Implementata tramite **K-Fold Cross-Validation** per identificare l'**Elbow Point** (punto di equilibrio tra bias e varianza).
    > **Best Practice:** Utilizzo di $k$ dispari per evitare pareggi e pre-processamento tramite **Feature Scaling** (Standardizzazione) obbligatorio.

---

## 🔌 Specifiche Endpoints

### Predizione via Logistic Regression
`POST /predict/iris_logistic`

### Predizione via KNN
`POST /predict/iris_KNN`

**Corpo della Richiesta (JSON):**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Esempio di Risposta:**
```json
{
  "model": "iris_KNN",
  "iris_prediction": 0,
  "class_name": "setosa",
  "timestamp": "2026-04-28T10:45:00Z"
}
```

---

## 🛠 Setup e Installazione

### Requisiti
* Python 3.9+ o Docker

### Installazione Locale
1. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
2. Avvia il server:
   ```bash
   uvicorn app.app:app --host 0.0.0.0 --port 8079
   ```

### Distribuzione con Docker
```bash
docker build -t iris-api .
docker run -p 8079:8079 iris-api
```

---

## 📂 Struttura del Progetto
```text
/app
├── app.py                # Logica FastAPI e routing
├── index.html            # Interfaccia UI minimale
├── iris_model.joblib     # Pesi Logistic Regression
├── iris_model_knn.joblib # Dataset indicizzato KNN
├── requirements.txt      # Dipendenze del progetto
└── Dockerfile            # Containerizzazione
```

---
*Developed for Iris Classification Analysis - 2026*
