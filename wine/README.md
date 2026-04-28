# 🍷 Wine Prediction API

## Accesso Rapido

* **Interactive API Docs:** [http://localhost:8098/docs](http://localhost:8079/docs)  
* **Web User Interface:** [http://localhost:8098/](http://localhost:8079/)

(+ interfaccia web minimale per input manuale)

---

## Modelli Implementati

L'API permette di confrontare i risultati tra un approccio parametrico e uno non parametrico.

### 1. Logistic Regression (Multinomiale)

Estensione della regressione logistica a più classi tramite softmax.

* **Logica Matematica (Softmax):**

    $$\text{softmax}(z)_j=\frac{e^{z_j}}{\sum_{k} e^{z_k}}$$

  L'elemento viene classificato in base alla probabilità massima tra le classi.

* **Ottimizzazione:** Si massimizza la Likelihood; si minimizza la Negative Log-Likelihood (NLL) per la classificazione multi-classe.

---

### 2. K-Nearest Neighbors (KNN)
Algoritmo di apprendimento supervisionato **non parametrico** che esegue previsioni basate sulla similarità locale.

* **Principio:** Classifica un punto $x$ analizzando i k-campioni più vicini nel training set.

---

* **Metriche di Distanza:**

La più comune è la Distanza Euclidea:

  $$d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}$$

Alternativa: Manhattan Distance

  $$d(x,y)=\sum_{i=1}^{n}|x_i-y_i|$$

---

* **Scelta del Parametro k:**

## 1. k piccolo (es. k=1)

Sensibile agli outlier — errore basso sul training ma possibile overfitting.

## 2. k grande (es. k elevato)

Modello troppo rigido; tende a prevedere la classe più frequente se k → N.

---

## 3. Ottimizzazione tramite Cross-Validation
Usare K-Fold CV per selezionare k o altri iperparametri: suddividere i dati, allenare su X-1 fold, validare sul fold rimanente e mediare le prestazioni.

---

## Endpoints

### Predizione via Logistic Regression
`POST /predict/wine_logistic`

### Predizione via KNN
`POST /predict/wine_KNN`

**Corpo della Richiesta (JSON):**
```json
{
  "alcohol": 13.2,
  "malic_acid": 2.14,
  "ash": 2.67,
  "alcalinity_of_ash": 18.6,
  "magnesium": 101,
  "total_phenols": 2.8,
  "flavanoids": 3.14,
  "nonflavanoid_phenols": 0.28,
  "proanthocyanins": 1.5,
  "color_intensity": 5.0,
  "hue": 1.05,
  "od280_od315": 3.0,
  "proline": 985
}
```

**Esempio di Risposta:**
```json
{
  "model": "wine_KNN",
  "wine_prediction": 1,
  "class_name": "class_1",
  "timestamp": "2026-04-28T10:45:00Z"
}
```

---

## Setup

### Distribuzione con Docker
```bash
docker build -t wine-api .
docker run -p 8079:8000 wine-api
```

---

## Struttura del Progetto
```text
/app
├── app.py                  # Logica FastAPI e routing
├── index.html              # Interfaccia UI minimale
├── wine_model.joblib       # Pesi Logistic Regression (multinomiale)
├── wine_model_knn.joblib   # Dataset indicizzato / oggetto KNN
├── requirements.txt        # Dipendenze del progetto
└── Dockerfile              # Containerizzazione
```
