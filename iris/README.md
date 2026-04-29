# 🌸 Iris Prediction API

## Accesso Rapido

* **Interactive API Docs:** [http://localhost:8079/docs](http://localhost:8079/docs)
* **Web User Interface:** [http://localhost:8079/](http://localhost:8079/)

(+ interfaccia web minimale per input manuale)

---

## Modelli Implementati

L'API permette di confrontare i risultati tra un approccio parametrico e uno non parametrico.

### 1. Logistic Regression (Multinomiale)

 a) Per la Regressione Logistica semplice:

* **Logica Matematica:** Si parte da una combinazione lineare delle features seguita dall'applicazione della funzione **Sigmoide**:
  
    $$\sigma(z)=\frac{1}{1+e^{-z}}$$

  L'elemento verrà classificato in base a dove si posiziona sulla curva ( o rispeto alla treshold).

  
* **Ottimizzazione:** L'obiettivo originale è la Maximum Likelihood Estimation (MLE), quindi trovare i parametri che massimizzano la probabilità di osservare i dati che abbiamo nel dataset.

    $$L = -\frac{1}{m}\sum_{i=1}^{m} [y^{(i)}\log(p^{(i)}) + (1-y^{(i)})\log(1-p^{(i)})]$$

Lavorare con i prodotti di probabilità che sono spesso numeri molto piccoli, può risultare complesso. Per questo si usa il logaritmo, che:

1) Trasforma i prodotti in somme, semplificando il calcolo del gradiente.

2) Mantiene i punti di massimo/minimo (poiché funzione monotona crescente).

3) Cambia  prospettiva del problema; aggiungendo il segno meno (Negative Log-Likelihood) si cerca di minimizzare l'errore.

b) Regressione Logistica Multinomiale:

* **Logica Matematica (Softmax):**

    $$\text{softmax}(z)_j=\frac{e^{z_j}}{\sum_{k} e^{z_k}}$$

La funzione softmax è una funzione matematica utilizzata per convertire un vettore di numeri reali in una distribuzione di probabilità.

Mappa i valori in input nell'intervallo e assicura che la somma di tutti gli output sia esattamente 1.

Caratteristiche:
1) Normalizzazione: Garantisce che la somma delle probabilità sia 1
2) Positività: Restituisce solo valori compresi tra 0 e 1
3) Amplificazione: L'esponenziale amplifica le differenze tra i punteggi, rendendo più netto il valore massimo
---

### 2. K-Nearest Neighbors (KNN)
Algoritmo di apprendimento supervisionato **non parametrico** che esegue previsioni basate sulla similarità locale.

* **Principio:** Classifica un punto $x$ analizzando i k-campioni più vicini nel training set.

---

* **Metriche di Distanza:**
  
La più comune ed usata è la Distanza Euclidea:

  $$d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}$$

Ci sono alternative, come ad esempio la Manhattan Distance:

  $$d(x,y)=\sum_{i=1}^{n}|x_i-y_i|$$

Dati due punti (P1, P2) in uno spazio bidimensionale, la distanza di Manhattan è la somma dei valori assoluti (moduli) delle differenze delle loro coordinate. (è sempre maggiore o uguale rispetto alla distanza euclidea)

---

## 1. k piccolo (es. k=1)

Quando k è molto basso, il modello si basa solo sul primo vicino, risutlando quindi sensibile alla presenza di eventuali outliers.

**Risultato:** Errore zero sul training set, ma pessime performance su test set.

## 2. k grande (es. k=101)

Quando k è molto alto, il modello diventa rigido, non riuscendo più a cogliere sfumature presenti nei dati.

Se k è uguale al numero totale di campioni, l'algoritmo predirrà sempre la classe più frequente nel dataset, a prescindere dalla posizione del punto.

**Risultato:** Errore simile sia su training che su test set.

---

## 3. Ottimizzazione tramite Cross-Validation
Non esiste un valore universale per k (dipende dai dati), quindi per trovare il valore migliore si usa la **K-Fold Cross-Validation**:

1.  **Suddivisione:** Il dataset viene diviso in X parti (fold).
2.  **Iterazione:** Si prova un valore di k (es. k=3). L'algoritmo viene addestrato su X-1 parti e testato sulla parte rimanente. Si ripete questo processo per ogni fold.
3.  **Media:** Si calcola l'errore medio per quel k.
4.  **Confronto:** Si ripete il tutto per diversi valori di k.

**Obiettivo:** Trovare il l'elbow point nel grafico dell'errore, dove l'errore sul set di validazione è minimo. (derivata = 0)

---

## Endpoints

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

## Setup

### Docker
```bash
docker build -t iris-api .
docker run -p 8079:8000 iris-api
```

---

## Struttura del Progetto
```text
/app
├── app.py                # Logica FastAPI e routing
├── index.html            # Interfaccia UI 
├── iris_model.joblib     # Pesi Logistic Regression
├── iris_model_knn.joblib # Pesi KNN
├── requirements.txt      # Dipendenze
└── Dockerfile            # Containerizzazione
```
