# IRIS PREDICTION API

API REST basata su FastAPI per classificazione del dataset Iris tramite:

* Regressione Logistica Multinomiale
* K-Nearest Neighbors (KNN)

(+ interfaccia web minimale per input manuale)

---

## Accesso

* API docs: http://localhost:8079/docs
* UI web: http://localhost:8079/

---

## API Endpoints


### 1. Logistic Regression (Multinomiale)

Basato su trasformazione sigmoide.

**Struttura matematica**

Si parte da una combinazione lineare delle feature.

Si applica la funzione sigmoide:

\sigma(z)=\frac{1}{1+e^{-z}}

L'elemento verrà classificato in base a dove si posiziona sulla curva ( o rispeto alla treshold).

**2. Interpretazione**

* Output = probabilità stimata
* Treshold: soglia

**3. Funzione obiettivo**
Massimizzazione della verosimiglianza → equivalente a minimizzare la **log-loss**:

L'obiettivo originale è la Maximum Likelihood Estimation (MLE), quindi trovare i parametri che massimizzano la probabilità di osservare i dati che abbiamo nel dataset.

Lavorare con i prodotti di probabilità che sono spesso numeri molto piccoli, può risultare complesso. Per questo si usa il logaritmo, che:

1) Trasforma i prodotti in somme, semplificando il calcolo del gradiente.

2) Mantiene i punti di massimo/minimo (poiché funzione monotona crescente).

3) Cambia la prospettiva del problema; aggiungendo il segno meno (Negative Log-Likelihood) si cerca di minimizzare l'errore.

[
L = -\frac{1}{m}\sum [y\log(p) + (1-y)\log(1-p)]
]


```
POST /predict/iris_logistic
```

### 2. KNN

### 1. K-Nearest Neighbors (KNN)

Algoritmo versatile di apprendimento supervisionato non parametrico ( cioè, numero di parametri non è fissato a priori e non c'è una funzione specifica che descriva il comportamento). 

!! Non costruisce un modello esplicito ma memorizza il dataset e esegue previsioni in base al confronto diretto tra i dati. !!

**1. Principio operativo**
Dato un punto (x), si calcola la distanza rispetto a tutti i punti del training set e si selezionano i k-valori più vicini.
La classe è determinata per maggioranza (nella classificazione) o media (nella regressione).

**2. Distanza**
La più comune ed usata è la distanza euclidea:

d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}

Ci sono alternative come ad esempio la Manhattan Distance:

Dati due punti (P1, P2) in uno spazio bidimensionale, la distanza di Manhattan è la somma dei valori assoluti (moduli) delle differenze delle loro coordinate.
(è sempre maggiore o uguale rispetto alla distanza euclidea)

**3. Scelta di (k)**

### 1. k piccolo (es. k=1)

Quando k è molto basso, il modello si basa solo sul primo vicino, risutlando quindi sensibile alla presenza di eventuali outliers.

**Risultato:** Errore zero sul training set, ma pessime performance su test set.

### 2. k grande (es. k=101)

Quando k è molto alto, il modello diventa rigido, non riuscendo più a cogliere sfumature presenti nei dati.

Se k è uguale al numero totale di campioni, l'algoritmo predirrà sempre la classe più frequente nel dataset, a prescindere dalla posizione del punto.

**Risultato:** Errore simile sia su training che su test set.
---

### 3. Ottimizzazione tramite Cross-Validation
Non esiste un valore universale per k (dipende dai dati), quindi per trovare il valore migliore si usa la **K-Fold Cross-Validation**:

1.  **Suddivisione:** Il dataset viene diviso in X parti (fold).
2.  **Iterazione:** Si prova un valore di k (es. k=3). L'algoritmo viene addestrato su X-1 parti e testato sulla parte rimanente. Si ripete questo processo per ogni fold.
3.  **Media:** Si calcola l'errore medio per quel k.
4.  **Confronto:** Si ripete il tutto per diversi valori di k.

**Obiettivo:** Trovare il l'elbow point nel grafico dell'errore, dove l'errore sul set di validazione è minimo. (derivata = 0)

---

**Note**

* **Numeri dispari:** Si usa quasi sempre un k dispari per evitare situazioni di parità nelle votazioni tra due classi.
* **Regola della radice:** Una regola empirica comune per iniziare è impostare k = \sqrt{n}, dove n è il numero di campioni nel dataset.
* **Scaling:** Prima di scegliere k, si deve **normalizzare o standardizzare** i dati. (Se una feature ha una scala più grande, dominerà il calcolo della distanza, rendendo vana qualsiasi scelta di k)


```
POST /predict/iris_KNN
```

### Response

```
{
  "model": "iris_logistic | iris_KNN",
  "iris_prediction": int
}
```

---

## I/O

* Input: vettore 4D
* Output: classe {0,1,2}


