import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from ucimlrepo import fetch_ucirepo

# fetch dataset
abalone = fetch_ucirepo(id=1)

"""
    The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings.
"""

X = abalone.data.features
y = abalone.data.targets

# Conversione Dataset in Pandas Dataframe
df = pd.DataFrame(X, columns=abalone.feature_names)

# Data Cleaning - Il dataset è già pulito ma controlliamo per sicurezza
df.isnull().sum()

"""
# --- ANALISI DATASET ---
print("Colonne dataset:\n",df.columns,"\n")
print("Prime righe dataset:\n",df.head(),"\n")
print("Shape dataset:\n",df.shape,"\n")
print("Info dataset:\n",df.info,"\n")
print("Statistiche dataset:\n",df.describe(),"\n")
"""

### ONE HOT ENCODING 'Sex'
# (drop della prima per evitare multicollinearità con altre dummy columns)

df = pd.get_dummies(df, columns=['Sex'], dtype=int, drop_first=True)


### NORMALIZZAZIONE DATASET
# Standardizza le feature sottraendo la media e dividendo per la deviazione standard
# z = (x - mean) / std

scaler = StandardScaler()
df_fit = scaler.fit_transform(df)            # calcola la media e dev standard
df_norm = scaler.transform(df)               # applica la trasformazione
#print(df_norm)

### MATRICE CORRELAZIONE & HEATMAP
#  Per trovare pattern tra variabili diverse, heatmap su dati normalizzati è quasi sempre scelta corretta.
#  Mostrare un report finale su grandezze reali  ---> heatmap con valori normali

correlation = df.corr()
# print("\nMatrice di Correlazione:\n", correlation,"\n")

"""plt.figure(figsize=(10,10))
sns.heatmap(correlation,
            annot=True,
            cmap="vlag",
            cbar=True,
            square=True ,
            annot_kws={"size": 8},
            fmt=".1f"              ## fmt=".1f" = 1 cifra dopo la virgola
        )
"""
#plt.show()

### FEATURE ENGINEERING
# Dalla heatmap le feature sono correlate tantissimo tra loro (>0.9)

# CONTROLLO PER DIVISIONE PER 0
df = df[(df['Height'] > 0) & (df['Whole_weight'] > 0) & (df['Diameter'] > 0)].copy()
y = y.loc[df.index] ### df.index restituisce la lista di tutti gli indici sopravvissuti al filtro
                    ### .loc[] prende le righe che hanno un indice presente in df

# INFO SPAZIALI
df["Volume"] = (4/3) * np.pi * (df["Length"]/2) * (df["Diameter"]/2) * (df["Height"]/2)
df["Density"] = df["Whole_weight"] / df["Volume"]
df["Meat_ratio"] = df["Shucked_weight"] / df["Whole_weight"]           ## rapporto tra peso sgusciato e peso totale
df["Viscera_ratio"] = df["Viscera_weight"] / df["Whole_weight"]
df["Shell_ratio"] = df["Shell_weight"] / df["Whole_weight"]

# INFO SULLA FORMA
# Gli abalone giovani tendono ad essere più tondeggianti, quelli maturi più allungati.
df["Shape_ratio"] = df["Length"] / df["Diameter"]
# Fortemente correlato con età e crescita del guscio, gli abalone più vecchi diventano più spessi.
df["Flatness"] = df["Height"] / df["Diameter"]
df["Meat_density"] = df["Shucked_weight"] / df["Volume"]

# DROP colonne originali correlate
df = df.drop(columns=['Length', 'Diameter', 'Height', 'Whole_weight',
                     'Viscera_weight', 'Shucked_weight', 'Shell_weight'])

### NORMALIZZAZIONE DATASET dopo FEATURE ENGINEERING
scaler = StandardScaler()
df_fit2 = scaler.fit_transform(df)
# Trasformiamo l'array di nuovo in un DataFrame per mantenere i nomi delle colonne
df_norm2 = pd.DataFrame(df_fit2, columns=df.columns)

"""### NUOVA HEATMAP
plt.figure(figsize=(12, 10))
sns.heatmap(df_norm2.corr(), annot=True, cmap="vlag", fmt=".2f")
plt.title("Heatmap dopo Feature Engineering e Normalizzazione")
#plt.show()"""

# SPLITTING DATASET
X_final = df_norm2
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

### MODEL TRAINING

models = [
    ("Linear Regression", LinearRegression()),
    ("Polynomial (Grado 2)", make_pipeline(PolynomialFeatures(degree=2), LinearRegression())),
    ("Lasso (L1)", Lasso()),
    ("Ridge (L2)", Ridge()),
    ("Random Forest", RandomForestRegressor(random_state=42)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.01,random_state=42)),
    ("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.01,random_state=42))
]

print(f"{'Modello':<20} | {'R2 Score':<10} | {'MAE':<10}")
print("-" * 45)

best_mae = float('inf')  # Iniziamo con un errore infinito
best_model_name = ""
best_cm = None

for name, model in models:
    model.fit(X_train, y_train.values.ravel())             # .ravel() serve a far vedere una lista e non una colonna
    predictions = model.predict(X_test)

    # Metriche Regressione
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Metriche Classificazione (Arrotondando)
    y_pred_discrete = np.round(predictions).astype(int)
    acc = accuracy_score(y_test, y_pred_discrete)

    # LOGICA PER IL MIGLIORE:
    if mae < best_mae:
        best_mae = mae
        best_model_name = name
        best_cm = confusion_matrix(y_test, y_pred_discrete)

    print(f"{name:<20} | R2: {r2:>6.3f} | MAE: {mae:>6.3f} | Acc: {acc:>6.2%}")

### VISUALIZZAZIONE CONFUSION MATRIX

plt.figure(figsize=(10, 7))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Greens') # Colore diverso per il vincitore
plt.title(f"Matrice di Confusione del Miglior Modello: {best_model_name} (MAE: {best_mae:.3f})")
plt.xlabel('Valori PREDETTI (Anelli)', fontsize=12, fontweight='bold')
plt.ylabel('Valori REALI (Anelli)', fontsize=12, fontweight='bold')
plt.show()

