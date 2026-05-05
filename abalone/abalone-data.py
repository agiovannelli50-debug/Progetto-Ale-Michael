import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from ucimlrepo import fetch_ucirepo
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


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


# --- ANALISI DATASET ---
print("Colonne dataset:\n",df.columns,"\n")
print("Prime righe dataset:\n",df.head(),"\n")
print("Shape dataset:\n",df.shape,"\n")
print("Info dataset:\n",df.info,"\n")
print("Statistiche dataset:\n",df.describe(),"\n")


### ONE HOT ENCODING 'Sex'
# (drop della prima per evitare multicollinearità con altre dummy columns)

df = pd.get_dummies(df, columns=['Sex'], dtype=int, drop_first=True)

### MATRICE CORRELAZIONE & HEATMAP
#  Per trovare pattern tra variabili diverse, heatmap su dati normalizzati è quasi sempre scelta corretta.
#  Mostrare un report finale su grandezze reali ---> heatmap con valori normali

correlation = df.corr()
# print("\nMatrice di Correlazione:\n", correlation,"\n")

plt.figure(figsize=(10,10))
sns.heatmap(correlation,
            annot=True,
            cmap="vlag",
            cbar=True,
            square=True ,
            annot_kws={"size": 8},
            fmt=".1f"              ## fmt=".1f" = 1 cifra dopo la virgola
        )

plt.show()

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
scaler = RobustScaler()
df_fit2 = scaler.fit_transform(df)
# Trasformiamo l'array di nuovo in un DataFrame per mantenere i nomi delle colonne
df_norm2 = pd.DataFrame(df_fit2, columns=df.columns)

### NUOVA HEATMAP
plt.figure(figsize=(12, 10))
sns.heatmap(df_norm2.corr(), annot=True, cmap="vlag", fmt=".2f")
plt.title("Heatmap dopo Feature Engineering e Normalizzazione")
plt.show()

# SPLITTING DATASET
X_final = df_norm2
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)




### MODEL TRAINING

models = [
    ("Linear Regression", LinearRegression()),
    ("Polynomial (Grado 2)", make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())),
    ("Lasso (L1)", Lasso()),
    ("Ridge (L2)", Ridge()),
    ("Random Forest", RandomForestRegressor(random_state=42)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.01,random_state=42)),
    ("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.01,random_state=42))
]

'''
### VISUALIZZAZIONE CONFUSION MATRIX

plt.figure(figsize=(10, 7))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Greens') # Colore diverso per il vincitore
plt.title(f"Matrice di Confusione del Miglior Modello: {best_model_name} (MAE: {best_mae:.3f})")
plt.xlabel('Valori PREDETTI (Anelli)', fontsize=12, fontweight='bold')
plt.ylabel('Valori REALI (Anelli)', fontsize=12, fontweight='bold')
plt.show()
'''



### Neural network Model

model = Sequential()
n_cols = df_norm2.shape[1]

model.add(Input(shape=(n_cols,)))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1))

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train.values.ravel(), epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

predictions = model.predict(X_test).flatten()
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"{'Neural Network':<20} | R2: {r2:>6.3f} | MAE: {mae:>6.3f}")




### SALVATAGGIO MODELLI

# Dizionario per salvare tutti i modelli sklearn/XGBoost addestrati
trained_models = {}

print(f"\n{'Modello':<20} | {'R2 Score':<10} | {'MAE':<10}")
print("-" * 45)

for name, sklearn_model in models:  # ← CAMBIATO: sklearn_model invece di model
    sklearn_model.fit(X_train, y_train.values.ravel())
    predictions = sklearn_model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Salva il modello addestrato nel dizionario
    trained_models[name] = sklearn_model

    print(f"{name:<20} | R2: {r2:>6.3f} | MAE: {mae:>6.3f}")


# Salva tutti i modelli sklearn/XGBoost in un unico file pickle
joblib.dump(trained_models, 'abalone_sklearn_models.pkl')
print("\n✓ Modelli sklearn/XGBoost salvati in 'abalone_sklearn_models.pkl'")

# Ora 'model' contiene ancora la rete neurale Keras
model.save('abalone_neural_network.keras')
print("✓ Rete neurale salvata in 'abalone_neural_network.keras'")

# Salva anche lo scaler
joblib.dump(scaler, 'abalone_scaler.pkl')
print("✓ Scaler salvato in 'abalone_scaler.pkl'")