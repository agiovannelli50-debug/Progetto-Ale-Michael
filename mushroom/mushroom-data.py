from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, \
    recall_score
import joblib
from xgboost import XGBClassifier
import warnings

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# fetch dataset
mushroom = fetch_ucirepo(id=73)

# data (as pandas dataframes)
X = mushroom.data.features

y = mushroom.data.targets
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# Conversione Dataset in Pandas Dataframe (NON encodato)
df = pd.DataFrame(X, columns=mushroom.feature_names)

# Data Cleaning - Il dataset è già pulito ma controlliamo per sicurezza
print(df.isnull().sum())
df["stalk-root"] = df["stalk-root"].fillna("Unknown")



### ANALISI DATASET


print("\n" + "=" * 60)
print("ANALISI DATASET")
print("=" * 60)
print(f"Colonne dataset: {list(df.columns)}\n")
print("Prime righe dataset:")
print(df.head())
print(f"\nShape dataset: {df.shape}")
print(f"Numero feature: {len(df.columns)}")
print("\nDistribuzione target (y):")
print(y.value_counts())
print()



'''### MATRICE CORRELAZIONE & HEATMAP
print("Generazione heatmap correlazione...")
correlation = df.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation,
            annot=True,
            cmap="vlag",
            cbar=True,
            square=True ,
            annot_kws={"size": 8},
            fmt=".1f"              ## fmt=".1f" = 1 cifra dopo la virgola
        )

plt.title("Heatmap Correlazione Feature (dopo Label Encoding)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('mushroom_correlation_heatmap.png', dpi=100, bbox_inches='tight')
plt.close()
print("  ✓ Salvata come 'mushroom_correlation_heatmap.png'\n")
'''




### FEATURE ENGINEERING


# Drop veil perché ha un solo valore in dataset, si puo verificare anche dalla heatmap
df = df.drop('veil-type', axis=1)

# Combinazione feature del cappello in un unico feature cap_profile
# Il cappello è uno dei principali indicatori di specie.
df["profilo_cappello"] = (
    df["cap-shape"] + "_" +
    df["cap-surface"] + "_" +
    df["cap-color"]
)

# Combinazione feature delle lamelle in un unico feature gill_profile
# Le lamelle sono fortemente correlate alla tossicità.
df["profilo_lamelle"] = (
    df["gill-attachment"] + "_" +
    df["gill-spacing"] + "_" +
    df["gill-size"] + "_" +
    df["gill-color"]
)

# # Combinazione feature del ganbo in un unico feature stalk_profile
df["profilo_gambo"] = (
    df["stalk-shape"] + "_" +
    df["stalk-root"] + "_" +
    df["stalk-surface-above-ring"] + "_" +
    df["stalk-surface-below-ring"] + "_" +
    df["stalk-color-above-ring"] + "_" +
    df["stalk-color-below-ring"]
)

# Creazione feature ring_profile
df["profilo_anello"] = df["ring-number"] + "_" + df["ring-type"]

# Dove cresce il fungo è molto indicativo.
df["profilo_ambientale"] = df["population"] + "_" + df["habitat"]

# Odore + colore delle spore = quasi classificazione perfetta.
df["profilo_odore_e_colore_spore"] = df["odor"] + "_" + df["spore-print-color"]

# DROP colonne originali correlate
df = df.drop(columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                      'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                      'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                      'stalk-surface-below-ring', 'stalk-color-above-ring',
                      'stalk-color-below-ring', 'veil-color',
                      'ring-number', 'ring-type', 'spore-print-color',
                      'population', 'habitat'])

# Encoding finale DOPO la feature engineering
df = df.apply(LabelEncoder().fit_transform)
print("\n",df.head())



'''### MATRICE CORRELAZIONE & HEATMAP DOPO FEATURE ENGINEERING E LABEL ENCODING
print("Generazione heatmap correlazione...")
correlation = df.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation,
            annot=True,
            cmap="vlag",
            cbar=True,
            square=True ,
            annot_kws={"size": 8},
            fmt=".1f"              ## fmt=".1f" = 1 cifra dopo la virgola
        )

plt.title("Heatmap Correlazione Feature (dopo Feature Engineering eLabel Encoding)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('mushroom_correlation_heatmap_2.png', dpi=100, bbox_inches='tight')
plt.close()
print("  ✓ Salvata come 'mushroom_correlation_heatmap_2.png'\n")

'''

# SPLITTING DATASET
X_train, X_test, y_train, y_test = train_test_split(df, y_encoded, test_size=0.2,random_state=42,
                                   stratify=y_encoded)



### MODEL TRAINING


print("TRAINING MODELLI")
print("=" * 60)

models = [
    ("XGBoost", XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,random_state=42,
        use_label_encoder=False, eval_metric="logloss")),

    ("Logistic Regression", LogisticRegression(max_iter=1000,random_state=42)),
]

# Scaling needed for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training loop
for name, model in models:
    print(f"\n{name}")

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))




### NEURAL NETWORK MODEL


print("\n" + "-" * 85)

model = Sequential()
n_cols = df.shape[1]

model.add(Input(shape=(n_cols,)))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))  # Sigmoid per classificazione binaria

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

#Binary crossentropy is used for binary classification problems
#It compares the true label (0 or 1) to the predicted probability (between 0 and 1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train,epochs=100,batch_size=32,validation_split=0.2,callbacks=[early_stop],
        verbose=0)


# Predizioni Neural Network
predictions_nn = (model.predict(X_test, verbose=0).flatten() > 0.5).astype(int)
accuracy_nn = accuracy_score(y_test, predictions_nn)
precision_nn = precision_score(y_test, predictions_nn, average='weighted')
recall_nn = recall_score(y_test, predictions_nn, average='weighted')
f1_nn = f1_score(y_test, predictions_nn, average='weighted')

print(f"{'Model':<25} | {'Accuracy':>8} | {'Precision':>8} | {'Recall':>8} | {'F1-score':>8}")
print("-" * 70)

print(f"{'Neural Network':<25} | {accuracy_nn:>8.4f} | {precision_nn:>8.4f} | {recall_nn:>8.4f} | {f1_nn:>8.4f}")





### SALVATAGGIO MODELLI
print("\n" + "=" * 60)


# Salvare XGBoost
xgb_model = models[0][1]   # XGBoost dal list models
joblib.dump(xgb_model, "xgboost_model.pkl")
print("XGBoost salvato: xgboost_model.pkl")

# Salvare Logistic Regression
log_reg = models[1][1]     # Logistic Regression dal list models
joblib.dump(log_reg, "logistic_regression_model.pkl")
print("Logistic Regression salvata: logistic_regression_model.pkl")

# Salvare scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler salvato: scaler.pkl")

# Salvare Neural Network (Keras)
model.save("neural_network_model.keras")
print("Neural Network salvata: neural_network_model.keras")



'''
df["target"] = y_encoded
#print(df)

# Per vedere quali funghi nel dataset sono commestibili
print("\nDateset funghi commestibili:\n",df[df["target"] == 0])

# Per vedere quali funghi nel dataset sono velenosi
print("\nDataset funghi velenosi:\n ",df[df["target"] == 1])
'''