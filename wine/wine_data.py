import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Caricamento dati
wine = load_wine()


# Separazione Target da Feature
X = wine.data
y = wine.target


# Conversione Dataset in Pandas Dataframe
df = pd.DataFrame(X, columns=wine.feature_names)


# Data Cleaning - Il dataset è già pulito ma controlliamo per sicurezza
print(df.isnull().sum(),"\n")

"""
# --- ANALISI DATASET ---
print("Colonne dataset:\n",df.columns,"\n")
print("Prime righe dataset:",df.head(),"\n")
print("Shape dataset:",df.shape,"\n")
print("Info dataset:",df.info,"\n")
print("Statistiche dataset:",df.describe(),"\n")

"""

# Matrice di Correlazione
correlation = df.corr()
print("\nMatrice di Correlazione:\n", correlation,"\n")

"""
# Heatmap
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
"""

# Splitting di Train Data and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
print("\n")
print(X.shape,X_train.shape,X_test.shape)
print("\n")


#Feature Scaling - Importante per Logistic Regression e KNN
# Standardizza le feature sottraendo la media e dividendo per la deviazione standard
# z = (x - mean) / std
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)            # calcola la media e dev standard
X_test_scaled = scaler.transform(X_test)                  #applica la trasformazione

print(X_test_scaled)




"""
# KNN - k=13 perché sqrt di num di righe del dataset
knn_model = KNeighborsClassifier(n_neighbors=13)
knn_model.fit(X_train_scaled, y_train)

# Prediction
knn_y_pred = knn_model.predict(X_test_scaled)

# Accuracy e Classification Report
print("\nKNN Accuracy Score:", accuracy_score(y_test, knn_y_pred),"\n")
print("\nKNN Classification Report:\n", classification_report(y_test, knn_y_pred),"\n")



# Modello Logistic Regression
model = LogisticRegression(solver='lbfgs', random_state=2, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Accuracy e Classification Report
print("\nLogistic Regression Accuracy Score:", accuracy_score(y_test, y_pred),"\n")
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred),"\n")



# Confusion Matrix per entrambi i modelli -- Mostra actual values vs predicted di ogni classe
plt.figure(figsize=(12, 5))

# Matrice di confusione -- KNN
plt.subplot(1, 2, 1)
knn_cm = confusion_matrix(y_test, knn_y_pred)
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])

plt.title('KNN Confusion Matrix\n')
plt.xlabel('Predicted\n')
plt.ylabel('Actual\n')

# Matrice di confusione -- Logistic Regression
plt.subplot(1, 2, 2)
lr_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])

plt.title('Logistic Regression Confusion Matrix\n')
plt.xlabel('Predicted\n')
plt.ylabel('Actual\n')

# SHOW
plt.tight_layout()        ## evita che gli oggetti nei grafici si sovrappongano
plt.show()


# Esportazione Modelli
print("\nEsportazione modelli")
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
"""