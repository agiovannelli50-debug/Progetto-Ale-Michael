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

#caricamento dati
wine = load_wine()
#print(wine)

#separazione target dalle altre feature
X = wine.data
y = wine.target
#print(X)

#conversione dataset in dataframe
df = pd.DataFrame(X, columns=wine.feature_names)
'''
print(df)
print("\n")
print(df.columns)
'''



#data cleaning - dataset già pulito, questo è un controllo per verificare
print(df.isnull().sum(),"\n")

#EDA
print("Colonne del dataset:\n",df.columns)
print("Prime righe del dataset:",df.head(),"\n")
print("Shape del dataset:",df.shape,"\n")
print("Dataset info:",df.info,"\n")
print("Info statistiche del dataset:",df.describe(),"\n")

#matrice di correlazione + heatmap
correlation = df.corr()
print("\nCorrelation matrix:\n", correlation)

plt.figure(figsize=(10,10))
sns.heatmap(correlation, annot=True, cmap="vlag",cbar=True, square=True , annot_kws={"size": 8},fmt=".1f")
plt.show()


X = df
y = y
#print(y)

#splitting di train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
print("\nShowing how X has been split into:")
print(X.shape,X_train.shape,X_test.shape)

#feature scaling - importante per logistic regression
#standardizza le feature sottraendo la media e dividendo
#per la deviazione standard -> z = (x - mean) / std.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # calcola la media e dev standard
X_test_scaled = scaler.transform(X_test) #applica la trasformazione


#modello KNN - k=13 calcolo-> sqrt di num di righe del dataset, per ottenere k adeguato
knn_model = KNeighborsClassifier(n_neighbors=13)
knn_model.fit(X_train_scaled, y_train)

#predictions
knn_y_pred = knn_model.predict(X_test_scaled)

#evaluation
print("\nKNN Accuracy Score:", accuracy_score(y_test, knn_y_pred))
print("\nKNN Classification Report:\n", classification_report(y_test, knn_y_pred))

#modello logistic regression
model = LogisticRegression(solver='lbfgs', random_state=2, max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("\nLogistic Regression Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred))

#visualizzazioni - Confusion Matrix per entrambi i modelli
#mostra actual vs predicted di ogni classe
plt.figure(figsize=(12, 5))

#KNN matrice di confusione
plt.subplot(1, 2, 1)
knn_cm = confusion_matrix(y_test, knn_y_pred)
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

#matrce di confusione logistic regression
plt.subplot(1, 2, 2)
lr_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()


# Esportazione dei modelli
print("\nEsportazione dei modelli")
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(model, 'logistic_regression_model.pkl')
print("Modelli esportati con successo")


