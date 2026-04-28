import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['target'] = iris.target

# Selezione Features e Target
X = df.drop(columns=['target'], axis=1)
Y = df['target']

# Splitting dei dati
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- LOGISTIC REGRESSION ---
model_lr = LogisticRegression(solver="lbfgs", max_iter=1000)
model_lr.fit(X_train, Y_train)

# Predizione su TEST set
lr_preds = model_lr.predict(X_test)

print("=== LOGISTIC REGRESSION ===")
print(f"Accuracy: {accuracy_score(Y_test, lr_preds)}")
print(classification_report(Y_test, lr_preds))

# --- K-NEAREST NEIGHBOURS (KNN) ---
model_knn = KNeighborsClassifier(n_neighbors=11)
model_knn.fit(X_train, Y_train)

# Predizione su TEST set
knn_preds = model_knn.predict(X_test)

print("\n=== K-NEAREST NEIGHBOURS ===")
print(f"Accuracy: {accuracy_score(Y_test, knn_preds)}")
print(classification_report(Y_test, knn_preds))

# --- CONFUSION MATRIX ---
plt.figure(figsize=(8,4))
sns.heatmap(confusion_matrix(Y_test, knn_preds), annot=True, cmap='Blues')
plt.title('Confusion Matrix - KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --- ESPORTAZIONE MODELLI ---
print("\nEsportazione modelli")
joblib.dump(knn_model, 'iris_model.joblib')
joblib.dump(model, 'iris_model_knn.joblib')
