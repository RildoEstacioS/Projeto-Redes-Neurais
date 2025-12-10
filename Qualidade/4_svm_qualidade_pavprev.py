import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import PROJECT_ROOT, DATAPATH  # DATAPATH só para log

data_dir = os.path.join(PROJECT_ROOT, "Qualidade", "data")

print("DATAPATH:", DATAPATH)
print("PROJECT_ROOT:", PROJECT_ROOT)

# Carrega os dados de qualidade com pavimento previsto SVM
X_train = pd.read_csv(os.path.join(data_dir, "X_train_qual_pavprev_SVM.csv")) 
y_train = pd.read_csv(os.path.join(data_dir, "y_train_qual_pavprev_SVM.csv")).squeeze()

X_test = pd.read_csv(os.path.join(data_dir, "X_test_qual_pavprev_SVM.csv"))
y_test = pd.read_csv(os.path.join(data_dir, "y_test_qual_pavprev_SVM.csv")).squeeze()


print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)

# Normalização
scaler = StandardScaler() # Normaliza dados
X_train_s = scaler.fit_transform(X_train) # Fit e transforma treino
X_test_s  = scaler.transform(X_test) # Transforma teste

# Define e treina SVM
svm = SVC(kernel="rbf", random_state=42) # Usa kernel RBF (padrão) e random_state=42
svm.fit(X_train_s, y_train) # Treina SVM

# Avalia no TESTE
print("\nTESTE: ")
y_pred = svm.predict(X_test_s)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de classificação:")
print(classification_report(y_test, y_pred))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))
