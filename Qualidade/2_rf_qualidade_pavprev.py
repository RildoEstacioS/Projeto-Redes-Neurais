import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import PROJECT_ROOT

data_dir = os.path.join(PROJECT_ROOT, "Qualidade", "data")

# Carrega dados
X_train = pd.read_csv(os.path.join(data_dir, "X_train_qual_pavprev_RF.csv")) # Carrega treino com pavimento previsto RF
y_train = pd.read_csv(os.path.join(data_dir, "y_train_qual_pavprev_RF.csv")).squeeze() # Carrega target de qualidade do treino

X_test = pd.read_csv(os.path.join(data_dir, "X_test_qual_pavprev_RF.csv")) # Carrega teste com pavimento previsto RF 
y_test = pd.read_csv(os.path.join(data_dir, "y_test_qual_pavprev_RF.csv")).squeeze() # Carrega target de qualidade do teste


print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)

# Treina RF
rf_qual = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1) # Mesma config do 2_rf_qualidade.py
rf_qual.fit(X_train, y_train) # Treina RF

# Avaliaç]ao
y_pred = rf_qual.predict(X_test) # Previsões
print("Acurácia:", accuracy_score(y_test, y_pred)) 
print("Relatório de classificação:")
print(classification_report(y_test, y_pred))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))
