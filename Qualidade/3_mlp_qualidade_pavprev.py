import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import PROJECT_ROOT

data_dir = os.path.join(PROJECT_ROOT, "Qualidade", "data")

# Carrega dados 
X_train = pd.read_csv(os.path.join(data_dir, "X_train_qual_pavprev_MLP.csv")) # Carrega treino com pavimento previsto MLP
y_train = pd.read_csv(os.path.join(data_dir, "y_train_qual_pavprev_MLP.csv")).squeeze() # Carrega target de qualidade do treino

X_test = pd.read_csv(os.path.join(data_dir, "X_test_qual_pavprev_MLP.csv")) # Carrega teste com pavimento previsto MLP
y_test = pd.read_csv(os.path.join(data_dir, "y_test_qual_pavprev_MLP.csv")).squeeze() # Carrega target de qualidade do teste

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)

# Normalização
scaler = StandardScaler() # Normaliza dados
X_train_s = scaler.fit_transform(X_train) # Fit e transforma treino
X_test_s  = scaler.transform(X_test) # Transforma teste

# Define e treina MLP
mlp = MLPClassifier( 
    hidden_layer_sizes=(64, 64),# Duas camadas ocultas com 64 neurônios cada
    activation="relu", # Função de ativação ReLU
    solver="adam", # Otimizador Adam
    max_iter=100, # Máximo de 100 iterações
    random_state=42 
)

mlp.fit(X_train_s, y_train) # Treina MLP

# Avalia no TESTE
print("\n TESTE QUALIDADE")
y_pred = mlp.predict(X_test_s)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de classificação:")
print(classification_report(y_test, y_pred))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))
