import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import DATAPATH

# Carrega treino de pavimento
X_train = pd.read_csv(os.path.join(DATAPATH, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATAPATH, "y_train.csv")).squeeze()

print("X_train:", X_train.shape, "y_train:", y_train.shape)

# Normaliza e treina MLP (mesma config do 3_mlp_pavimento.py)
scaler = StandardScaler() # Fit e transforma treino
X_train_s = scaler.fit_transform(X_train) # Transforma treino

mlp = MLPClassifier( # Mesma config do 3_mlp_pavimento.py
    hidden_layer_sizes=(50, 25), # Duas camadas ocultas
    activation="relu", # Função de ativação ReLU
    solver="adam", # Otimizador Adam
    max_iter=300, # Máximo de 300 iterações
    random_state=42 # Para reprodutibilidade 
)
mlp.fit(X_train_s, y_train) # Treina MLP

# Carrega dataset completo de pavimento
df_pav = pd.read_csv(os.path.join(DATAPATH, "pavimento_pronto.csv"))
print("pavimento_pronto:", df_pav.shape)

sensor_cols = X_train.columns.tolist() # Lista das colunas de sensores
X_all = df_pav[sensor_cols].copy() # Seleciona só as colunas de sensores
X_all_s = scaler.transform(X_all) # Normaliza todas as linhas

print("X_all_s:", X_all_s.shape)

# Prever TODAS as linhas
y_pred_all = mlp.predict(X_all_s)

# Salva
df_pred_all = pd.DataFrame({"pavimento_previsto_MLP": y_pred_all})
OUTFILE = os.path.join(DATAPATH, "pavimento_previsto_MLP_todo_dataset.csv")
df_pred_all.to_csv(OUTFILE, index=False)

print("Salvo:", OUTFILE, "shape:", df_pred_all.shape)
