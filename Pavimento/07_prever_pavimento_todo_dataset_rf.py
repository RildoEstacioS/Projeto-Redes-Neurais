import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import DATAPATH

# Carrega treino de PAVIMENTO
X_train = pd.read_csv(os.path.join(DATAPATH, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATAPATH, "y_train.csv")).squeeze()

print("X_train:", X_train.shape, "y_train:", y_train.shape)

# Treina RF igual ao 2_rf_pavimento.py
rf = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1) # Mesma config do 2_rf_pavimento.py
rf.fit(X_train, y_train) # Treina RF

# Carrega dataset completo de pavimento
df_pav = pd.read_csv(os.path.join(DATAPATH, "pavimento_pronto.csv")) # Dataset com 91.555 linhas
print("pavimento_pronto:", df_pav.shape) 

# Usa as mesmas 32 colunas de sensores de X_train
sensor_cols = X_train.columns.tolist() # Lista das colunas de sensores
X_all = df_pav[sensor_cols].copy() # Seleciona só as colunas de sensores
print("X_all:", X_all.shape) # (91555, 32 colunas)

# Preve pavimento para TODAS as linhas
y_pred_all = rf.predict(X_all) 

# Salva
df_pred_all = pd.DataFrame({"pavimento_previsto_RF": y_pred_all}) # DataFrame
OUTFILE = os.path.join(DATAPATH, "pavimento_previsto_RF_todo_dataset.csv") # Caminho do arquivo de saída
df_pred_all.to_csv(OUTFILE, index=False) # Salva sem índice

print("Salvo:", OUTFILE, "shape:", df_pred_all.shape) #Salvo
    