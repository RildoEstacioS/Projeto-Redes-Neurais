import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# sobe para a raiz do projeto e importa config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import DATAPATH, PROJECT_ROOT

# Carrega o modelo
models_dir = os.path.join(PROJECT_ROOT, "Pavimento", "models")
MODEL_PATH = os.path.join(models_dir, "lstm_pavimento.h5")
model = load_model(MODEL_PATH)
print("Modelo carregado de:", MODEL_PATH)

# Carrega X_train para ajustar scaler igual ao script de sequência
X_train = pd.read_csv(os.path.join(DATAPATH, "X_train.csv")) # 1) Carrega treino de pavimento
sensor_cols = X_train.columns.tolist() # Lista das colunas de sensores

scaler = StandardScaler() # 2) Normaliza treino
X_train_s = scaler.fit_transform(X_train) # Fit e transforma treino

# Carrega dataset completo e normaliza
df_pav = pd.read_csv(os.path.join(DATAPATH, "pavimento_pronto.csv"))
print("pavimento_pronto:", df_pav.shape)

X_all = df_pav[sensor_cols].copy()
X_all_s = scaler.transform(X_all)

# Constrói sequências deslizantes (janela=20)
WINDOW = 20 
seq_X = []
for i in range(len(X_all_s) - WINDOW + 1):
    seq_X.append(X_all_s[i:i+WINDOW])
seq_X = np.array(seq_X)
print("seq_X:", seq_X.shape)

# Prediz classes para cada sequência
y_pred_seq = model.predict(seq_X, verbose=0).argmax(axis=1) 

#Alinha uma predição por linha (label da última amostra da janela)
y_full = np.full(len(X_all_s), np.nan)
y_full[WINDOW-1:] = y_pred_seq

df_pred_all = pd.DataFrame({"pavimento_previsto_LSTM": y_full})
OUTFILE = os.path.join(DATAPATH, "pavimento_previsto_LSTM_todo_dataset.csv")
df_pred_all.to_csv(OUTFILE, index=False)
#Salva
print("Salvo:", OUTFILE, "shape:", df_pred_all.shape)
print("Primeiras linhas:")
print(df_pred_all.head(25))
