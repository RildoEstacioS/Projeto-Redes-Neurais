import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import DATAPATH, PROJECT_ROOT

print("DATAPATH:", DATAPATH)
print("PROJECT_ROOT:", PROJECT_ROOT)

data_dir = os.path.join(PROJECT_ROOT, "Qualidade", "data")
print("data_dir:", data_dir)

# Carrega base QUALIDADE com pavimento_previsto_RF (não encontrei um modo de fazer a previsão com LSTM)
X_train = pd.read_csv(os.path.join(data_dir, "X_train_qual_pavprev_RF.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train_qual_pavprev_RF.csv")).squeeze()

X_test = pd.read_csv(os.path.join(data_dir, "X_test_qual_pavprev_RF.csv"))
y_test = pd.read_csv(os.path.join(data_dir, "y_test_qual_pavprev_RF.csv")).squeeze()

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test: ", X_test.shape,  "y_test: ",  y_test.shape)

# Separa 32 features de sensor + 1 coluna pavimento_previsto
sensor_cols = X_train.columns[:32].tolist() # primeiras 32 colunas são sensores
pav_col = X_train.columns[32] # última coluna é pavimento_previsto

X_train_feats = X_train[sensor_cols].copy() # Sensores para treino
X_test_feats  = X_test[sensor_cols].copy() # Sensores para teste 

#Normaliza apenas as 32 features
scaler = StandardScaler() # Normaliza dados
X_train_feats_s = scaler.fit_transform(X_train_feats) # Fit e transforma treino
X_test_feats_s  = scaler.transform(X_test_feats) # Transforma teste

#Reconstrói matrizes com 32 features normalizadas + pavimento_previsto bruto
X_train_norm = np.concatenate([X_train_feats_s, X_train[[pav_col]].values], axis=1)
X_test_norm  = np.concatenate([X_test_feats_s,  X_test[[pav_col]].values],  axis=1)

# Cria sequências deslizantes
WINDOW = 20

def make_seq(X, y, window): 
    seq_X, seq_y = [], [] 
    for i in range(len(X) - window + 1): 
        seq_X.append(X[i:i+window])
        seq_y.append(y.iloc[i+window-1])  
    return np.array(seq_X), np.array(seq_y)

X_train_seq, y_train_seq = make_seq(X_train_norm, y_train, WINDOW) # Cria sequências de treinamento
X_test_seq,  y_test_seq  = make_seq(X_test_norm,  y_test,  WINDOW) # Cria sequências de teste

print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
print("X_test_seq: ", X_test_seq.shape,  "y_test_seq: ",  y_test_seq.shape)

# Salva .npy
np.save(os.path.join(data_dir, "X_train_qual_seq_pavprev.npy"), X_train_seq)
np.save(os.path.join(data_dir, "y_train_qual_seq_pavprev.npy"), y_train_seq)
np.save(os.path.join(data_dir, "X_test_qual_seq_pavprev.npy"),  X_test_seq)
np.save(os.path.join(data_dir, "y_test_qual_seq_pavprev.npy"),  y_test_seq)

print("\nArquivos de sequência salvos em:", data_dir)
