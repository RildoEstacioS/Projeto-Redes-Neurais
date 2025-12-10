import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import DATAPATH
# Define o tamanho da janela para as sequências LSTM
WINDOW = 20

# Carrega dados tabulares
X_train = pd.read_csv(os.path.join(DATAPATH, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATAPATH, "y_train.csv")).squeeze()

X_test = pd.read_csv(os.path.join(DATAPATH, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATAPATH, "y_test.csv")).squeeze()

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Função para criar sequências
def create_sequences(X, y, window_size):
    seq_X, seq_y = [], []
    for i in range(len(X) - window_size + 1):
        seq = X[i:i+window_size]
        label = y.iloc[i+window_size-1]
        seq_X.append(seq)
        seq_y.append(label)
    return np.array(seq_X), np.array(seq_y)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, WINDOW)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test,  WINDOW)

print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
print("X_test_seq: ", X_test_seq.shape,  "y_test_seq: ", y_test_seq.shape)

# Salva em .npy na pasta de dados original
np.save(os.path.join(DATAPATH, "X_train_seq.npy"), X_train_seq)
np.save(os.path.join(DATAPATH, "y_train_seq.npy"), y_train_seq)
np.save(os.path.join(DATAPATH, "X_test_seq.npy"),  X_test_seq)
np.save(os.path.join(DATAPATH, "y_test_seq.npy"),  y_test_seq)

print("\nArquivos salvos em:", DATAPATH)
