import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ajuste de paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from _1_config import DATAPATH, PROJECT_ROOT

print("DATAPATH:", DATAPATH)
print("PROJECT_ROOT:", PROJECT_ROOT)

data_dir = os.path.join(PROJECT_ROOT, "Qualidade", "data")
print("data_dir:", data_dir)

os.makedirs(data_dir, exist_ok=True)
print(data_dir, "criado.");

X_train = pd.read_csv(os.path.join(data_dir, "X_train_qual_pavprev_LSTM.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train_qual_pavprev_LSTM.csv")).squeeze()

X_test  = pd.read_csv(os.path.join(data_dir, "X_test_qual_pavprev_LSTM.csv"))
y_test  = pd.read_csv(os.path.join(data_dir, "y_test_qual_pavprev_LSTM.csv")).squeeze()

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)

sensor_cols = X_train.columns[:32].tolist()   # primeiras 32 = sensores
pav_col     = X_train.columns[32]             # última = pavimento previsto

X_train_feats = X_train[sensor_cols].copy()
X_test_feats  = X_test[sensor_cols].copy()


scaler = StandardScaler()
X_train_feats_s = scaler.fit_transform(X_train_feats)
X_test_feats_s  = scaler.transform(X_test_feats)

X_train_norm = np.concatenate([X_train_feats_s,
                               X_train[pav_col].values.reshape(-1, 1)], axis=1)
X_test_norm  = np.concatenate([X_test_feats_s,
                               X_test[pav_col].values.reshape(-1, 1)], axis=1)

print("X_train_norm:", X_train_norm.shape, "X_test_norm:", X_test_norm.shape)

WINDOW = 20

def make_seq(X, y, window):
    seq_X, seq_y = [], []
    for i in range(len(X) - window + 1):
        seq_X.append(X[i:i+window])
        seq_y.append(y.iloc[i+window-1])
    return np.array(seq_X), np.array(seq_y)

X_train_seq, y_train_seq = make_seq(X_train_norm, y_train, WINDOW)
X_test_seq,  y_test_seq  = make_seq(X_test_norm,  y_test,  WINDOW)

print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
print("X_test_seq :", X_test_seq.shape,  "y_test_seq :", y_test_seq.shape)

np.save(os.path.join(data_dir, "X_train_qual_seq_pavprevLSTM.npy"), X_train_seq)
np.save(os.path.join(data_dir, "y_train_qual_seq_pavprevLSTM.npy"), y_train_seq)
np.save(os.path.join(data_dir, "X_test_qual_seq_pavprevLSTM.npy"),  X_test_seq)
np.save(os.path.join(data_dir, "y_test_qual_seq_pavprevLSTM.npy"),  y_test_seq)

print("Arquivos de sequência salvos em:", data_dir)
