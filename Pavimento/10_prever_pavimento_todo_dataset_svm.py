import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import DATAPATH, PROJECT_ROOT

print("DATAPATH:", DATAPATH)
print("PROJECT_ROOT:", PROJECT_ROOT)

#Carrega treino de pavimento
X_train = pd.read_csv(os.path.join(DATAPATH, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATAPATH, "y_train.csv")).squeeze()

print("X_train:", X_train.shape, "y_train:", y_train.shape)

# Normaliza e treina SVM (mesma config do 4_svm_pavimento.py)
scaler = StandardScaler() #Normaliza treino
X_train_s = scaler.fit_transform(X_train) # Fit e transforma treino

svm = SVC(kernel="rbf", random_state=42) # Usa kernel RBF (padrão) e random_state=42
svm.fit(X_train_s, y_train) # Treina SVM com dados normalizados

#Carrega dataset completo de pavimento
df_pav = pd.read_csv(os.path.join(DATAPATH, "pavimento_pronto.csv"))
print("pavimento_pronto:", df_pav.shape)

sensor_cols = X_train.columns.tolist() # Lista das colunas de sensores
X_all = df_pav[sensor_cols].copy() # Seleciona só as colunas de sensores
X_all_s = scaler.transform(X_all) # Normaliza todas as linhas

print("X_all_s:", X_all_s.shape)

# Preve TODAS as linhas
y_pred_all = svm.predict(X_all_s)

# Salva
df_pred_all = pd.DataFrame({"pavimento_previsto_SVM": y_pred_all})
OUTFILE = os.path.join(DATAPATH, "pavimento_previsto_SVM_todo_dataset.csv")
df_pred_all.to_csv(OUTFILE, index=False)

print("Salvo:", OUTFILE, "shape:", df_pred_all.shape)
