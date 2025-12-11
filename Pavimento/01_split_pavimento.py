import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from _1_config import DATAPATH as DATA_PATH, PROJECT_ROOT
FINAL_DATASET = os.path.join(DATA_PATH, "pavimento_pronto.csv")
df_final = pd.read_csv(FINAL_DATASET)

# Features e alvo
X = df_final.drop(columns=['target_pavimento'])
y = df_final['target_pavimento']

# Split 80/20 estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Tamanhos dos conjuntos:")
print(f"Treino: {X_train.shape[0]}")
print(f"Teste:  {X_test.shape[0]}")

# Salva para reutilizar
X_train.to_csv(os.path.join(DATA_PATH, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(DATA_PATH, "y_train.csv"), index=False)
X_test.to_csv(os.path.join(DATA_PATH, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(DATA_PATH, "y_test.csv"), index=False)
