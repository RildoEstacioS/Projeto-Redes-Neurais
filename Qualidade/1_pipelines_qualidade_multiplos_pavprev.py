import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import DATAPATH, PROJECT_ROOT

data_dir = os.path.join(PROJECT_ROOT, "Qualidade", "data")
os.makedirs(data_dir, exist_ok=True)

print("DATAPATH:", DATAPATH)
print("data_dir:", data_dir)

# Base de qualidade original (sensores + target_qualidade)
df_qual = pd.read_csv(os.path.join(DATAPATH, "qualidade_pronto.csv")) # Carrega base de qualidade pronta
sensor_cols = df_qual.columns[:32].tolist() # Colunas de sensores

print("df_qual:", df_qual.shape)

pav_files = {
    "RF":   "pavimento_previsto_RF_todo_dataset.csv",
    "MLP":  "pavimento_previsto_MLP_todo_dataset.csv",
    "LSTM": "pavimento_previsto_LSTM_todo_dataset.csv",
    "SVM":  "pavimento_previsto_SVM_todo_dataset.csv",
} # Arquivos gerados previamente com previsões de pavimento


# Loop para cada método de pavimento previsto
for key, fname in pav_files.items():
    print(f"\n=== Montando base de QUALIDADE com pavimento_previsto_{key} ===")
    df_pav_prev = pd.read_csv(os.path.join(DATAPATH, fname)) # Carrega pavimento previsto
    print("df_pav_prev:", df_pav_prev.shape)

    # Garante alinhamento
    assert len(df_qual) == len(df_pav_prev)

    col_name = f"pavimento_previsto_{key}" # Nome da coluna de pavimento previsto

    # Se no arquivo a coluna tiver outro nome, renomeia
    if df_pav_prev.columns[0] != col_name:
        df_pav_prev = df_pav_prev.rename(columns={df_pav_prev.columns[0]: col_name})

    # Monta dataframe final
    df_final = df_qual[sensor_cols].copy()
    df_final[col_name] = df_pav_prev[col_name]
    df_final["target_qualidade"] = df_qual["target_qualidade"]

    # Split
    X = df_final.drop(columns=["target_qualidade"])
    y = df_final["target_qualidade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    prefix = f"qual_pavprev_{key}"

    X_train.to_csv(os.path.join(data_dir, f"X_train_{prefix}.csv"), index=False)
    X_test.to_csv(os.path.join(data_dir, f"X_test_{prefix}.csv"), index=False)
    y_train.to_csv(os.path.join(data_dir, f"y_train_{prefix}.csv"), index=False)
    y_test.to_csv(os.path.join(data_dir, f"y_test_{prefix}.csv"), index=False)

    print(f"{prefix}: X_train {X_train.shape}, X_test {X_test.shape}")

