import sys
import pandas as pd
import os

# Descobre a raiz do projeto e adiciona ao path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from _1_config import DATAPATH as DATA_PATH

# Mostra caminhos para conferência
print("DATA_PATH  =", DATA_PATH)

SENSOR_FILE = os.path.join(DATA_PATH, "dataset_gps_mpu_left.csv")
LABEL_FILE  = os.path.join(DATA_PATH, "dataset_labels.csv")

print("SENSOR_FILE =", SENSOR_FILE)
print("LABEL_FILE  =", LABEL_FILE)

# Carrega sensores
df_sensores = pd.read_csv(SENSOR_FILE)

# Carrega apenas as colunas de pavimento
pavimento_cols = ["asphalt_road", "cobblestone_road", "dirt_road"]
df_labels = pd.read_csv(LABEL_FILE, usecols=pavimento_cols)

# Função para mapear one-hot -> rótulo inteiro
def get_pavimento(row):
    if row["asphalt_road"] == 1:
        return 0  # asfalto
    elif row["cobblestone_road"] == 1:
        return 1  # paralelepípedo
    elif row["dirt_road"] == 1:
        return 2  # terra
    else:
        return -1  # indefinido / inconsistência

# Cria coluna target_pavimento
df_labels["target_pavimento"] = df_labels.apply(get_pavimento, axis=1)

# Junta sensores + rótulo
df_final = df_sensores.copy()
df_final["target_pavimento"] = df_labels["target_pavimento"]

# Remove linhas com rótulo indefinido ANTES de salvar
df_final = df_final[df_final["target_pavimento"] != -1]

# Salva CSV final
output_path = os.path.join(DATA_PATH, "pavimento_pronto.csv")
df_final.to_csv(output_path, index=False)
print("\nCSV salvo em:", output_path)

# Prints de conferência
cols_preview = ["target_pavimento"] + df_final.columns.tolist()[:5]
print("\nPrévia dos dados:")
print(df_final[cols_preview].head())

print(f"\nTotal de amostras prontas para classificação: {len(df_final)}")
print("Valores únicos em target_pavimento (0=asfalto, 1=paralelepípedo, 2=terra):")
print(df_final["target_pavimento"].value_counts())
