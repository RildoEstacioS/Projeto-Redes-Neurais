import sys
import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
from _1_config import DATAPATH as DATA_PATH

SENSOR_FILE = os.path.join(DATA_PATH, "dataset_gps_mpu_left.csv")
LABEL_FILE = os.path.join(DATA_PATH, "dataset_labels.csv")

df_sensores = pd.read_csv(SENSOR_FILE)

pavimento_cols = ['asphalt_road', 'cobblestone_road', 'dirt_road']
df_labels = pd.read_csv(LABEL_FILE, usecols=pavimento_cols)

def get_pavimento(row):
    if row['asphalt_road'] == 1:
        return 0  # asfalto
    elif row['cobblestone_road'] == 1:
        return 1  # paralelepípedo
    elif row['dirt_road'] == 1:
        return 2  # terra
    else:
        return -1 # indefinido

# Aplica `get_pavimento` em cada linha e cria a coluna `target_pavimento`
df_labels['target_pavimento'] = df_labels.apply(get_pavimento, axis=1)

df_final = df_sensores.copy()
df_final['target_pavimento'] = df_labels['target_pavimento']

df_final.to_csv(os.path.join(DATA_PATH, "pavimento_pronto.csv"), index=False)


df_final = df_final[df_final['target_pavimento'] != -1]

print(df_final[['target_pavimento'] + df_final.columns.tolist()[:5]])  # Mostra target e 5 primeiras colunas

print(f"\nTotal de amostras prontas para classificação: {len(df_final)}")
print("Valores únicos em target_pavimento (0=asfalto, 1=paralelepípedo, 2=terra):")
print(df_final['target_pavimento'].value_counts())
