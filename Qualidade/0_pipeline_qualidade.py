"""
Script 2 (qualidade): Pipeline de pré-processamento para QUALIDADE DA VIA

Objetivo:
- Ler dataset_gps_mpu_left.csv (sensores + GPS)
- Ler dataset_labels.csv
- Criar a coluna target_qualidade com 3 classes (lado esquerdo):
    0 = good_road_left
    1 = regular_road_left
    2 = bad_road_left
- Juntar tudo em um único DataFrame
- Salvar em qualidade_pronto.csv
"""

import pandas as pd
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
from _1_config import DATAPATH as DATA_PATH

SENSOR_FILE = os.path.join(DATA_PATH, "dataset_gps_mpu_left.csv")
LABEL_FILE  = os.path.join(DATA_PATH, "dataset_labels.csv")
OUTPUT_FILE = os.path.join(DATA_PATH, "qualidade_pronto.csv")

print("Carregando dados de sensores...")
df_sensores = pd.read_csv(SENSOR_FILE)
print(f"df_sensores shape: {df_sensores.shape}")

qualidade_cols = ['good_road_left', 'regular_road_left', 'bad_road_left']
print("Carregando labels de qualidade...")
df_labels = pd.read_csv(LABEL_FILE, usecols=qualidade_cols)
print(f"df_labels shape: {df_labels.shape}")

def map_qualidade(row):
    if row['good_road_left'] == 1:
        return 0  # boa
    elif row['regular_road_left'] == 1:
        return 1  # regular
    elif row['bad_road_left'] == 1:
        return 2  # ruim
    else:
        return -1  # indefinido (caso apareça)

df_labels['target_qualidade'] = df_labels.apply(map_qualidade, axis=1)

if len(df_sensores) != len(df_labels):
    print("número de linhas diferente entre sensores e labels!")
    print(f"sensores: {len(df_sensores)}, labels: {len(df_labels)}")

df_final_qualidade = df_sensores.copy()
df_final_qualidade['target_qualidade'] = df_labels['target_qualidade']

n_before = len(df_final_qualidade)
df_final_qualidade = df_final_qualidade[df_final_qualidade['target_qualidade'] != -1]
n_after = len(df_final_qualidade)

print(f"\nRegistros antes de remover indefinidos: {n_before}")
print(f"Registros após remover indefinidos:   {n_after}")

df_final_qualidade.to_csv(OUTPUT_FILE, index=False)
print(f"\nArquivo salvo em: {OUTPUT_FILE}")
print("Columns:", df_final_qualidade.columns[:10], "...")
