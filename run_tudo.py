# run_tudo.py
# Roda todos os scripts na ordem correta e salva métricas consolidadas.

import subprocess
import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(__file__)


def run(relpath):
    path = os.path.join(ROOT, relpath)
    print(f"\n=== Rodando: {relpath} ===")
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"ERRO ao rodar {relpath}, parando.")
        sys.exit(1)


# ===================== PAVIMENTO =====================

# Modelos tabulares + LSTM
run(r"Pavimento\00_pipeline.py")
run(r"Pavimento\01_split_pavimento.py")
run(r"Pavimento\02_rf_pavimento.py")
run(r"Pavimento\03_mlp_pavimento.py")
run(r"Pavimento\04_svm_pavimento.py")
run(r"Pavimento\05_lstm_pavimento_sequencias.py")
run(r"Pavimento\06_lstm_pavimento_modelo.py")

# Previsto para TODO o dataset (RF, MLP, LSTM, SVM)
run(r"Pavimento\07_prever_pavimento_todo_dataset.py")          # RF
run(r"Pavimento\08_prever_pavimento_todo_dataset_mlp.py")      # MLP
run(r"Pavimento\09_prever_pavimento_todo_dataset_lstm.py")     # LSTM
run(r"Pavimento\10_prever_pavimento_todo_dataset_svm.py")      # SVM


# ===================== QUALIDADE =====================

# Monta bases de qualidade para cada pavimento previsto
run(r"Qualidade\1_pipelines_qualidade_multiplos_pavprev.py")

# Modelos de qualidade (cada um com sua base)
run(r"Qualidade\2_rf_qualidade_pavprev.py")
run(r"Qualidade\3_mlp_qualidade_pavprev.py")
run(r"Qualidade\4_svm_qualidade_pavprev.py")
run(r"Qualidade\5_lstm_qualidade_sequencias_pavprev.py")
run(r"Qualidade\6_lstm_qualidade_modelo_pavprev.py")


# ===================== MÉTRICAS CONSOLIDADAS =====================

print("\n=== Calculando métricas consolidadas ===")

from metrics_utils import (
    avaliar_rf_pavimento,
    avaliar_mlp_pavimento,
    avaliar_svm_pavimento,
    avaliar_lstm_pavimento,
    avaliar_rf_qualidade_pavprev_RF,
    avaliar_mlp_qualidade_pavprev_RF,
    avaliar_svm_qualidade_pavprev_RF,
    avaliar_lstm_qualidade_pavprev_RF,
)

metricas = []
matrizes = {}

# Pavimento: RF, MLP, SVM, LSTM
for func in [
    avaliar_rf_pavimento,
    avaliar_mlp_pavimento,
    avaliar_svm_pavimento,
    avaliar_lstm_pavimento,
]:
    m = func()
    metricas.append({
        "tarefa": m["tarefa"],
        "modelo": m["modelo"],
        "pav_prev_origem": m["pav_prev_origem"],
        "acc": m["acc"],
    })
    key = f"cm_{m['tarefa']}_{m['modelo']}"
    matrizes[key] = m["cm"]

# Qualidade com pavimento_previsto_RF: RF, MLP, SVM, LSTM
for func in [
    avaliar_rf_qualidade_pavprev_RF,
    avaliar_mlp_qualidade_pavprev_RF,
    avaliar_svm_qualidade_pavprev_RF,
    avaliar_lstm_qualidade_pavprev_RF,
]:
    m = func()
    metricas.append({
        "tarefa": m["tarefa"],
        "modelo": m["modelo"],
        "pav_prev_origem": m["pav_prev_origem"],
        "acc": m["acc"],
    })
    key = f"cm_{m['tarefa']}_{m['modelo']}_pavRF"
    matrizes[key] = m["cm"]

# Salva CSV de métricas
df_metrics = pd.DataFrame(metricas)
out_csv = os.path.join(ROOT, "resultados_metricas_basico.csv")
df_metrics.to_csv(out_csv, index=False)
print("Métricas salvas em:", out_csv)

# Salva matrizes de confusão em .npz
out_npz = os.path.join(ROOT, "resultados_matrizes_basico.npz")
np.savez(out_npz, **matrizes)
print("Matrizes de confusão salvas em:", out_npz)

# ===================== GRÁFICOS =====================

# Gera gráficos de barras de acurácia e matrizes de confusão
run(r"plots_modelos.py")
run(r"plots_matriz.py")

print("\n=== Pipeline completo finalizado com métricas salvas. ===")
