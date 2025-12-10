import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pavimento – ajuste valores
acc_pav = {
    "RF": 0.9997,
    "MLP": 0.9982,
    "SVM": 0.9980,
    "LSTM": 0.9940
}

# Qualidade – ajuste se precisar
acc_qual = {
    "RF": 0.9940,
    "MLP": 0.9611,
    "SVM": 0.9433,
    "LSTM": 0.9378
}

def plot_bar(acc_dict, title, filename, ylim=(90, 100)):
    modelos = list(acc_dict.keys())
    valores = [v * 100 for v in acc_dict.values()]

    plt.figure(figsize=(6, 4))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
    bars = plt.bar(modelos, valores, color=colors[:len(modelos)])
    plt.ylim(*ylim)
    plt.ylabel("Acurácia (%)")
    plt.title(title)

    for b, v in zip(bars, valores):
        plt.text(b.get_x() + b.get_width()/2, v + 0.1,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfico salvo em: {out_path}")

if __name__ == "__main__":
    plot_bar(acc_pav,
             "Acurácia por modelo – Tipo de pavimento",
             "acc_pavimento_modelos.png")
    plot_bar(acc_qual,
             "Acurácia por modelo – Qualidade da via",
             "acc_qualidade_modelos.png",
             ylim=(90, 100))
