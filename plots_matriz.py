import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

classes_pav = ["asphalt", "cobblestone", "dirt"]
classes_qual = ["good", "regular", "bad"]

def plot_cm(cm, classes, title, filename):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classes)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Matriz salva em:", out_path)


if __name__ == "__main__":
    npz_path = os.path.join(BASE_DIR, "resultados_matrizes_basico.npz")
    data = np.load(npz_path)

    # Pavimento
    plot_cm(data["cm_pavimento_RF"],   classes_pav, "RF (Pavimento)",   "cm_pavimento_rf.png")
    plot_cm(data["cm_pavimento_MLP"],  classes_pav, "MLP (Pavimento)",  "cm_pavimento_mlp.png")
    plot_cm(data["cm_pavimento_SVM"],  classes_pav, "SVM (Pavimento)",  "cm_pavimento_svm.png")
    plot_cm(data["cm_pavimento_LSTM"], classes_pav, "LSTM (Pavimento)", "cm_pavimento_lstm.png")

    # Qualidade com pavimento_previsto_RF
    plot_cm(data["cm_qualidade_RF_pavRF"],   classes_qual, "RF (Qualidade)",   "cm_qualidade_rf.png")
    plot_cm(data["cm_qualidade_MLP_pavRF"],  classes_qual, "MLP (Qualidade)",  "cm_qualidade_mlp.png")
    plot_cm(data["cm_qualidade_SVM_pavRF"],  classes_qual, "SVM (Qualidade)",  "cm_qualidade_svm.png")
    plot_cm(data["cm_qualidade_LSTM_pavRF"], classes_qual, "LSTM (Qualidade)", "cm_qualidade_lstm.png")
