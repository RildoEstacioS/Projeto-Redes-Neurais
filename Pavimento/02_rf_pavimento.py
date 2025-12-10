import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importa caminhos
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import DATAPATH, PROJECT_ROOT

# Carrega conjuntos de PAVIMENTO
X_train = pd.read_csv(os.path.join(DATAPATH, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATAPATH, "y_train.csv")).squeeze()

X_test = pd.read_csv(os.path.join(DATAPATH, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATAPATH, "y_test.csv")).squeeze()

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)

# Treina RF
rf = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

# Avaliação no teste
y_pred_test = rf.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))

# Salva predições para uso em qualidade de via
os.makedirs(os.path.join(PROJECT_ROOT, "Pavimento", "outputs"), exist_ok=True)

df_preds = pd.DataFrame({
    "y_true_pav": y_test,
    "y_pred_pav_RF": y_pred_test
}, index=X_test.index)

OUTFILE = os.path.join(PROJECT_ROOT, "Pavimento", "outputs", "rf_pavimento_test_preds.csv")
df_preds.to_csv(OUTFILE, index=True)

print("\nSalvo em:")
print(OUTFILE)
