import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import DATAPATH, PROJECT_ROOT

# Carrega dados
X_train = pd.read_csv(os.path.join(DATAPATH, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATAPATH, "y_train.csv")).squeeze()

X_test = pd.read_csv(os.path.join(DATAPATH, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATAPATH, "y_test.csv")).squeeze()

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)

# Normalização
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Treina MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(50, 25),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
)
mlp.fit(X_train_s, y_train)

# Teste
y_pred_test = mlp.predict(X_test_s)
print("Acurácia:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))

# Salva predições
os.makedirs(os.path.join(PROJECT_ROOT, "Pavimento", "outputs"), exist_ok=True)

df_preds = pd.DataFrame({
    "y_true_pav": y_test,
    "y_pred_pav_MLP": y_pred_test
}, index=X_test.index)

OUTFILE = os.path.join(PROJECT_ROOT, "Pavimento", "outputs", "mlp_pavimento_test_preds.csv")
df_preds.to_csv(OUTFILE, index=True)

print("\nSalvas em:")
print(OUTFILE)
