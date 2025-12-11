import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from _1_config import DATAPATH, PROJECT_ROOT


# ========= PAVIMENTO =========


def avaliar_rf_pavimento():
    X_train = pd.read_csv(os.path.join(DATAPATH, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATAPATH, "y_train.csv")).squeeze()
    X_test  = pd.read_csv(os.path.join(DATAPATH, "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(DATAPATH, "y_test.csv")).squeeze()

    rf = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    return {
        "tarefa": "pavimento",
        "modelo": "RF",
        "pav_prev_origem": "N/A",
        "acc": accuracy_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
    }


def avaliar_mlp_pavimento():
    X_train = pd.read_csv(os.path.join(DATAPATH, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATAPATH, "y_train.csv")).squeeze()
    X_test  = pd.read_csv(os.path.join(DATAPATH, "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(DATAPATH, "y_test.csv")).squeeze()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
    )
    mlp.fit(X_train_s, y_train)
    y_pred = mlp.predict(X_test_s)

    return {
        "tarefa": "pavimento",
        "modelo": "MLP",
        "pav_prev_origem": "N/A",
        "acc": accuracy_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
    }


def avaliar_svm_pavimento():
    X_train = pd.read_csv(os.path.join(DATAPATH, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATAPATH, "y_train.csv")).squeeze()
    X_test  = pd.read_csv(os.path.join(DATAPATH, "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(DATAPATH, "y_test.csv")).squeeze()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    svm = SVC(kernel="rbf", random_state=42)
    svm.fit(X_train_s, y_train)
    y_pred = svm.predict(X_test_s)

    return {
        "tarefa": "pavimento",
        "modelo": "SVM",
        "pav_prev_origem": "N/A",
        "acc": accuracy_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
    }


def avaliar_lstm_pavimento():
    # Usa as sequências geradas por 05_lstm_pavimento_sequencias.py
    X_test_seq = np.load(os.path.join(DATAPATH, "X_test_seq.npy"))
    y_test_seq = np.load(os.path.join(DATAPATH, "y_test_seq.npy"))

    from tensorflow.keras.models import load_model
    model_path = os.path.join(PROJECT_ROOT, "Pavimento", "models", "lstm_pavimento.h5")
    model = load_model(model_path)

    y_pred = model.predict(X_test_seq, verbose=0).argmax(axis=1)

    return {
        "tarefa": "pavimento",
        "modelo": "LSTM",
        "pav_prev_origem": "N/A",
        "acc": accuracy_score(y_test_seq, y_pred),
        "cm": confusion_matrix(y_test_seq, y_pred),
    }


# ========= QUALIDADE (pavimento previsto em cascata) =========


def _carrega_qual_RF():
    data_dir = os.path.join(PROJECT_ROOT, "Qualidade", "data")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_qual_pavprev_RF.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train_qual_pavprev_RF.csv")).squeeze()
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test_qual_pavprev_RF.csv"))
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test_qual_pavprev_RF.csv")).squeeze()
    return data_dir, X_train, y_train, X_test, y_test


def avaliar_rf_qualidade_pavprev_RF():
    _, X_train, y_train, X_test, y_test = _carrega_qual_RF()

    rf = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    return {
        "tarefa": "qualidade",
        "modelo": "RF",
        "pav_prev_origem": "RF",
        "acc": accuracy_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
    }


def avaliar_mlp_qualidade_pavprev_RF():
    _, X_train, y_train, X_test, y_test = _carrega_qual_RF()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        max_iter=100,
        random_state=42,
    )
    mlp.fit(X_train_s, y_train)
    y_pred = mlp.predict(X_test_s)

    return {
        "tarefa": "qualidade",
        "modelo": "MLP",
        "pav_prev_origem": "RF",
        "acc": accuracy_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
    }


def avaliar_svm_qualidade_pavprev_RF():
    _, X_train, y_train, X_test, y_test = _carrega_qual_RF()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    svm = SVC(kernel="rbf", random_state=42)
    svm.fit(X_train_s, y_train)
    y_pred = svm.predict(X_test_s)

    return {
        "tarefa": "qualidade",
        "modelo": "SVM",
        "pav_prev_origem": "RF",
        "acc": accuracy_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
    }


def avaliar_lstm_qualidade_pavprev_RF():
    """
    Avalia a LSTM de qualidade usando as sequências geradas a partir
    da base com pavimento previsto pela LSTM de pavimento.
    """
    data_dir = os.path.join(PROJECT_ROOT, "Qualidade", "data")

    # Arquivos gerados pelo script 5_lstm_qualidade_sequencias_pavprev.py (versão LSTM)
    X_test_seq = np.load(os.path.join(data_dir, "X_test_qual_seq_pavprevLSTM.npy"))
    y_test_seq = np.load(os.path.join(data_dir, "y_test_qual_seq_pavprevLSTM.npy"))

    from tensorflow.keras.models import load_model
    model_path = os.path.join(
        PROJECT_ROOT,
        "Qualidade",
        "models",
        "lstm_qualidade_pavprevLSTM.h5",
    )
    model = load_model(model_path)

    y_pred = model.predict(X_test_seq, verbose=0).argmax(axis=1)

    return {
        "tarefa": "qualidade",
        "modelo": "LSTM",
        "pav_prev_origem": "LSTM",
        "acc": accuracy_score(y_test_seq, y_pred),
        "cm": confusion_matrix(y_test_seq, y_pred),
    }
