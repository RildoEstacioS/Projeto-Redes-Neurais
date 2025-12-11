import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# Ajuste de paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from _1_config import DATAPATH, PROJECT_ROOT

print("DATAPATH:", DATAPATH)
print("PROJECT_ROOT:", PROJECT_ROOT)

data_dir = os.path.join(PROJECT_ROOT, "Qualidade", "data")
print("data_dir:", data_dir)


X_train_seq = np.load(os.path.join(data_dir, "X_train_qual_seq_pavprevLSTM.npy"))
y_train_seq = np.load(os.path.join(data_dir, "y_train_qual_seq_pavprevLSTM.npy"))




X_test_seq  = np.load(os.path.join(data_dir, "X_test_qual_seq_pavprevLSTM.npy"))
y_test_seq  = np.load(os.path.join(data_dir, "y_test_qual_seq_pavprevLSTM.npy"))

print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
print("X_test_seq :", X_test_seq.shape,  "y_test_seq :", y_test_seq.shape)


n_timesteps = X_train_seq.shape[1]
n_features  = X_train_seq.shape[2]
n_classes   = len(np.unique(y_train_seq))

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dense(n_classes, activation="softmax"))

opt = Adam(learning_rate=0.0005)
model.compile(optimizer=opt,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary())

classes = np.unique(y_train_seq)
class_weights = compute_class_weight(class_weight="balanced",
                                     classes=classes,
                                     y=y_train_seq)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
print("class_weight:", class_weight_dict)


es = EarlyStopping(monitor="val_loss",
                   patience=5,
                   restore_best_weights=True)

history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=20,
    batch_size=64,
    callbacks=[es],
    class_weight=class_weight_dict,
    verbose=2
)

y_pred_probs = model.predict(X_test_seq, verbose=0)
y_pred = y_pred_probs.argmax(axis=1)

print("Acurácia final:", accuracy_score(y_test_seq, y_pred))
print("Relatório de classificação:")
print(classification_report(y_test_seq, y_pred))
print("Matriz de confusão:")
print(confusion_matrix(y_test_seq, y_pred))

models_dir = os.path.join(PROJECT_ROOT, "Qualidade", "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "lstm_qualidade_pavprevLSTM.h5")
model.save(model_path)
print("Modelo salvo em:", model_path)
