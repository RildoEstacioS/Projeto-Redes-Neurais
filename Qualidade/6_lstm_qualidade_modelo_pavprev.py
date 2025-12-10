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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import DATAPATH, PROJECT_ROOT

print("DATAPATH:", DATAPATH)
print("PROJECT_ROOT:", PROJECT_ROOT)

data_dir = os.path.join(PROJECT_ROOT, "Qualidade", "data")
print("data_dir:", data_dir)

# Carrega sequências já salvas
X_train_seq = np.load(os.path.join(data_dir, "X_train_qual_seq_pavprev.npy")) 
y_train_seq = np.load(os.path.join(data_dir, "y_train_qual_seq_pavprev.npy"))
X_test_seq  = np.load(os.path.join(data_dir, "X_test_qual_seq_pavprev.npy"))
y_test_seq  = np.load(os.path.join(data_dir, "y_test_qual_seq_pavprev.npy"))

print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
print("X_test_seq: ", X_test_seq.shape,  "y_test_seq: ",  y_test_seq.shape)

# Define modelo LSTM reforçado
n_timesteps = X_train_seq.shape[1] # número de timesteps na sequência
n_features = X_train_seq.shape[2] # número de features por timestep
n_classes = len(np.unique(y_train_seq)) # número de classes alvo 

model = Sequential() # Define modelo sequencial
model.add(LSTM(64, return_sequences=True, input_shape=(n_timesteps, n_features))) # Primeira camada LSTM com 64 unidades
model.add(Dropout(0.3))  # Dropout para regularização
model.add(LSTM(64)) # Segunda camada LSTM com 64 unidades
model.add(Dropout(0.3)) #  Dropout para regularização
model.add(Dense(64, activation="relu")) # Camada densa intermediária com ReLU
model.add(Dense(n_classes, activation="softmax")) # Camada de saída com softmax para classificação multi-classe

opt = Adam(learning_rate=0.0005) # Otimizador Adam com taxa de aprendizado ajustada
model.compile(optimizer=opt, # Compila modelo
              loss="sparse_categorical_crossentropy", # perda para classificação multi-classe
              metrics=["accuracy"]) # Métricas de avaliação - Acuracia...

# class_weight para desbalanceamento
classes = np.unique(y_train_seq) # classes únicas no y_train_seq
class_weights = compute_class_weight(class_weight="balanced", # calcula pesos para cada classe
                                     classes=classes, # classes únicas
                                     y=y_train_seq) # vetor de classes do treino
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)} # converte para dicionário
print("class_weight:", class_weight_dict) # pesos das classes

es = EarlyStopping(monitor="val_loss", # EarlyStopping para evitar overfitting
                   patience=5, # paciência de 5 épocas
                   restore_best_weights=True) # restaura melhores pesos

# Treinamento do modelo
history = model.fit( 
    X_train_seq, y_train_seq, 
    validation_data=(X_test_seq, y_test_seq),
    epochs=20, # 20 épocas
    batch_size=64, # tamanho do batch 64
    callbacks=[es], 
    class_weight=class_weight_dict, # aplica pesos das classes
    verbose=2, # mostra progresso
)

# Avaliação no teste
y_pred_probs = model.predict(X_test_seq, verbose=0)
y_pred = y_pred_probs.argmax(axis=1)

print("\nAcurácia final:", accuracy_score(y_test_seq, y_pred))
print("\nRelatório de classificação:")
print(classification_report(y_test_seq, y_pred))
print("\nMatriz de confusão:")
print(confusion_matrix(y_test_seq, y_pred))

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import PROJECT_ROOT

models_dir = os.path.join(PROJECT_ROOT, "Qualidade", "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "lstm_qualidade_pavprev.h5")
model.save(model_path)
print("\nModelo salvo em:", model_path)
