import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import save_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from _1_config import DATAPATH, PROJECT_ROOT

print("DATAPATH:", DATAPATH)
print("PROJECT_ROOT:", PROJECT_ROOT)

#Carrega sequências
X_train_seq = np.load(os.path.join(DATAPATH, "X_train_seq.npy"))
y_train_seq = np.load(os.path.join(DATAPATH, "y_train_seq.npy"))

X_test_seq = np.load(os.path.join(DATAPATH, "X_test_seq.npy"))
y_test_seq = np.load(os.path.join(DATAPATH, "y_test_seq.npy"))

print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
print("X_test_seq: ", X_test_seq.shape,  y_test_seq.shape)


# Numero de Classes
n_classes = 3
y_train_cat = to_categorical(y_train_seq, num_classes=n_classes) # 
y_test_cat  = to_categorical(y_test_seq,  num_classes=n_classes)

# Modelo LSTM
timesteps = X_train_seq.shape[1] # Número de timesteps na sequência
n_features = X_train_seq.shape[2] # Número de features por timestep

model = Sequential([
    LSTM(32, input_shape=(timesteps, n_features), return_sequences=False), # Camada LSTM
    Dropout(0.3), # Dropout para evitar overfitting
    Dense(32, activation='relu'), # Tipo de ativação: ReLU
    Dense(n_classes, activation='softmax') # Classificação multi-classe
])

model.compile(optimizer='adam', # Otimizador Adam
              loss='categorical_crossentropy', # Função de perda para multi-classe
              metrics=['accuracy']) # Métrica de acurácia

early_stop = EarlyStopping(monitor='val_loss', # Monitora a perda na validação
                           patience=3, # Paciência de 3 épocas
                           restore_best_weights=True) # Restaura os melhores pesos

history = model.fit( # Treinamento do modelo
    X_train_seq, y_train_cat, # Dados de treinamento
    epochs=10, # Número de épocas
    batch_size=256, # Tamanho do batch
    callbacks=[early_stop], # Callbacks
    verbose=2 # Verbosidade do treinamento
)

# Avaliação no teste
test_loss, test_acc = model.evaluate(X_test_seq, y_test_cat, verbose=0)
print(f"\nAcurácia final no teste (LSTM pavimento): {test_acc:.4f}")

y_pred_test = model.predict(X_test_seq).argmax(axis=1)

print("\nRelatório de classificação no teste:")
print(classification_report(y_test_seq, y_pred_test))
print("Matriz de confusão:")
print(confusion_matrix(y_test_seq, y_pred_test))
print("Acurácia (sklearn):", accuracy_score(y_test_seq, y_pred_test))


models_dir = os.path.join(PROJECT_ROOT, "Pavimento", "models")
os.makedirs(models_dir, exist_ok=True)
MODEL_PATH = os.path.join(models_dir, "lstm_pavimento.h5")
model.save(MODEL_PATH)
print("\nLSTM de pavimento salvo em:", MODEL_PATH)