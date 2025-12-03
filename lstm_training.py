# Configuration pour éviter les erreurs de threading sur macOS
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif
import matplotlib.pyplot as plt

# Désactiver les warnings TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# Configuration TensorFlow pour macOS
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fonction pour générer une onde ECG simulée
def ecg_synthetique(t):
    return (
        0.1 * np.sin(2 * np.pi * t * 1) +  # Onde P
        -0.15 * np.exp(-((t - 0.25) ** 2) / 0.001) +  # Onde Q
        1.0 * np.exp(-((t - 0.3) ** 2) / 0.0005) +  # Pic R
        -0.2 * np.exp(-((t - 0.35) ** 2) / 0.001) +  # Onde S
        0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)  # Onde T
    )

print("=" * 60)
print("ENTRAÎNEMENT DU MODÈLE LSTM POUR RECONSTRUCTION ECG")
print("=" * 60)

# 1. Génération du signal d'entraînement
print("\n1. Génération du signal d'entraînement...")
t_single = np.linspace(0, 1, 500)
cycle = ecg_synthetique(t_single)

# Ajouter un peu de bruit
bruit = 0.02 * np.random.normal(size=cycle.shape)
cycle_bruite = cycle + bruit

# Créer 20 cycles
signal_complet = np.tile(cycle_bruite, 20)
print(f"   Signal généré : {len(signal_complet)} points ({len(signal_complet)//500} cycles)")

# 2. Préparation des séquences
print("\n2. Préparation des séquences d'entraînement...")
sequence_length = 50
X, y = [], []

for i in range(len(signal_complet) - sequence_length):
    X.append(signal_complet[i:i + sequence_length])
    y.append(signal_complet[i + sequence_length])

X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))
print(f"   X shape: {X.shape}, y shape: {y.shape}")

# Split train/validation
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
print(f"   Train: {len(X_train)} samples, Validation: {len(X_val)} samples")

# 3. Construction du modèle
print("\n3. Construction du modèle LSTM...")
model = Sequential([
    LSTM(64, input_shape=(sequence_length, 1), return_sequences=True),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("   ✓ Modèle construit")

# 4. Entraînement (SANS callbacks pour éviter les erreurs)
print("\n4. Entraînement du modèle...")
print("   (Cela peut prendre quelques minutes...)\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,  # Réduit à 30 pour plus rapide
    batch_size=32,
    verbose=1
)

# 5. Évaluation
print("\n5. Évaluation du modèle...")
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
print(f"   Train Loss (MSE): {train_loss:.6f}, MAE: {train_mae:.6f}")
print(f"   Validation Loss (MSE): {val_loss:.6f}, MAE: {val_mae:.6f}")

# 6. Sauvegarde du modèle
print("\n6. Sauvegarde du modèle...")
model.save('ecg_lstm_model.keras')
print("   ✓ Modèle sauvegardé : ecg_lstm_model.keras")

# 7. Visualisation de l'entraînement
print("\n7. Génération des graphiques...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Évolution de la loss pendant l\'entraînement')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Évolution de la MAE pendant l\'entraînement')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("   ✓ Graphique sauvegardé : training_history.png")

# 8. Test de prédiction
print("\n8. Test rapide de prédiction...")
y_pred = model.predict(X_val[:500], verbose=0)

plt.figure(figsize=(12, 5))
plt.plot(y_val[:500], label='Signal réel', linewidth=2)
plt.plot(y_pred[:500], label='Prédiction LSTM', linestyle='--', alpha=0.8)
plt.title('Test de prédiction du modèle LSTM')
plt.xlabel('Point')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig('prediction_test.png', dpi=150)
print("   ✓ Graphique sauvegardé : prediction_test.png")

print("\n" + "=" * 60)
print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
print("Vous pouvez maintenant utiliser le modèle dans edge_receiver_lstm.py")
print("=" * 60)