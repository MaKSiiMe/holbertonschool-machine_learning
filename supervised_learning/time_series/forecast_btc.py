#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import joblib
import os

# --- PARAMÈTRES ---
SEQ_LEN = 24       # Fenêtre de 24h
PRED_HORIZON = 1   # Prédiction à T+1h
BATCH_SIZE = 64    # Optimisé pour GPU
EPOCHS = 15        # Avec EarlyStopping


def create_dataset_windows(data, seq_len):
    """Crée les fenêtres glissantes X et la cible y"""
    X, y = [], []
    # On assume que la colonne 'Close' est à l'index 0
    target_col_index = 0

    # data est un numpy array (n_samples, n_features)
    for i in range(len(data) - seq_len - PRED_HORIZON):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len + PRED_HORIZON - 1, target_col_index])

    return np.array(X), np.array(y)


def main():
    # 1. Chargement
    if not os.path.exists('preprocessed_btc.csv'):
        print("Erreur: Lancez d'abord preprocess_data.py")
        return

    print("--- Chargement des données ---")
    df = pd.read_csv('preprocessed_btc.csv', index_col=0, parse_dates=True)
    data = df.values.astype('float32')
    scaler = joblib.load('btc_scaler.save')

    # 2. Préparation des fenêtres
    print("--- Création des séquences ---")
    X, y = create_dataset_windows(data, SEQ_LEN)

    # 3. Split Chronologique (Train 80% / Val 10% / Test 10%)
    n = len(X)
    train_split = int(n * 0.8)
    val_split = int(n * 0.9)

    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:val_split], y[train_split:val_split]
    X_test, y_test = X[val_split:], y[val_split:]

    # Création tf.data.Dataset
    def make_tf_ds(X, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(10000)
        # .repeat() permet de gérer les epochs manuellement via steps_per_epoch
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).repeat()

    train_ds = make_tf_ds(X_train, y_train, shuffle=True)
    val_ds = make_tf_ds(X_val, y_val, shuffle=False)
    test_ds = make_tf_ds(X_test, y_test, shuffle=False)

    steps_per_epoch = len(X_train) // BATCH_SIZE
    val_steps = len(X_val) // BATCH_SIZE

    # 4. Modèle
    print("--- Construction du Modèle ---")
    model = Sequential([
        # Couche 1 : Capture les patterns complexes
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
        Dropout(0.3),
        # Couche 2 : Synthétise l'info
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        # Sortie
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=4, restore_best_weights=True
    )

    print("--- Entraînement ---")
    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[early_stop]
    )

    # 5. Évaluation et Graphique
    print("--- Génération du Graphique Final ---")

    # On prend les 200 derniers points du Test Set pour y voir clair
    last_n = 200
    X_sample = X_test[-last_n:]
    y_sample_true = y_test[-last_n:]

    # Prédiction
    preds = model.predict(X_sample, verbose=0)

    # INVERSE TRANSFORM (C'est ici qu'on retrouve les vrais prix)
    # Le scaler a besoin de 3 colonnes (Close, Volume, MA24), on crée une matrice dummy
    def inverse_prices(target_array, scaler):
        dummy = np.zeros((len(target_array), scaler.n_features_in_))
        dummy[:, 0] = target_array.flatten()  # On met nos prix dans la colonne 0 (Close)
        return scaler.inverse_transform(dummy)[:, 0]

    real_prices = inverse_prices(y_sample_true, scaler)
    predicted_prices = inverse_prices(preds, scaler)

    # Lissage léger de la prédiction pour le visuel (Rolling mean sur 3 points)
    predicted_smooth = pd.Series(predicted_prices).rolling(3, center=True, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(real_prices, label='Prix Réel (USD)', color='blue', linewidth=1.5)
    plt.plot(predicted_smooth, label='Prédiction (USD)', color='orange', linestyle='--', linewidth=1.5)
    plt.title(f"Prédiction BTC vs Réalité (Zoom sur les {last_n} dernières heures)")
    plt.xlabel("Heures")
    plt.ylabel("Prix ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig('btc_prediction_graph.png')
    model.save('btc_model.keras')

    print("✅ Graphique sauvegardé : btc_prediction_graph.png")
    print("✅ Modèle sauvegardé : btc_model.keras")


if __name__ == '__main__':
    main()
