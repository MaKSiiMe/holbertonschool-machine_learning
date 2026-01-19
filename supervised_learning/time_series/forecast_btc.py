#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


def load_data(max_samples=None):
    files = [
        "btc_preprocessed_train.npz",
        "btc_preprocessed_val.npz",
        "btc_preprocessed_test.npz"
    ]
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Fichier manquant : {f}\n"
                "Assurez-vous d'avoir exécuté preprocess_data.py avec les "
                "bons fichiers CSV et que les fichiers prétraités existent "
                "dans le répertoire courant."
            )
    train = np.load(files[0])
    val = np.load(files[1])
    test = np.load(files[2])
    X_train, y_train = train['X'], train['y']
    X_val, y_val = val['X'], val['y']
    X_test, y_test = test['X'], test['y']
    # Limite le nombre d'échantillons si demandé
    if max_samples is not None:
        X_train, y_train = X_train[:max_samples], y_train[:max_samples]
        X_val, y_val = X_val[:max_samples], y_val[:max_samples]
        X_test, y_test = X_test[:max_samples], y_test[:max_samples]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def make_dataset(X, y, batch_size=16, shuffle=True):
    # batch_size réduit pour limiter la mémoire
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(X)))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        keras.layers.LSTM(32),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def check_data(X, y):
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Les données ou les cibles contiennent des NaN.")
    if np.isinf(X).any() or np.isinf(y).any():
        raise ValueError(
            "Les données ou les cibles contiennent des valeurs infinies."
        )


if __name__ == "__main__":
    # Limitez le nombre d'échantillons pour debug ou machine limitée
    max_samples = 5000  # Ajustez selon la RAM disponible
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
        max_samples=max_samples
    )
    check_data(X_train, y_train)
    check_data(X_val, y_val)
    check_data(X_test, y_test)
    train_ds = make_dataset(X_train, y_train)
    val_ds = make_dataset(X_val, y_val, shuffle=False)
    test_ds = make_dataset(X_test, y_test, shuffle=False)
    model = build_model(X_train.shape[1:])
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=10)
    loss, mae = model.evaluate(test_ds)
    print(
        f"Test MSE: {loss:.4f}, Test MAE: {mae:.4f}"
    )
    model.save("btc_rnn_model.keras")
