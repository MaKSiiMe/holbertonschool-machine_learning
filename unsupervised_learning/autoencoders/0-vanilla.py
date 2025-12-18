#!/usr/bin/env python3
"""0. "Vanilla" Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Crée un autoencoder simple (vanilla) avec Keras.

    Args:
        input_dims (int): dimension de l'entrée
        hidden_layers (list): nb de node pour chaque hidden layer de l'encodeur
        latent_dims (int): dimension de l'espace latent

    Returns:
        encoder, decoder, auto
    """
    # Encoder
    input_encoder = keras.Input(shape=(input_dims,))
    x = input_encoder
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    # Decoder
    input_decoder = keras.Input(shape=(latent_dims,))
    x = input_decoder
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    output_decoder = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=input_decoder, outputs=output_decoder)

    # Autoencoder
    input_auto = keras.Input(shape=(input_dims,))
    encoded = encoder(input_auto)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=input_auto, outputs=decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
