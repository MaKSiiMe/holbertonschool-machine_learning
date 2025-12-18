#!/usr/bin/env python3
"""3. Variational Autoencoder"""
import tensorflow.keras as keras
import tensorflow as tf


def sampling(args):
    """Échantillonnage par la réparamétrisation de Gaussienne."""
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Crée un variational autoencoder (VAE).

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
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(inputs=input_encoder, outputs=[z, z_mean, z_log_var])

    # Decoder
    input_decoder = keras.Input(shape=(latent_dims,))
    x = input_decoder
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    output_decoder = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=input_decoder, outputs=output_decoder)

    # VAE Model
    input_auto = keras.Input(shape=(input_dims,))
    z, z_mean, z_log_var = encoder(input_auto)
    reconstructed = decoder(z)
    auto = keras.Model(inputs=input_auto, outputs=reconstructed)

    # Custom loss (binary crossentropy + KL divergence)
    reconstruction_loss = keras.losses.binary_crossentropy(
        input_auto, reconstructed
    )
    reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=1)
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
    )
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
