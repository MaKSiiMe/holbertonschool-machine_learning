#!/usr/bin/env python3
"""2. Convolutional Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Crée un autoencoder convolutionnel.

    Args:
        input_dims (tuple): dimensions de l'entrée (H, W, C)
        filters (list): nb de filtres pour chaque couche conv de l'encodeur
        latent_dims (tuple): dimensions de l'espace latent

    Returns:
        encoder, decoder, auto
    """
    # Encoder
    input_encoder = keras.Input(shape=input_dims)
    x = input_encoder
    for f in filters:
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    shape_before_flatten = keras.backend.int_shape(x)[1:]
    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(units=int(keras.backend.prod(latent_dims)),
                                activation='relu')(x)
    latent = keras.layers.Reshape(latent_dims)(latent)
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    # Decoder
    input_decoder = keras.Input(shape=latent_dims)
    x = keras.layers.Flatten()(input_decoder)
    x = keras.layers.Dense(units=int(keras.backend.prod(shape_before_flatten)),
                           activation='relu')(x)
    x = keras.layers.Reshape(shape_before_flatten)(x)
    for i, f in enumerate(reversed(filters)):
        if i < len(filters) - 2:
            x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                    activation='relu', padding='same')(x)
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
        elif i == len(filters) - 2:
            x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                    activation='relu', padding='valid')(x)
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
        else:
            x = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                                    activation='sigmoid', padding='same')(x)
    decoder = keras.Model(inputs=input_decoder, outputs=x)

    # Autoencoder
    input_auto = keras.Input(shape=input_dims)
    encoded = encoder(input_auto)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=input_auto, outputs=decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
