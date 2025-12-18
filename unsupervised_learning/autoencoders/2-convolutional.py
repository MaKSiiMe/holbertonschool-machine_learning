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
    latent = x
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    # Decoder
    input_decoder = keras.Input(shape=latent.shape[1:])
    x = input_decoder
    for f in reversed(filters):
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
    # Dernière couche pour revenir au nombre de canaux d'entrée, sans upsampling
    x = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                            activation='sigmoid', padding='same')(x)
    # Ajustement de la taille de sortie si nécessaire
    height_diff = x.shape[1] - input_dims[0]
    width_diff = x.shape[2] - input_dims[1]
    if height_diff > 0 or width_diff > 0:
        x = keras.layers.Cropping2D(
            ((height_diff // 2, height_diff - height_diff // 2),
             (width_diff // 2, width_diff - width_diff // 2))
        )(x)
    decoder = keras.Model(inputs=input_decoder, outputs=x)

    # Autoencoder
    input_auto = keras.Input(shape=input_dims)
    encoded = encoder(input_auto)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=input_auto, outputs=decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
