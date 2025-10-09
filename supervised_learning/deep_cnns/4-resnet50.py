#!/usr/bin/env python3
"""Module for building ResNet-50 architecture"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)

    Returns:
        the keras model
    """
    # He normal initializer with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # Input layer
    X_input = K.Input(shape=(224, 224, 3))

    # Initial convolution: 7x7 conv, stride 2
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Max pooling: 3x3, stride 2
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same')(X)

    # Stage 1 (conv2_x): 1 projection + 2 identity blocks
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 2 (conv3_x): 1 projection + 3 identity blocks
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Stage 3 (conv4_x): 1 projection + 5 identity blocks
    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Stage 4 (conv5_x): 1 projection + 2 identity blocks
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Average pooling
    X = K.layers.AveragePooling2D(pool_size=(7, 7), padding='same')(X)

    # Output layer: fully connected with 1000 classes
    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(X)

    # Create model
    model = K.Model(inputs=X_input, outputs=X)

    return model
