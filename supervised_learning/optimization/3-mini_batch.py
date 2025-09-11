#!/usr/bin/env python3
"""Mini-batch creation module for neural network training."""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Creates mini-batches to be used for training a neural network using
    mini-batch gradient descent."""
    m = X.shape[0]
    
    shuffled_X, shuffled_Y = shuffle_data(X, Y)
    
    mini_batches = []
    num_complete_minibatches = m // batch_size
    
    for k in range(num_complete_minibatches):
        X_batch = shuffled_X[k * batch_size:(k + 1) * batch_size]
        Y_batch = shuffled_Y[k * batch_size:(k + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))
    
    if m % batch_size != 0:
        X_batch = shuffled_X[num_complete_minibatches * batch_size:]
        Y_batch = shuffled_Y[num_complete_minibatches * batch_size:]
        mini_batches.append((X_batch, Y_batch))
    
    return mini_batches
