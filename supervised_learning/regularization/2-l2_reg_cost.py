#!/usr/bin/env python3
"""2. L2 Regularization Cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the cost of a neural network with L2 regularization"""
    l2_losses = model.losses

    costs_per_layer = [cost + l2_loss for l2_loss in l2_losses]

    return tf.stack(costs_per_layer)
