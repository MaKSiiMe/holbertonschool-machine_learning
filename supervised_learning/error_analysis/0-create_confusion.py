#!/usr/bin/env python3
"""0. Create Confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix"""
    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    classes = labels.shape[1]

    confusion = np.zeros((classes, classes))

    for true, pred in zip(true_labels, pred_labels):
        confusion[true, pred] += 1

    return confusion
