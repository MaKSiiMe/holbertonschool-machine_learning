#!/usr/bin/env python3
"""2. Precision"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix"""
    classes = confusion.shape[0]

    precisions = np.zeros(classes)

    for i in range(classes):
        tp = confusion[i, i]
        predicted_positives = np.sum(confusion[:, i])

        if predicted_positives > 0:
            precisions[i] = tp / predicted_positives
        else:
            precisions[i] = 0

    return precisions
