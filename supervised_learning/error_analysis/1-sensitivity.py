#!/usr/bin/env python3
"""1. Sensitivity"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix"""
    classes = confusion.shape[0]

    sensitivities = np.zeros(classes)

    for i in range(classes):
        tp = confusion[i, i]
        actual_positives = np.sum(confusion[i, :])

        if actual_positives > 0:
            sensitivities[i] = tp / actual_positives
        else:
            sensitivities[i] = 0

    return sensitivities
