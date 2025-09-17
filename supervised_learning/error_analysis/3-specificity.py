#!/usr/bin/env python3
"""3. Specificity"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix"""
    classes = confusion.shape[0]

    specificities = np.zeros(classes)

    total_samples = np.sum(confusion)

    for i in range(classes):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp
        fn = np.sum(confusion[i, :]) - tp
        tn = total_samples - tp - fp - fn

        if (tn + fp) > 0:
            specificities[i] = tn / (tn + fp)
        else:
            specificities[i] = 0

    return specificities
