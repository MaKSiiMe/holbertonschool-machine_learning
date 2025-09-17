#!/usr/bin/env python3
"""4. F1 Score"""
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix"""
    sens = sensitivity(confusion)
    prec = precision(confusion)

    f1_scores = np.zeros(sens.shape[0])

    for i in range(len(sens)):
        if (prec[i] + sens[i]) > 0:
            f1_scores[i] = 2 * (prec[i] * sens[i]) / (prec[i] + sens[i])
        else:
            f1_scores[i] = 0

    return f1_scores
