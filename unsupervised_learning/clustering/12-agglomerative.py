#!/usr/bin/env python3
"""Agglomerative clustering with Ward linkage"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Runs agglomerative clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: maximum cophenetic distance

    Returns:
        clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method='ward')

    clss = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, t=dist, criterion='distance'
    )

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(
        linkage_matrix,
        color_threshold=dist
    )
    plt.show()

    return clss
