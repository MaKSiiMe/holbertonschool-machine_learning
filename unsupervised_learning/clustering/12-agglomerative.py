#!/usr/bin/env python3
"""Agglomerative clustering with Ward linkage"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: maximum cophenetic distance for all clusters

    Returns:
        clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    # Perform hierarchical clustering with Ward linkage
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method='ward')

    # Get cluster indices using fcluster with distance criterion
    clss = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, t=dist, criterion='distance'
    )

    # Display dendrogram with different colors for each cluster
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(
        linkage_matrix,
        color_threshold=dist
    )
    plt.show()

    return clss
