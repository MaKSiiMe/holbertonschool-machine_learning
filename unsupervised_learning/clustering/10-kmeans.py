#!/usr/bin/env python3
"""K-means clustering using sklearn"""

import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters

    Returns:
        C: numpy.ndarray of shape (k, d) containing centroid means
        clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    # Create and fit K-means model
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_model.fit(X)

    # Get centroids and labels
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_

    return C, clss
