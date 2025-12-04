#!/usr/bin/env python3
"""GMM clustering using sklearn"""

import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters

    Returns:
        pi: numpy.ndarray of shape (k,) containing cluster priors
        m: numpy.ndarray of shape (k, d) containing centroid means
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices
        clss: numpy.ndarray of shape (n,) containing cluster indices
        bic: BIC value for the model
    """
    # Create and fit GMM model
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)
    gmm_model.fit(X)

    # Get parameters
    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
