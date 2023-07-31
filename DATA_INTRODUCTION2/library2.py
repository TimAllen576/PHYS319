# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:59:45 2023

@author: tal75
"""

import numpy as np

def Principal_Component_Analysis(data):
    """Finds the principal components of some data
    Normalizes, calculates covariance, uses numpy functions to find eigenvalues and eigenvectors then sorts by size of eigenvalue
    
    Input: data
    Returns: [eigen_values,eigen_vectors,anomaly,covariance] """
    anomaly = np.empty(np.shape(data))
    for (row, col), val in np.ndenumerate(data):
        anomaly[:,col] = data[:,col] - np.mean(data[:,col])
    covariance = np.matmul(np.transpose(anomaly), anomaly)/(len(anomaly)-1)
    unord_eigen_values, unord_eigen_vectors = np.linalg.eigh(covariance)
    sort_mask = np.argsort(unord_eigen_values)
    eigen_values,eigen_vectors = unord_eigen_values[sort_mask], unord_eigen_vectors[:,sort_mask]
    return [eigen_values,eigen_vectors,anomaly,covariance]
