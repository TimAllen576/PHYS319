# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:59:45 2023

@author: tal75
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.special import betainc
from scipy.stats import ttest_ind
from seaborn import displot


        
def mean_and_num(data):
    "Finds the mean and number of entries then returns both"
    mean = np.mean(data)
    num = data.size
    return mean, num

def stderrormean(data1, data2, mean1, mean2, num1, num2):
    "Finds the std error"
    sum_difs = np.sum((data1-mean1)**2) + np.sum((data2-mean2)**2)
    tot_num  = num1+num2+2
    inv_nums = 1/num1 +1/num2
    s_d = np.sqrt((inv_nums)*sum_difs/tot_num)
    return s_d
    
def t_valsd(mean1, mean2, s_d):
    """
    Computes t value by difference of means divides by standard error of means
    """
    t = (mean1 - mean2)/s_d
    return t


def incbeta_f(num1,num2,t):
    "Computes the uncomplete beta function"
    df= num1 + num2 + 2
    #p_val = betainc(df/(df+t**2), df/2, 0.5)       # equation as written
    p_val = betainc(df/2, 0.5, df/(df+t**2))        # equation corrected for transcription errors
    return p_val
    
def sig_dif_means(filename1, filename2):
    "Does the first thing"
    SampleA = np.genfromtxt(filename1)
    SampleA = SampleA[~np.isnan(SampleA)] # removes nans
    SampleB = np.genfromtxt(filename2)
    SampleB = SampleB[~np.isnan(SampleB)] # removes nans
    p_val = ttest_ind(SampleA, SampleB, equal_var= False).pvalue
    print(f"With Scipy probability functions the p-value that the two samples {filename1}, {filename2} have identical means is:  {p_val}")
    mean1, num1 = mean_and_num(SampleA)
    mean2, num2 = mean_and_num(SampleB)
    s_d = stderrormean(SampleA, SampleB, mean1, mean2, num1, num2)
    t =t_valsd(mean1, mean2, s_d)
    beta_f = incbeta_f(num1,num2,t)
    print(f"Using the given functions the p-value that the two samples have identical means is: {beta_f}")
    """
    print(f'With t-value:{t}')
    SampleA = np.append(SampleA, np.full(len(SampleB)-len(SampleA), np.nan))
    SampleA = SampleA.reshape(len(SampleA), 1)
    SampleB = SampleB.reshape(len(SampleB), 1)
    sample_df = pd.DataFrame(np.concatenate((SampleA, SampleB), axis=1), columns= ('SampleA', 'SampleB'))
    displot(data=sample_df)
    plt.show()
    """
    
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
"""
def model_func_res(params, y_obs, x):
    "Takes data and computes the residual sum of squares with relation to a specified model, template"
    y_model = model_func(params, x)
    return np.sum((y_obs-y_model)**2)
"""