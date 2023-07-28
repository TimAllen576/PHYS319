# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:59:45 2023

@author: tal75
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.special import betainc
from scipy.stats import ttest_ind


def load_file(filename):
    "Loads the data"
    data = np.genfromtxt(filename)
    return data
        
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
    p_val = betainc(df/(df+t**2), df/2, 0.5)
    return p_val
    
def sig_dif_means(filename1, filename2):
    "Does the first thing"
    SampleA = load_file(filename1)
    SampleA = SampleA[~np.isnan(SampleA)] # removes nans
    SampleB = load_file(filename2)
    SampleB = SampleB[~np.isnan(SampleB)] # removes nans
    #print(SampleA)
    #print(SampleB)
    p_val = ttest_ind(SampleA, SampleB, equal_var= False).pvalue
    print(f"The p-value that the two samples {filename1}, {filename2} have identical means is:  {p_val}")
    mean1, num1 = mean_and_num(SampleA)
    mean2, num2 = mean_and_num(SampleB)
    s_d = stderrormean(SampleA, SampleB, mean1, mean2, num1, num2)
    t =t_valsd(mean1, mean2, s_d) 
    beta_f = incbeta_f(num1,num2,t)
    print(f"Using the given functions we get: {beta_f}")
    
def Principal_Component_Analysis(data):
    """Probably does a lot, I should come back to this comment"""
    anomaly = np.empty(np.shape(data))
    for (row, col),x in np.ndenumerate(data):
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

"""
def plotter_template():
    "To be used as template only"
    plt.figure()
    plt.errorbar(x,y,xerrr=None,yerr=sigma,linestyle='none', marker='*', label="Data")
    plt.plot(xx,np.polyval(p,xx),'m-', label= "Original fit")
    #fit a 4-coefficient polynomial (i.e. a cubic) to the data
    #and quantify the quality of the fit
    alpha0 = np.array([1.0, -10.0, -10.0, 20.0])
    uncert = 25
    res_lsq = least_squares(cubic_res, alpha0, args=(x, y, uncert))
    print(f"This gives a chi squared value of {res_lsq.cost}.\nThe expected value is {len(x)-4} with variance {2*(len(x)-4)}.")
    y_lsq = cubic_fun(res_lsq.x, xx)
    plt.plot(xx, y_lsq, label= "My fit")
    plt.legend()
    plt.show(block=False)
"""