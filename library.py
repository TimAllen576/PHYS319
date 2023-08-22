# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:59:45 2023

@author: tal75
"""


import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn
#import cartopy
#import serial
#import glob
#from matplotlib.cm import ScalarMappable
#from matplotlib.colors import Normalize
from scipy.special import betainc


def  standard_error_of_difference_of_means(dataset1, dataset2):
    """Estimate the standard error of the difference of the means
    Inputs: dataset1 (1-D array)
            dataset2 (1-D array)
    Returns: S_D (float)
    """
    norm_data1 = dataset1 - dataset1.mean
    norm_data2 = dataset2 - dataset2.mean
    num1, num2 = dataset1.size, dataset2.size
    sum_difs = np.sum((norm_data1)**2) + np.sum((norm_data2)**2)
    tot_num  = num1+num2+2
    inv_nums = 1/num1 +1/num2
    s_d = np.sqrt( inv_nums * sum_difs / tot_num) 
    return s_d
    
def one_sample_students_t_value(sample, population):
    """
    Computes students t value by difference of means divides by standard error of difference of means
    Inputs: sample : 1-D array
            population : 1-D array
    Returns: students t value : float
    """
    s_d = standard_error_of_difference_of_means(sample, population)
    t = (sample.mean - population.mean)/s_d
    return t

def pvalue_from_incbeta(dataset1, dataset2):
    """Computes the p value for null hypothesis using the incomplete beta function and students t-distribution
    Inputs: dataset1 : 1-D array
            dataset2 : 1-D array
    Returns: p value : float
    """
    num1, num2 = dataset1.size, dataset2.size
    t = one_sample_students_t_value(dataset1, dataset2)
    df= num1 + num2 + 2
    p_val = betainc(df/2, 0.5, df/(df+t**2))
    return p_val

def pearsons_correlation_coefficient(dataset1, dataset2):
    """Computes Pearson's Correlation Coefficient for two simple datasets with one to one data
    Inputs: dataset1 : 1-D array
            dataset2 : 1-D array
    Returns: r value : float
    """
    norm_dataset1 = dataset1 - dataset1.mean
    norm_dataset2 = dataset2 - dataset2.mean
    sum_top = np.sum(norm_dataset1*norm_dataset2)
    sum_x_bot = np.sqrt(np.sum(norm_dataset1**2))
    sum_y_bot = np.sqrt(np.sum(norm_dataset2**2))
    r = sum_top / (sum_x_bot*sum_y_bot)
    return r

def one_sample_students_t_value_from_r(dataset1, dataset2):
    """
    Computes students t value from Pearson's Correlation Coefficient and number of data points. Only suitable for small N (eg. N < 200)
    Inputs: dataset1 : 1-D array
            dataset2 : 1-D array
    Returns: students t value : float
    """
    r = pearsons_correlation_coefficient(dataset1, dataset2)
    n = dataset1.size + dataset2.size
    t = r*np.sqrt((n-2)/(1-r**2))
    return t

def stat_sig(p):
    """Prints if a p value is statistically significant (95%)
    Input: p value : float
    Returns: None
    """
    if p < 0.05:
        print(f"This is statistically significant with p = {p}")
    else:
        print(f"This is not statistically significant with p = {p}")

def Principal_Component_Analysis(dataset):
    """Finds the principal components of some data
    Normalizes, calculates covariance, uses numpy functions to find eigenvalues and eigenvectors then sorts by size of eigenvalue
    Input: dataset          : 2-D array, shape (a, b)
    Returns: [eigen_values  : 1-D array, shape (b,)
            eigen_vectors   : 2-D array, shape (b, b)
            centred_data         : 2-D array, shape (a, b)
            covariance_matrix]     : 2-D array, shape (b, b)
    """
    centred_data = np.empty(np.shape(dataset))
    for (row, col), val in np.ndenumerate(dataset):
        centred_data[:,col] = dataset[:,col] - np.mean(dataset[:,col])
    covariance_matrix = np.matmul(np.transpose(centred_data), centred_data)/(len(centred_data)-1)
    unord_eigen_values, unord_eigen_vectors = np.linalg.eigh(covariance_matrix)
    sort_mask = np.argsort(unord_eigen_values)
    eigen_values,eigen_vectors = unord_eigen_values[sort_mask], unord_eigen_vectors[:,sort_mask]
    return [eigen_values,eigen_vectors,centred_data,covariance_matrix]

def fourier_transform(dataset, sampling_interval):
    """Does a fourier transform of the data, cuts off duplicate parts and gives amplitude and frequency
    Inputs: dataset : 1-D array
            sampling_interval : float
    Returns: amplitudes : 1-D array
            frequencies : 1-D array
    """
    fft_length = len(dataset)
    fourier_transform = np.fft.fft(dataset)
    amplitudes = abs(fourier_transform[0:int(fft_length/2)]) * (2.0/fft_length)
    frequencies = np.arange(0,int(fft_length/2)) / (fft_length*sampling_interval)
    return amplitudes, frequencies

"""
def plotter(eigen_vectors, PC, latitude, longitude):
    "plotter template"
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.scatter(longitude, latitude, c=eof1, cmap=cmap, label="stations")
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax1, location='right', label="EOF 1", shrink=0.5) 
    plt.savefig('PCA4_worldmap.png')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.plot(PC[:,-1])
    
    plt.savefig('recognisable_graph_name.png')
"""

def main():
    "idk lol"

if __name__ == '__main__':
    main()