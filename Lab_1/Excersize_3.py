# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:53:16 2023

@author: tal75
"""


import numpy as np
import matplotlib.pyplot as plt

from library import incbeta_f
from scipy.stats import pearsonr
#from scipy.special import betainc


def slice_data(filename):
    "Loads then splits three columns of data into x,y,z and returns it"
    data = np.genfromtxt(filename)
    x,y,z = data[:,0], data[:,1], data[:,2],
    return x,y,z

def plot_scatter(x, y, labelx, labely, units, caption):
    "plots a scatter graph"
    if units != "":
        units= f"({units})"
    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax1.set_xlabel(f"{labelx}{units}")
    ax1.set_ylabel(f"{labely}{units}")
    fig.text(0.5, 0.08, caption, wrap=True, horizontalalignment='center')
    ax1.scatter(x,y)
    fig.set_size_inches(7, 6, forward=True)
    plt.show(block = False)

def pearson_corr_coeff(x, y):
    "Takes x and y one to one data and returns r the Pearsons Correlation coefficient"
    sum_top = np.sum((x-np.mean(x))*(y-np.mean(y)))
    sum_x_bot = np.sqrt(np.sum((x-np.mean(x))**2))
    sum_y_bot = np.sqrt(np.sum((y-np.mean(y))**2))
    r = sum_top / (sum_x_bot*sum_y_bot)
    return r

def t_val2(r, n):
    """
    Computes t value from r value
    """
    t = r*np.sqrt((n-2)/(1-r**2))
    return t

def stat_sig(p):
    "Prints if a p value is statistically significant (95%)"
    if p < 0.05:
        print(f"This is statistically significant with p = {p}")
    else:
        print(f"This is not statistically significant with p = {p}")


def main(filename):
    "Does the thingymajig"
    x,y,z = slice_data(filename)
    plot_scatter(x, y, "x", "y", "", f'Scatter of x vs y from {filename}')
    plot_scatter(x, z, "x", "z", "", f'Scatter of x vs z from {filename}')
    r_y = pearson_corr_coeff(x, y)
    print(f"Correlation coefficient for x vs y is: {r_y}")
    t_y = t_val2(r_y, len(y))
    p_y = incbeta_f(len(x), len(y), t_y)
    stat_sig(p_y)
    r_z = pearson_corr_coeff(x, z)
    t_z = t_val2(r_z, len(z))
    print(f"Correlation coefficient for x vs z is: {r_z}")
    p_z = incbeta_f(len(x), len(z), t_z)
    stat_sig(p_z)
    plt.show()


main("DAex3.dat")