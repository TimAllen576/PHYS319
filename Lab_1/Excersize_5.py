# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 2023

@author: tal75
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin

INI_ARGS1= [10, 0, 7, 1/365.25, 2]
INI_ARGS2= [10, 0, 7, 1/365.25, 0]


def split_data(filename):
    """Loads the file and gets the good stuff
    Assumes that data is evenly spaced intervals from start of year
    Trims -99 values
    """
    temps_i = np.genfromtxt(filename, skip_header=21, usecols= range(1, 13))
    temps_r = np.reshape(temps_i, (-1, 1))
    temps_trim = np.delete(temps_r, np.where(temps_r == -99.0))
    return temps_trim

def model_func(params, t):
    """
    Returns temperature for a specific model
    Params should be of the form [A, B, C, f, phase]
    """
    return params[0]*np.cos(2*np.pi*params[3]*t-params[4])+params[1]*t + params[2]

def model_func_res(params, y_obs, x):
    "Takes data and computes the residual sum of squares with relation to a specified model"
    y_model = model_func(params, x)
    return np.sum((y_obs-y_model)**2)

def model_do(temps, ini_args):
    """Minimises the model with temps and assumed time values and returns the minimsed output
    day 0 is the first measured day"""
    t = np.linspace(0, len(temps)*365.25/12, len(temps))
    minimised = fmin(model_func_res, ini_args, args = (temps, t), disp=0)
    plotter(temps, t, minimised)
    return minimised

def printer(filename, res):
    "prints the results nicely"
    print(
        f"The optimised values for {filename} were:\n"
        f"Amplitude={res[0]}\n"
        f"Temperature gradient={res[1]}\n"
        f"C={res[2]}\n"
        f"Frequency={res[3]}\n"
        f"Phase={res[4]}\n\n")

def plotter(temps, time, res):
    "sanity check"
    plt.figure()
    plt.plot(time, temps)
    y_new = model_func(res, time)
    plt.plot(time, y_new)
    #plt.xlim(0, 5000)   # Zoom in
    plt.show(block = False)


def main():
    """Doobe doobe doobe
    """
    temps1 = split_data("DAex5_data1.dat")
    temps2 = split_data("DAex5_data2.dat")
    minimised1 = model_do(temps1, INI_ARGS1)
    minimised2 = model_do(temps2, INI_ARGS2)
    printer("DAex5_data1.dat", minimised1)
    printer("DAex5_data2.dat", minimised2)
    plt.show()
    
    
main()