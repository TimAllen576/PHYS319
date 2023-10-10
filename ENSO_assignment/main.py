# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:48:08 2021

@author: ajm226
edited: tal75 10th Oct 2023
"""

import numpy as np
import ddeint as dde
import matplotlib.pyplot as plt


ALPHA = 0.75
DELTA = 8
TF = 25.0
DT = 0.001

INITIAL_VALUE = 0.55
R = 0


def dao_model(t_func, t, alpha, delta):
    """???"""
    first_order = t_func(t)
    print(t)
    second_order = - alpha * t_func(t - delta)
    third_order = - (t_func(t) * t_func(t) * t_func(t))
    return first_order + second_order + third_order, 0


def dao(alpha, delta, tf, dt):
    """
    Description:  This function numerically solves the delayed
    difference equation for the delayed action oscillator model for El Niño

    Input:
    alpha        Constant in equation
    delta        Time delay
    tf           The solver will produce a solution from t=0 to t=tf.
    dt  time increment in solver
    Output:
    t         The vector of time values at which T is calculated.
    T         A vector of temperature anomaly values.
    """
    time_range = np.arange(0, tf, dt)
    temperature_anomaly = dde.ddeint(dao_model, initial_value, time_range,
                                     fargs=(alpha, delta))
    return time_range, temperature_anomaly


def initial_value(*args):
    """pain"""
    return np.array([INITIAL_VALUE, 0])


def main():
    """
    This function plots the temperature anomaly over time
    for the delayed action oscillator model
    """
    time_output, temperature_output = dao(ALPHA, DELTA, TF, DT)
    plt.plot(time_output, temperature_output[:, 0])
    plt.show()

if __name__ == '__main__':
    main()
