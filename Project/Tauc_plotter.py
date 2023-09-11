#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reads spectroscopy data for varying temps and wavelengths and uses a Tauc-like plot method to determine energy of optical band-gap changes due to temperature

Created Sept 2023
@author: Timothy Allen 66522411
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename, rows, label):
    """Reads data from a spectroscopy file and returns numpy arrays for varying temps and wavelengths
    Input:  csv filename(string), Number of rows of measures(int), Label of missing data column(string)
    Returns:   dataframe with multi-index (filename and measurement)
    """
    data = pd.read_csv(filename, header=[0,1], nrows=rows)
    data.drop(label, axis=1, level=0, inplace=True)
    return data

def tauc_plotter(dataframe):
    """Plots multiple tauc plots on one set of axes
    Input:  dataframe with multi-index (filename and measurement)
    Returns: None, prints plot
    """
    for file in dataframe.columns[5::2]:
        x= dataframe.get("T015")
        y= dataframe.get(file)
        plt.plot(x, y)
    plt.grid()
    plt.show()

def tauc_calc(dataframe):
    """Calculates optical band gaps for spectroscopic data using a Tauc-like method
    Input:  dataframe with multi-index (filename and measurement)
    Returns: band gaps (list of ints)
    """

def main():
    lowTfilename = "20230906 REAL554 PHYS381 LowT.csv"
    wavelen_points = 3201
    bad_label = "Unnamed: 32_level_0"
    lowTdata = read_data(lowTfilename, wavelen_points, bad_label)
    print(lowTdata)
    tauc_plotter(lowTdata)


if __name__ == "__main__":
    main()