# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:18:33 2023

@author: tal75
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

from library import Principal_Component_Analysis
from read_station_data import read_station_data_file

def load_all_data():
    "Loads all data in the dir Station_data"
    cache = []
    lat = []
    lon = []
    datetime_array = np.empty(0)
    temperature_array = np.empty((1,1))
    #for path in glob.glob("P:/My\ Documents/PHYS381/DATA_INTRODUCTION2/Station_data/**", recursive= True ):
    for path in glob.glob("Station_data/*/*")[0:40]:
        cache = read_station_data_file(path)
        lat.append(cache[0])
        lon.append(cache[1])
        datetime_array = np.array([datetime_array, cache[2]])
        temperature_array = np.array([temperature_array, cache[3]])
    return lat,lon, datetime_array,temperature_array

def main():
    "Do the thingy"
    lat,lon, datetime_array,temperature_array= load_all_data()
    [eigen_values,eigen_vectors,anomaly,covariance]=Principal_Component_Analysis(temperature_array)
    PC=np.matmul(anomaly,eigen_vectors)
    for j in [-1]: #,-2,-3]:     # plot the last three values in the PCA analysis - note ordering
        plt.figure()
        plt.subplot(2,1,1)
        #plt.pcolor(np.reshape(eigen_vectors[:,j],(13,13)))
        plt.subplot(2,1,2)
        plt.plot(PC[:,j])
        plt.savefig('PCA_Test_%04d.png' %j)  # ensures that ordering is nice
    return eigen_values,eigen_vectors,anomaly,covariance, lat,lon, datetime_array,temperature_array

eigen_values,eigen_vectors,anomaly,covariance, lat,lon, datetime_array,temperature_array = main()