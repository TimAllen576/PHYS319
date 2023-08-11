

import numpy as np
import matplotlib.pyplot as plt
from read_station_data import datetime_array_create
from Tim_allen_task1_PCA_artificial_data import Principal_Component_Analysis


def plotter(data1, data2, acc):
    "Plots two time series against each other to compare"
    time = datetime_array_create(1990, 2006)
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(time, data1, label="Original timeseries")
    ax1.plot(time, data2, label="Reconstructed timeseries")
    ax1.set_xlabel("years")
    ax1.set_ylabel("Temperature(C)")
    plt.legend()
    plt.savefig('PCA5_test%01d.png'%acc)

def main():
    """Plots timeseries from first valid station first three EOFs
    compares with input
    """
    acc = 3 # Number of EOFs to use
    temperature_array = np.load("good_station_data.npz").get("temperature_array")
    first_weatherstation_data = temperature_array[:, 0]
    [eigen_values,eigen_vectors,anomaly,covariance]=Principal_Component_Analysis(temperature_array)
    first3EOF = eigen_vectors[:, -acc:]
    proj = np.matmul(first3EOF, np.transpose(first3EOF))
    weatherstation_pca = np.matmul(anomaly, proj)     # Reconstructing the data
    plotter(first_weatherstation_data, weatherstation_pca[:, 0], acc)
    
main()