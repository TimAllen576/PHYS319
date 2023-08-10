import numpy as np
import matplotlib.pyplot as plt
from read_station_data import datetime_array_create
from PCA_artificial_data import Principal_Component_Analysis

# def reconstructor():
#     "Reconstructs a time series from 3 EOFs"

def plotter(data1, data2, acc):
    "Plots two time series against each other to compare"
    time = datetime_array_create(1990, 2006)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(time, data1)
    ax1.set_title("Original timeseries")
    ax1.set_xlabel("years")
    ax1.set_ylabel("Temperature(C)")

    ax2 = fig.add_subplot(212)
    ax2.plot(time, data2)
    ax2.set_title("Reconstructed timeseries")
    ax2.set_xlabel("years")
    ax2.set_ylabel("Temperature(C)")

    fig.tight_layout()
    plt.savefig('PCA5_test%01d.png'%acc)


def main():
    """Plots timeseries from first valid station first three EOFs
    compares with input
    """
    acc= 400
    temperature_array = np.load("good_station_data.npz").get("temperature_array")
    first_weatherstation_data = temperature_array[:, 0]
    [eigen_values,eigen_vectors,anomaly,covariance]=Principal_Component_Analysis(temperature_array)
    first3EOF = eigen_vectors[:, -acc:]
    proj = np.matmul(first3EOF, np.transpose(first3EOF))
    first_weatherstation_pca = np.matmul(anomaly, proj)     # Reconstructing the data
    plotter(first_weatherstation_data, first_weatherstation_pca[:, 0], acc)
    
main()