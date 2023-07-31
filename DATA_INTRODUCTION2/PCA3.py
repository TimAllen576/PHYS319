
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

from library2 import Principal_Component_Analysis
from PCA2 import load_all_data

def main():
    "Does somtehing"
    lat,lon, tt_df= load_all_data()
    [eigen_values,eigen_vectors,anomaly,covariance]=Principal_Component_Analysis(tt_df.values)
    PC=np.matmul(anomaly,eigen_vectors)
    for j in [-1,-2,-3]:     # Plot the last three values in the PCA analysis - note ordering
        plt.figure()
        plt.subplot(2,1,1)
        print(np.shape(eigen_vectors[:,j]))
        plt.pcolor(np.reshape(eigen_vectors[:,j],(13,13)))  # Reshapes the eigenvectors to be plotted
        plt.subplot(2,1,2)
        plt.plot(PC[:,j])
        plt.savefig('PCA2_Test_%01d.png' %j)  # ensures that ordering is nice
