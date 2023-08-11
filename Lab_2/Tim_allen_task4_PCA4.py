

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from Tim_allen_task1_PCA_artificial_data import Principal_Component_Analysis


def plotter(eigen_vectors, PC, latitude, longitude):
    "plots EOF1 for the temperature data onto world map and timeseries then saves plots"
    eof1 = eigen_vectors[:,-1]
    norm = Normalize(vmin=-0.1, vmax=0.1)
    cmap="seismic"                          # Setup args for coloring 

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
    
    plt.savefig('PCA4_timeseries.png')

def main():
    "loads temperatures, does PCA and plots world map and timeseries"
    data_dict = np.load("good_station_data.npz")
    temperature_time_good = data_dict.get("temperature_array")
    latitude, longitude = data_dict.get("latitude_array"), data_dict.get("longitude_array")
    [eigen_values,eigen_vectors,anomaly,covariance]=Principal_Component_Analysis(temperature_time_good)
    PC=np.matmul(anomaly,eigen_vectors)
    plotter(eigen_vectors, PC, latitude, longitude)

main()