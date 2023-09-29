# -*- coding: utf-8 -*-
"""
This code reads in the various Shortwave and Long-wave components
available in the CERES database and plots a diagram of the absorbed
shortwave, the outgoing long-wave and the net flux.

This code also contains a function which integrates the net energy
imbalance to derive the meridional heat transport in each hemisphere.

Created on Sat Jun 13 15:12:10 2020
Updated on Tues 26th September 2023

@author: ajm226
"""

import numpy as np
import xarray as xr
from scipy import integrate
import matplotlib.pyplot as plt


EARTH_RADIUS = 6.371E+06
filepath = "CERES_EBAF-TOA_Ed4.2_Subset_200003-202306.nc"


def inferred_heat_transport(energy_in, lat_deg):
    """
    Returns the inferred heat transport (in PW) by integrating the
    net energy imbalance from pole to pole.
    Modified from climlab project by Brian Rose.
    """
    lat_rad = np.deg2rad(lat_deg)
    heat_transport = (
            1E-15 * 2 * np.pi * EARTH_RADIUS ** 2 *
            integrate.cumtrapz(np.cos(lat_rad) * energy_in,
                               x=lat_rad, initial=0.))
    return heat_transport


def data_slicer(time_slice):
    """
    This function reads in the CERES data and slices it to the required times
    """
    nc_id_1 = xr.open_dataset(filepath)
    # Examines values for a specified time slice
    nc_id_2 = nc_id_1.sel(time=time_slice)
    # Calculate the mean over all times and
    # longitudes to obtain a mean at every latitude
    ceres = nc_id_2.mean('time').mean('lon')
    return ceres


def zonal_mean_plotter(ceres, filename):
    """
    This function plots the energy balance diagram
    """
    ceres_variable_name1 = 'solar_mon'
    ceres_variable_name2 = 'toa_sw_all_mon'
    ceres_variable_name3 = 'toa_lw_all_mon'
    ceres_variable_name4 = 'toa_net_all_mon'
    # Plot a figure
    plt.figure(figsize=(210.0 / 25.4, 120.0 / 25.4))
    plt.plot(ceres['lat'],
             ceres[ceres_variable_name1] - ceres[ceres_variable_name2],
             color='red', linewidth=2.0, label='Absorbed Shortwave')
    plt.plot(ceres['lat'], ceres[ceres_variable_name3], color='blue',
             linewidth=2.0, label='Outgoing Long-wave')
    plt.plot(ceres['lat'], ceres[ceres_variable_name4], color='black',
             linewidth=2.0, label='Net Flux')
    plt.legend(loc='upper left')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Energy Flux (W m$^{-2}$)')
    plt.xlim([-90.0, 90.0])
    plt.ylim([-150.0, 450.0])
    plt.xticks([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
    plt.grid(True)
    plt.savefig(filename, dpi=600)


def meridional_heat_transport_plotter(ceres, meridional_heat, filename):
    """
    This function plots the meridional heat transport
    """
    plt.figure(figsize=(120.0 / 25.4, 210.0 / 25.4))
    plt.plot(meridional_heat, ceres['lat'])
    plt.ylabel('Latitude (degrees)')
    plt.xlabel('Meridional Heat Transport (PW)')
    plt.ylim([-90.0, 90.0])
    plt.grid(True)
    plt.savefig(filename, dpi=600)


def main():
    """
    Main function
    """
    ceres = data_slicer(slice(None, None))
    ceres_original = data_slicer(slice("2000-12-31", "2001-12-31"))
    ceres_aus_summer = data_slicer(slice("2000-12-01", "2001-02-28"))
    ceres_aus_winter = data_slicer(slice("2001-06-01", "2001-08-31"))
    zonal_mean_plotter(ceres_original, "Energy_Balance1_figure1.png")
    zonal_mean_plotter(ceres_aus_summer, "Energy_Balance1_aus_Summer.png")
    zonal_mean_plotter(ceres_aus_winter, "Energy_Balance1_aus_Winter.png")
    meridional_heat_all = inferred_heat_transport(
        ceres_original['toa_net_all_mon'], ceres['lat'])
    meridional_heat_summer = inferred_heat_transport(
        ceres_aus_summer['toa_net_all_mon'], ceres['lat'])
    meridional_heat_winter = inferred_heat_transport(
        ceres_aus_winter['toa_net_all_mon'], ceres['lat'])
    meridional_heat_transport_plotter(
        ceres, meridional_heat_all, "Meridional_heat_transport.png")
    meridional_heat_transport_plotter(
        ceres_aus_summer, meridional_heat_summer,
        "Heat_transport_summer.png")
    meridional_heat_transport_plotter(
        ceres_aus_winter, meridional_heat_winter,
        "Heat_transport_winter.png")
    plt.show()


if __name__ == '__main__':
    main()
