# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:08:21 2023

@author: tal75
"""

import matplotlib.pyplot as plt
import numpy as np

G = 9.81
OMEGA = 7.292 * 10 ** (-5)
A = -12
B = 40
ZT = 12


def temperature_function(latitude, altitude):
    """Finds the temperature for a given latitude
    
    Input: latitude (float or array of float)
    Latitude in degrees
    altitude (float or array of float)
    Altitude in km
    Returns: temperature (float or array of float)
    Temperature at given latitude
    """
    phi, z = np.meshgrid(np.deg2rad(latitude), altitude)
    b = B * (1 - np.divide(z, ZT))
    temperature = A + b * (
            (3 / 2) * np.cos(phi) ** 3 * (2 / 3 + np.sin(phi) ** 2))
    return temperature


def temperature_heatmap(data):
    """
    Create and plots a heatmap from a numpy array and certain specifics.

    Input: data (2D numpy array)
    Temperature data.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="seismic", aspect="auto",
                   extent=[-90, 90, 0, 22])
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel("T (deg C)", rotation=90, va="top")
    ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Altitude (km)")
    ax.set_xticks(np.linspace(-75, 75, 7))
    ax.set_yticks(np.linspace(0, 20, 5))
    plt.savefig("temperature_heatmap.png")
    plt.show(block=False)


def zonal_wind_field_calc(temperature, latitude, altitude):
    """Calculates the zonal wind field from temperature, latitude and
    altitude data.

    Input: temperature (float or array of float)
    Temperature in degrees Celsius.
    latitude (float or array of float)
    Latitude in degrees.
    altitude (float or array of float)
    Altitude in km.
    Returns: zonal_wind_field (float or array of float)
    Zonal wind field in m/s.
    """
    # Ensuring cancelling units of km and kelvin
    temperature += 273.15
    meridional_gradient_t = np.gradient(temperature, 111.21, axis=1)
    f = 2 * OMEGA * np.sin((np.deg2rad(latitude)))
    zonal_wind_z_derivative = - G * meridional_gradient_t / (f * temperature)
    # Rearranged Euler's method
    h = np.abs(np.diff(altitude)[0])
    # Find the cumulative sums of the zonal_wind_z_derivative starting
    # at the bottom of the array
    right_way_up_z_derivative = np.flip(zonal_wind_z_derivative, axis=0)
    zonal_wind_field_upside_down = h * np.cumsum(right_way_up_z_derivative,
                                                 axis=0)
    # Flip the array back to the correct orientation
    zonal_wind_field = np.flip(zonal_wind_field_upside_down, axis=0)
    return zonal_wind_field


# noinspection PyTypeChecker
def zonal_wind_heatmap(data, cut=False):
    """
    Create and plots a heatmap from a numpy array and certain specifics
    then cuts or blacks out equatorial data.

    Input: data (2D numpy array)
    Wind data for the North to the South Pole and 0-22 km altitudes.
    cut (boolean)
    Whether the central data should be cut or blacked out.
    Returns: None
    """
    fig, ax = plt.subplots()
    if cut:
        min_lat_index = int(7 * len(data) / 18)
        max_lat_index = int(11 * len(data) / 18)
        data = np.delete(data, np.s_[min_lat_index:max_lat_index], 1)
    im = ax.imshow(data, cmap="plasma", norm="linear", aspect="auto",
                   extent=[-90, 90, 0, 22])
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel("u (m/s)", rotation=90, va="top")
    ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Altitude (km)")
    if not cut:
        ax.fill_between([-20, 20], 0, 22, facecolor="black", alpha=1)
        plt.savefig("zonal_wind_heatmap.png")
        plt.show(block=False)
    else:
        ax.set_xticks(np.linspace(-90, 90, 15),
                      labels=[-90, -80, -70, -60, -50, -40, -30,
                              r"$\pm$20", 30, 40, 50, 60, 70, 80, 90])
        plt.savefig("zonal_wind_heatmap_cut.png")
        plt.show()


def main():
    """Does the thingamabob"""
    resolution = 1000
    altitude = np.linspace(22, 0, resolution)
    latitude = np.linspace(-90, 90, resolution)
    temperatures = temperature_function(latitude, altitude)
    temperature_heatmap(temperatures)
    zonal_wind_field = zonal_wind_field_calc(temperatures,
                                             latitude, altitude)
    zonal_wind_heatmap(zonal_wind_field)
    zonal_wind_heatmap(zonal_wind_field, cut=True)


if __name__ == '__main__':
    main()
