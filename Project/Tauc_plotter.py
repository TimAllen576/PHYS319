#!/usr/bin/env python

"""
Reads spectroscopy data for varying temps and wavelengths and uses a
Tauc-like plot method to determine energy of optical band-gap changes
due to temperature.

Created Sept 2023
@author: Timothy Allen 66522411
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.widgets import Slider

HEADER_LEN = 2   # Different header types would require redoing a lot of code
#NUM_MEASURES = 3201 # Comment for preferred way of giving number of values
INI_WAVELENGTH = 1000
FINAL_WAVELENGTH = 200
WAVELENGTH_STEP = 0.25
GLASS_INTERFERENCE_ENERGY = 5

r=1/2


def get_data(filename):
    """Reads data from a spectroscopy file and returns numpy arrays for
    varying temps and wavelengths.
    Input:  csv filename(string)
    Returns:   dataframe with multi-index (filename and measurement)
    """
    # Hacky, user prompts or automatic depending on performance requirements
    if 'NUM_MEASURES' in globals():
        rows = NUM_MEASURES  # type: ignore
    else:
        rows = (INI_WAVELENGTH-FINAL_WAVELENGTH)/WAVELENGTH_STEP+1  # type: ignore

    raw_data = pd.read_csv(
        filename, header=[0,HEADER_LEN-1], nrows=rows)

    # Assume all samples are over same wavelengths so first column is taken
    wavelengths = raw_data.loc[
        :, (slice(None), "Wavelength (nm)")].iloc[:,0]
    wavelengthsdf = pd.DataFrame(
        wavelengths.values, columns=["Wavelength (nm)"])

    # Does not preserve relative order of %T and Abs measurements
    # Could use a loop or be sorted if necessary
    measurements = raw_data.loc[:, (slice(None), ["%T","Abs"])]

    # Makes new dataframe with nice indexes
    new_cols = index_aligner(raw_data)
    measurementsdf = pd.DataFrame(measurements.values, columns=new_cols)
    formatted_data = wavelengthsdf.join(measurementsdf)
    return formatted_data

def index_aligner(raw_data):
    """Specialised function for reindexing particular dataframes.
    Shifts the index so filenames align with unit then merges the
    levels of the MultiIndex.
    Input: Cary6000i csv with header as MultiIndex (Dataframe)
    Returns: New column labels (List)
    """
    top_labels = raw_data.columns.get_level_values(0)
    # Shift right 1
    new_top_labels = pd.Index(['']).append(top_labels[:-1])
    shifted_cols = pd.MultiIndex.from_arrays([
        new_top_labels, raw_data.columns.get_level_values(1)])
    new_cols = []
    for old_cols_tuple in shifted_cols[1::2]:
        new_cols.append(old_cols_tuple[0] + " ("+ old_cols_tuple[1]+")")
    return new_cols

def data_cleaner(lowTdata, highTdata):
    """Specialised function, removes known bad data points and
    seperates baseline data.
    Input: Specific low and high temp data (two dataframes)
    Returns: Combined data with known bad points removed (Dataframe)
    and combined baseline data (Dataframe) 
    """
    lowTbaseline = lowTdata.iloc[:,1:3]
    highTbaseline = highTdata.iloc[:,1:3]
    allbaselines = pd.concat([lowTbaseline, highTbaseline], axis=1)

    # Slices tests, baselines and data where the detector? broke from heat
    sliced_highTdata = highTdata.iloc[:,3:-1]

    malfunction_mask = np.zeros(sliced_highTdata.shape, dtype=bool)
    malfunction_mask[:801, -10:] = True
    clean_highTdata=sliced_highTdata.mask(malfunction_mask,other=np.nan)
    # Janky little line to stop errors from duplicate columns labels
    clean_highTdata.rename(columns={"T325 (Abs)":"T325Hot (Abs)"}, inplace=True)
    allTdata = pd.concat(
        [lowTdata["Wavelength (nm)"],lowTdata.iloc[:,3:],
        clean_highTdata], axis=1)
    return allTdata, allbaselines

def tauc_plotter(scaled_data):
    """Plots multiple tauc plots on one set of axes.
    Input:  dataframe with multi-index (filename and measurement)
    Returns: None, shows plot
    """
    #formatting, fix calc first
    for sample in scaled_data.columns[1:2:]:            # temp slice
        plt.plot("eV", sample, data=scaled_data)
    plt.xlabel("Photon energy (eV)")
    plt.ylabel(r"$(\alpha h v)^{1/r}$")
    plt.grid()
    plt.savefig("Tauc-like plot")
    #plt.show()

def scale_n_differentiate(datadf):                                        # refactor
    """Calculates optical band gaps for spectroscopic data using a
    Tauc-like method.
    Input:  dataframe with index as wavelengths and columns as data
    Returns: scaled dataframe with photon energy, ahv^2 data, smoothed
    data and the derivative of the data 
    """
    # Set params for smoothing filter
    window, p_order = 30, 1
    # Create new dataframe with photon energy
    energy_df = pd.DataFrame({"eV":1240 / datadf["Wavelength (nm)"]})
    scaled_df = pd.concat({"Energy": energy_df}, axis=1)
    # Find index to cut off data where glass interferes
    glass_int_index = (
        scaled_df["Energy","eV"] >
        GLASS_INTERFERENCE_ENERGY).idxmax()
    scaled_df = scaled_df[:glass_int_index]
    for sample_name in datadf.columns[1:]:
        sample = sample_name.replace("Abs", "ahv")
        # Make absorbance scaling calculations
        scaled_df["ahv_data",sample]=((
            datadf[sample_name] * scaled_df["Energy", "eV"]) ** (1/r))[
            :glass_int_index]
        # Smooth data using preferred filter
        scaled_df["ahv_data_smooth",sample] = smoother(
            scaled_df["ahv_data",sample],window, p_order)
        # Differentiate curve using numpy gradient
        scaled_df["ahv_derivative", sample] = np.gradient(
            scaled_df["ahv_data_smooth",sample],
            scaled_df["Energy","eV"])
    sorted_scaled_df = scaled_df.sort_index(axis=1, level=0)
    #smoothing-var opts, derivative rel max
    # multiple options -max dev, regr fit within window, band calcs
    return sorted_scaled_df

def smoother(rough_data,window, porder):
    """Smooths data using a filter ? sued for testing several.
    Input:  Pandas Series of rough data
    Returns: Pandas Series of smooth data
    """
    smooth_data = signal.savgol_filter(rough_data,window, porder)
    #smooth_data = signal
    return smooth_data

def tauc_demo(scaled_df):
    """Interactively shows different smoothing options
    Input:  dataframe
    Returns: band gaps (list of ints)
    """
    window, porder = 30, 1

    for sample_name in data.columns[1:2]:                   # temp slice
        scaled_df[sample] = smoother(data[sample],window, porder)

        # Differentiate curve using with numpy gradient
        deriv = np.gradient(data[sample], data["eV"])

        fig=plt.figure()
        ax=fig.subplots()
        plt.subplots_adjust(bottom=0.25)
        p,=ax.plot(data["eV"], deriv, "r-",linewidth=0.5)
        # Defining the Slider position
        ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
        # Properties of the slider
        win_size = Slider(
            ax_slide, 'Window size', valmin=1, valmax=99, valinit=30, valstep=1)

    fig=plt.figure()
    ax=fig.subplots()
    ax.plot(scaled_df["Energy", "eV"],
            scaled_df["ahv_derivative", sample],
            "r-",linewidth=0.5)
    plt.xlabel("Photon energy (eV)")
    plt.ylabel("Gradient")
    plt.grid(True)
    plt.savefig(f"Derivative plot(presmoothed, {window, p_order})")
    
    win_size.on_changed(update)
    plt.show()

    return None

# Updating the plot
def update(val, window, ahv):
    current_v = int(window.val)
    new_y = signal.savgol_filter(y, current_v, 3)
    p.set_ydata(new_y)
    fig.canvas.draw() #redraw the figure

def main():
    lowTfilename = "20230906 REAL554 PHYS381 LowT.csv"
    highTfilename = "20230912 REAL554 PHYS381 HighT.csv"
    lowTdata = get_data(lowTfilename)
    highTdata = get_data(highTfilename)
    usingdata, allbaselines = data_cleaner(lowTdata, highTdata)
    scaled_data = scale_n_differentiate(usingdata)
    print(scaled_data)
    #tauc_demo(scaled_data)
    #tauc_plotter(scaled_data)

if __name__ == "__main__":
    main()
