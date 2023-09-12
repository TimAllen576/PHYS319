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

HEADERLEN = 2   #   Different header types would require redoing a lot of code
#NUMMEASURES = 3201 # Comment for preferred way of giving number of values
INIWAVELENGTH = 1000
FINALWAVELENGTH = 200
WAVELENGTHSTEP = 0.25


def get_data(filename):
    """Reads data from a spectroscopy file and returns numpy arrays for
    varying temps and wavelengths.
    Input:  csv filename(string)
    Returns:   dataframe with multi-index (filename and measurement)
    """
    # Hacky, user prompts or automatic depending on performance requirements
    if 'NUMMEASURES' in globals():
        rows = NUMMEASURES  # type: ignore
    else:
        rows = (INIWAVELENGTH-FINALWAVELENGTH)/WAVELENGTHSTEP+1  # type: ignore

    raw_data = pd.read_csv(
        filename, header=[0,HEADERLEN-1], nrows=rows)

    # Assume all samples are over same wavelengths so first column is index
    wavelengths = raw_data.loc[
        :, (slice(None), "Wavelength (nm)")].iloc[:,0]

    # Does not preserve relative order of %T and Abs measurements
    # Could use a loop or be sorted if necessary
    measurements = raw_data.loc[:, (slice(None), ["%T","Abs"])]

    # Makes new dataframe with nice indexes
    new_cols = index_aligner(raw_data)
    formatted_data = pd.DataFrame(
        measurements.values, columns=new_cols, index = wavelengths.array)
    return formatted_data

def index_aligner(raw_data):
    """Specialised function for reindexing particular dataframes.
    Shifts the index so filenames align with unit then merges the
    levels of the MultiIndex.
    Input: Cary6000i csv with header as MultiIndex (Dataframe)
    Returns: New column labels (List)
    """
    top_labels = raw_data.columns.get_level_values(0)
    new_top_labels = pd.Index(['']).append(top_labels[:-1])  # Shift right 1
    shifted_cols = pd.MultiIndex.from_arrays([
        new_top_labels, raw_data.columns.get_level_values(1)])
    new_cols = []
    for old_cols_tuple in shifted_cols[1::2]:
        new_cols.append(old_cols_tuple[0]+" ("+old_cols_tuple[1]+")")
    return new_cols

def data_cleaner(lowTdata, highTdata):
    """Specialised function, removes known bad data points and
    seperates baseline data.
    Input: Specific low and high temp data (two dataframes)
    Returns: Combined data with known bad points removed (Dataframe)
    and combined baseline data (Dataframe) 
    """
    lowTbaseline = lowTdata.iloc[:,:2]
    highTbaseline = highTdata.iloc[:,:2]
    allbaselines = pd.concat([lowTbaseline, highTbaseline], axis=1)

    # Slices tests, baselines and data where the detector? broke from heat
    sliced_highTdata = highTdata.iloc[:,2:-1]
    malfunction_mask = np.zeros(sliced_highTdata.shape, dtype=bool)
    malfunction_mask[:801, -10:] = True
    clean_highTdata = sliced_highTdata.mask(malfunction_mask,other=np.nan)
    allTdata = pd.concat([lowTdata.iloc[:,2:], clean_highTdata], axis=1)
    return allTdata, allbaselines

def tauc_plotter(dataframe):
    """Plots multiple tauc plots on one set of axes.
    Input:  dataframe with multi-index (filename and measurement)
    Returns: None, shows plot
    """
    #add calculated reg line
    #formatting
    wavelengths = dataframe.index
    for sample in dataframe.columns:
        sample_data= dataframe.get(sample)
        plt.plot(wavelengths, sample_data)
    #plt.ylim(0,2)
    #plt.xlim(200,600)
    plt.grid()
    plt.savefig("Tauc-like plot")
    plt.show()

def tauc_calc(dataframe):
    """Calculates optical band gaps for spectroscopic data using a
    Tauc-like method.
    Input:  dataframe with multi-index (filename and measurement)
    Returns: band gaps (list of ints)
    """
    #smoothing-var opts, derivatives,
    # multiple options -max dev, regr fit within window

def main():
    lowTfilename = "20230906 REAL554 PHYS381 LowT.csv"
    highTfilename = "20230912 REAL554 PHYS381 HighT.csv"
    lowTdata = get_data(lowTfilename)
    highTdata = get_data(highTfilename)
    allTdata, allbaselines = data_cleaner(lowTdata, highTdata)
    tauc_plotter(allTdata)


if __name__ == "__main__":
    main()
