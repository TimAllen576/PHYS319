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

import extractor


LOWTFILE = "20230906 REAL554 PHYS381 LowT.csv"
HIGHTFILE = "20230912 REAL554 PHYS381 HighT.csv"
HEADER_LEN = 2   # Different header types could require redoing a lot of code
#NUM_MEASURES = 3201 # Comment for preferred way of giving number of values
INI_WAVELENGTH = 1000
FINAL_WAVELENGTH = 200
WAVELENGTH_STEP = 0.25
GLASS_INTERFERENCE_ENERGY = 5
R=1/2   # r-value denoting the direct allowed transition
FILTER_WINDOW = 35
FILTER_ORDER = 1


def get_data(filename):
    """
    Reads data from a spectroscopy file and returns numpy arrays for
    varying temps and wavelengths.

    Input: filename (string)
    CSV filename.
    Returns: formatted_data (DataFrame)
    Dataframe with columns of wavelengths, baselines and abs data.
    """
    # Hacky, user prompts or automatic depending on performance requirements
    if 'NUM_MEASURES' in globals():
        rows = NUM_MEASURES  # type: ignore
    else:
        rows = (INI_WAVELENGTH-FINAL_WAVELENGTH)/WAVELENGTH_STEP+1  # type: ignore

    raw_data = pd.read_csv(
        filename, header=[0,HEADER_LEN-1], nrows=rows)

    # Assume all samples are over same wavelengths so first column is taken
    wavelengths = raw_data.loc[:, (slice(None),
                                   "Wavelength (nm)")].iloc[:,0]
    wavelengthsdf = pd.DataFrame(wavelengths.values,
                                columns=["Wavelength (nm)"])

    # Creates new DataFrame, does not preserve relative order of
    # %T and Abs measurements, could use a loop or be sorted if necessary
    measurements = raw_data.loc[:, (slice(None), ["%T","Abs"])]
    new_cols = index_aligner(raw_data)
    measurementsdf = pd.DataFrame(measurements.values, columns=new_cols)
    formatted_data = wavelengthsdf.join(measurementsdf)
    return formatted_data

def index_aligner(raw_data):
    """
    Specialised helper function for reindexing particular dataframes.
    Shifts the index so filenames align with unit then merges the
    levels of the MultiIndex.

    Input: raw_data (Dataframe)
    Cary6000i csv data with header as MultiIndex.
    Returns: new_cols (List)
    Formatted list of proper column names for data.
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
    """
    Specialised function which removes known bad data points and
    seperates baseline data.

    Input: lowTdata (dataframe), highTdata (dataframe)
    Specific low and high temp data with wavelengths, baselines and abs data.
    Returns: allTdata (Dataframe)
    Combined abs data with data after detector malfunction removed.
    allbaselines (Dataframe)
    Combined baseline data.
    """
    lowTbaseline = lowTdata.iloc[:,1:3]
    highTbaseline = highTdata.iloc[:,1:3]
    allbaselines = pd.concat([lowTbaseline, highTbaseline], axis=1)

    # Slices tests, baselines and data where the detector broke from heat?
    sliced_highTdata = highTdata.iloc[:,3:-1]

    malfunction_mask = np.zeros(sliced_highTdata.shape, dtype=bool)
    malfunction_mask[:801, -10:] = True
    clean_highTdata=sliced_highTdata.mask(malfunction_mask,other=np.nan)
    # Janky little line to stop errors from duplicate columns labels
    clean_highTdata.rename(columns={"T325 (Abs)":"T325NoVac (Abs)"}, inplace=True)
    allTdata = pd.concat(
        [lowTdata["Wavelength (nm)"],lowTdata.iloc[:,3:],
        clean_highTdata], axis=1)
    return allTdata, allbaselines

def scale_n_differentiate(datadf):                                  # refactor?
    """
    Calculates optical band gaps for spectroscopic data using a
    Tauc-like method.

    Input : datadf (DataFrame)
    Clean data to be used for analysis with first column as wavelengths
    and other columns as sample data.
    Returns : sorted_scaled_df (DataFrame)
    Data with columns of wavelengths scaled to photon energy,
    absorbance scaled to ahv^2,
    derivative after data smoothing.
    """
    # Create new dataframe with photon energy
    energy_df = pd.DataFrame({"eV":1240 / datadf["Wavelength (nm)"]})
    scaled_df = pd.concat({"Energy": energy_df}, axis=1)
    # Find index to cut off data where glass interferes
    glass_int_index = (
        scaled_df["Energy","eV"] >
        GLASS_INTERFERENCE_ENERGY).idxmax()
    scaled_df = scaled_df[:glass_int_index]
    for temperature_Abs in datadf.columns[1:]:
        temperature_ahv = temperature_Abs.replace("Abs", "ahv")
        # Make absorbance scaling calculations
        scaled_df["ahv_data",temperature_ahv]=((
            datadf[temperature_Abs] * scaled_df["Energy", "eV"]) ** (1/R))[
            :glass_int_index]
        # Smooth data and find its derivative using preferred filter #Pull out?
        scaled_df["data_deriv",temperature_ahv] = smooth_n_deriv(
            scaled_df["ahv_data", temperature_ahv],
            scaled_df["Energy","eV"],
            FILTER_WINDOW, FILTER_ORDER)
    sorted_scaled_df = scaled_df.sort_index(axis=1, level=0)
    #smoothing-various opts,
    return sorted_scaled_df

def smooth_n_deriv(rough_data, energy,window, porder):
    """
    Helper function to smooth data using a Savitzky-Golay filter.

    Input: rough_data (Series)
    Spec data scaled to ahv.
    Returns: derived_data (Series)
    Deritive of smoothed data.
    """
    smooth_data = signal.savgol_filter(rough_data,window, porder)
    derived_data = np.gradient(smooth_data,energy)
    return derived_data

def linear_region_extrapolater(scaled_df):
    """
    Calculates the band gap using Tauc-like plots from a line built
    from the max derivative. order should be around the width of the
    derivitive 'bump'.

    Input : scaled_df (DataFrame)
    Dataframe with columns of energy, ahv data and data derivitives
    for each temperature.
    Returns : linear_region_df (DataFrame)
    Dataframe of gradient and y-intercep of lines extrapolated from the
    linear region of the data, energy value at x-intercept is
    the optical band gap.
    """
    # Approximate width of derivative peak around linear region
    extrema_width = 50
    linear_region_df = pd.DataFrame()
    for temperature in scaled_df["data_deriv"].columns[25:26]:  # temp slice
        derivative_array = np.array(scaled_df["data_deriv" ,temperature])
        #print(temperature) t525 bad?

        # Correct index is second to last, final is at far right and 
        #since derivatives are so low at low energies
        # many false peaks appear
        max_index = signal.argrelextrema(derivative_array,
                                         np.greater,
                                         order=extrema_width)[0][1:-1] # testing slice
        
        # Testing
        plt.plot(scaled_df["Energy","eV"], derivative_array)
        idx=max_index
        plt.vlines(scaled_df["Energy","eV"][idx],
                    0, 60)
        plt.vlines(scaled_df["Energy","eV"][idx+extrema_width],
                    0, 60, linestyles='dashed')
        plt.vlines(scaled_df["Energy","eV"][idx-extrema_width],
                    0, 60, linestyles='dashed')
        plt.show()

        max_deriv = derivative_array[max_index]
        max_deriv_data = scaled_df["ahv_data", temperature][max_index]
        max_deriv_energy = scaled_df["Energy","eV"][max_index]
        linear_region_yint = max_deriv_data - max_deriv * max_deriv_energy
        #Important to match multi-index
        linear_region_df["ahv_data",temperature] = [max_deriv, linear_region_yint]
        
    # add option for regr fit within window
    return linear_region_df

class InteractivePlot:
    """Interactively shows different smoothing options to show Daniel."""

    def __init__(self, scaled_df):
        # Initialize data and plot
        self.window, self.p_order= FILTER_WINDOW, FILTER_ORDER
        self.demo_data = scaled_df.iloc[:, 1:]
        self.temperature = self.demo_data.columns[0][1]
        self.energy = scaled_df["Energy","eV"]
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        self.ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
        # Window size for Savgol filter should be odd
        self.win_size = Slider(
            self.ax_slide, 'Window size',
            valmin=3, valmax=99, valinit=FILTER_WINDOW, valstep=2) 
        self.ax.set_xlabel("Photon energy (eV)")
        self.ax.set_ylabel("Gradient")
        self.ax.grid(True)

        # Initial calculations
        self.demo_data["data_deriv_raw",
                       self.temperature] = np.gradient(
            self.demo_data["ahv_data", self.temperature],
            self.energy)
        # self.raw =self.ax.plot(self.energy,
        #         self.demo_data["data_deriv_raw",self.temperature],
        #         "b.",markersize=1, alpha = 0.5)
        self.demo_data["data_deriv",self.temperature] = smooth_n_deriv(
            self.demo_data["ahv_data", self.temperature],
            self.energy, FILTER_WINDOW, FILTER_ORDER)
        self.current,=self.ax.plot(self.energy,
                self.demo_data["data_deriv", self.temperature],
                "r-",linewidth=0.5)

        self.win_size.on_changed(self.update_win)

    def update_win(self,window):
        """Updates the plot with data smoothed with new window size."""
        self.demo_data["data_deriv",self.temperature] = smooth_n_deriv(
            self.demo_data["ahv_data", self.temperature],
            self.energy,
            window, self.p_order)
        self.current.set_ydata(self.demo_data["data_deriv",
                                              self.temperature])
        self.fig.canvas.draw()

    def show(self):
        plt.show()

def tauc_plotter(scaled_df):#, linear_region_df):
    """Plots multiple tauc plots on one set of axes.
    Input:  dataframe with multi-index (filename and measurement)
    Returns: None
    Shows plot.
    """
    # Linear region disappears?
    for temperature in scaled_df["ahv_data"].columns[1:5]:            # temp slice
        temperature_label = temperature.rstrip("(ahv)")
        plt.plot(scaled_df[("Energy","eV")], scaled_df["ahv_data",temperature],
                 label=temperature_label)
        
        # max_deriv, linear_region_yint = linear_region_df[temperature]
        # band_gap = -linear_region_yint/max_deriv
        # linear_region_line = max_deriv * scaled_df[("Energy","eV")] + linear_region_yint
        # plt.plot(scaled_df[("Energy","eV")], linear_region_line, "--",
        #          label=f"{temperature_label} band-gap: {band_gap:.3f}")

    plt.xlabel("Photon energy (eV)")
    plt.xlim(0, 5.5)
    plt.ylabel(r"$(\alpha h v)^{1/r}$")
    plt.ylim(0, 80)
    plt.grid(True)
    plt.legend()
    plt.savefig("Tauc-like plot")
    plt.show()

def other_tests(data):
    """
    Creates CSVs with single columns of data to run with other tools
    """
    wavelength = data.iloc[:, 0]
    values = data.iloc[:, 1:]
    new_values = values + 1 + np.sqrt(4*values**2+8*values)
    new_data = pd.concat([wavelength, new_values], axis=1)
    #print(new_data)
    extractor.autoextract(new_data, 'example-output',
                          'Ex_other_tools',intensity_scale=100,
                          verbose=True)
    

def main():
    """Main function to take UV-Vis spec data and create Tauc-like plots
    determining bandgap temperature dependance."""
    lowTdata = get_data(LOWTFILE)
    highTdata = get_data(HIGHTFILE)
    usingdata, allbaselines = data_cleaner(lowTdata, highTdata)
    scaled_df = scale_n_differentiate(usingdata)

    other_tests(usingdata[1245:3000])

    # demo = InteractivePlot(scaled_df.iloc[:, [0, 25]])
    # demo.show()

    # linear_region_df = linear_region_extrapolater(scaled_df)
    # tauc_plotter(scaled_df)#, linear_region_df)

if __name__ == "__main__":
    main()
