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
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Slider

LOW_T_FILE = "20230906 REAL554 PHYS381 LowT.csv"
HIGH_T_FILE = "20230912 REAL554 PHYS381 HighT.csv"
INI_WAVELENGTH = 1000
FINAL_WAVELENGTH = 200
WAVELENGTH_STEP = 0.25
GLASS_INTERFERENCE_ENERGY = 5.2
SAMPLE_THICKNESS = 44 * 10 ** - 7  # Units of cm
R = 1 / 2  # r-value denoting the direct allowed transition
FILTER_WINDOW = 31
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
    rows = int((INI_WAVELENGTH - FINAL_WAVELENGTH) / WAVELENGTH_STEP + 1)
    raw_data = pd.read_csv(filename, header=[0, 1], nrows=rows)
    # Assume all samples are over same wavelengths
    wavelengths_column = raw_data.loc[:, (slice(None),
                                          "Wavelength (nm)")].iloc[:, 0]
    wavelengths = pd.DataFrame(wavelengths_column.values,
                               columns=["Wavelength (nm)"])
    # Creates new DataFrame, does not preserve relative order of
    # %T and Abs measurements, could use a loop / be sorted if necessary
    measurements_column = raw_data.loc[:, (slice(None), ["%T", "Abs"])]
    aligned_columns = index_aligner(raw_data)
    measurements = pd.DataFrame(measurements_column.values,
                                columns=aligned_columns)
    formatted_data = wavelengths.join(measurements)
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
        new_cols.append(old_cols_tuple[0] + " (" + old_cols_tuple[1] + ")")
    return new_cols


def data_cleaner(low_t_data, high_t_data):
    """
    Specialised function which removes known bad data points and
    separates baseline data.

    Input: lowTdata (dataframe), highTdata (dataframe)
    Specific low and high temp data with wavelengths, baselines and Abs.
    Returns: all_t_data (Dataframe)
    Combined abs data with data after detector malfunction removed.
    all_baselines (Dataframe)
    Combined baseline data.
    """
    low_t_baseline = low_t_data.iloc[:, 1:3]
    high_t_baseline = high_t_data.iloc[:, 1:3]
    all_baselines = pd.concat([low_t_baseline, high_t_baseline], axis=1)
    # Slice test and mask data where the detector broke from heat?
    sliced_high_t_data = high_t_data.iloc[:, 3:-1]
    malfunction_mask = np.zeros(sliced_high_t_data.shape, dtype=bool)
    malfunction_mask[:801, -10:] = True
    masked_high_t_data = sliced_high_t_data.mask(malfunction_mask,
                                                 other=np.nan)
    # Janky little line to stop errors from duplicate columns labels
    masked_high_t_data.rename(columns={"T325 (Abs)": "T325NoVac (Abs)"},
                              inplace=True)
    all_t_data = pd.concat([low_t_data["Wavelength (nm)"],
                            low_t_data.iloc[:, 3:],
                            masked_high_t_data], axis=1)
    # Cut off data where glass interferes
    glass_interference_wavelength = 1240 / GLASS_INTERFERENCE_ENERGY
    glass_int_index = (
            all_t_data["Wavelength (nm)"] <
            glass_interference_wavelength).idxmax()
    clean_data = all_t_data[:glass_int_index]
    return clean_data, all_baselines


def scale_n_differentiate(data):
    """
    Calculates optical band gaps for spectroscopic data using a
    Tauc-like method.

    Input : data (DataFrame)
    Clean data to be used for analysis with first column as wavelengths
    and other columns as sample data.
    Returns : sorted_scaled_df (DataFrame)
    Data with columns of wavelengths scaled to photon energy,
    absorbance scaled to ahv^2,
    derivative after data smoothing.
    """
    energy_df = pd.DataFrame({"eV": 1240 / data["Wavelength (nm)"]})
    scaled_df = pd.concat({"Energy": energy_df}, axis=1)
    abs_factor = 1 / (np.log(np.e) * SAMPLE_THICKNESS)
    for sample_abs in data.columns[1:]:
        sample_ahv = sample_abs.replace("Abs", "ahv")
        scaled_df["ahv_data", sample_ahv] = (
                (abs_factor * data[sample_abs] *
                 scaled_df["Energy", "eV"]) ** (1 / R))
        scaled_df["data_derivative", sample_ahv] = smooth_n_derivative(
            scaled_df["ahv_data", sample_ahv],
            scaled_df["Energy", "eV"],
            FILTER_WINDOW, FILTER_ORDER)
    sorted_scaled_df = scaled_df.sort_index(axis=1, level=0)
    return sorted_scaled_df


def smooth_n_derivative(rough_data, energy, window, porder):
    """
    Helper function to smooth data using a Savitzky-Golay filter.

    Input: rough_data (Series)
    Spec data scaled to ahv.
    Returns: derived_data (Series)
    Derivative of smoothed data.
    """
    smooth_data = signal.savgol_filter(rough_data, window, porder)
    derived_data = np.gradient(smooth_data, energy)
    return derived_data


# noinspection PyUnusedLocal
def gaussian_smooth(rough_data, energy, sigma, *args):
    """
    Helper function to smooth data using a Gaussian filter.

    Input: rough_data (Series)
    Spec data scaled to ahv.
    Returns: derived_data (Series)
    Derivative of smoothed data.
    """
    smooth_data = gaussian_filter(rough_data, sigma, order=0)
    derived_data = np.gradient(smooth_data, energy)
    return derived_data


def linear_region_extrapolater(scaled_df, test=False):
    """
    Calculates the band gap using Tauc-like plots from a line built
    from the max derivative. order should be around the width of the
    derivative 'bump'.

    Input : scaled_df (DataFrame)
    Dataframe with columns of energy, ahv data and data derivatives
    for each temperature.
    Returns : linear_region_df (DataFrame)
    Dataframe of gradient and y-intercept of lines extrapolated from the
    linear region of the data, energy value at x-intercept is
    the optical band gap.
    """
    # Approximate width of derivative peak around linear region
    extrema_width = 50
    linear_region_df = pd.DataFrame()
    for temperature in scaled_df["data_derivative"].columns[3:5]:  # temp slice
        derivative_array = np.array(scaled_df["data_derivative", temperature])

        # Correct index is second to last, final is at far right and 
        # since derivatives are so low at low energies
        # many false peaks appear
        max_index = signal.argrelextrema(derivative_array,
                                         np.greater,
                                         order=extrema_width)[0][
            -2]  # testing slice

        if test:
            plt.plot(scaled_df["Energy", "eV"], derivative_array)
            idx = max_index
            plt.vlines(scaled_df["Energy", "eV"][idx],
                       0, 60)
            plt.vlines(scaled_df["Energy", "eV"][idx + extrema_width],
                       0, 60, linestyles='dashed')
            plt.vlines(scaled_df["Energy", "eV"][idx - extrema_width],
                       0, 60, linestyles='dashed')
            plt.grid(True)
            plt.show()

        max_derivative = derivative_array[max_index]
        max_derivative_data = scaled_df["ahv_data", temperature][max_index]
        max_derivative_energy = scaled_df["Energy", "eV"][max_index]
        linear_region_y_int = max_derivative_data - max_derivative * max_derivative_energy
        # Important to match multi-index
        linear_region_df[
            "ahv_data", temperature] = max_derivative, linear_region_y_int

    # add option for regression fit within window
    return linear_region_df


class InteractivePlot:
    """Interactively shows different smoothing options to show Daniel."""

    def __init__(self, scaled_df):
        # Initialize data and plot
        self.window, self.p_order = FILTER_WINDOW, FILTER_ORDER
        self.demo_data = scaled_df.iloc[:, 1:]
        self.temperature = self.demo_data.columns[0][1]  # Only uses 1st temp
        self.energy = scaled_df["Energy", "eV"]
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.4)
        self.ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.ax_slide_p_ord = plt.axes([0.25, 0.2, 0.65, 0.03])
        # Window size for Sav-gol filter should be odd
        self.win_size = Slider(
            self.ax_slide, 'Window size',
            valmin=3, valmax=99, valinit=FILTER_WINDOW, valstep=2)
        self.porder_size = Slider(
            self.ax_slide_p_ord, 'Poly order',
            valmin=0, valmax=9, valinit=FILTER_ORDER, valstep=1)
        self.ax.set_xlabel("Photon energy (eV)")
        self.ax.set_ylabel("Gradient")
        self.ax.grid(True)

        # Initial calculations
        self.demo_data["data_derivative_raw", self.temperature] = np.gradient(
            self.demo_data["ahv_data", self.temperature],
            self.energy)
        self.raw = self.ax.plot(self.energy,
                                self.demo_data["data_derivative_raw",
                                               self.temperature],
                                "b.", markersize=1, alpha=0.5)
        self.demo_data[
            "data_derivative", self.temperature] = smooth_n_derivative(
            self.demo_data["ahv_data", self.temperature],
            self.energy, FILTER_WINDOW, FILTER_ORDER)
        self.current, = self.ax.plot(self.energy,
                                     self.demo_data[
                                         "data_derivative", self.temperature],
                                     "r-", linewidth=0.5)
        self.win_size.on_changed(self.update_win)
        self.porder_size.on_changed(self.update_porder)
        plt.show()

    def update_win(self, window):
        """Updates the plot with data smoothed with new window size."""
        self.window = window
        self.demo_data[
            "data_derivative", self.temperature] = smooth_n_derivative(
            self.demo_data["ahv_data", self.temperature],
            self.energy,
            window, self.p_order)
        self.current.set_ydata(
            self.demo_data["data_derivative", self.temperature])
        self.fig.canvas.draw()

    def update_porder(self, porder):
        """Updates the plot with data smoothed with new polynomial order."""
        self.p_order = porder
        self.demo_data[
            "data_derivative", self.temperature] = smooth_n_derivative(
            self.demo_data["ahv_data", self.temperature],
            self.energy,
            self.window, porder)
        self.current.set_ydata(
            self.demo_data["data_derivative", self.temperature])
        self.fig.canvas.draw()


def tauc_plotter(scaled_df, linear_region_df):
    """Plots multiple tauc plots on one set of axes.
    Input:  scaled_df (DataFrame)
    Dataframe with columns of energy, ahv data and data derivatives
    linear_region_df (DataFrame)
    DataFrame with columns of max derivative and y-intercept for each temp
    Returns: None
    Shows plot.
    """
    for temperature in scaled_df["ahv_data"].columns:
        temperature_label = temperature.rstrip("(ahv)")
        plt.plot(scaled_df[("Energy", "eV")],
                 scaled_df["ahv_data", temperature],
                 label=temperature_label)

        max_derivative, linear_region_y_int = linear_region_df[
            "ahv_data", temperature]
        band_gap = -linear_region_y_int / max_derivative
        lin_region = max_derivative * scaled_df[
            ("Energy", "eV")] + linear_region_y_int
        plt.plot(scaled_df[("Energy", "eV")], lin_region, "--",
                 label=f"{temperature_label} band-gap: {band_gap:.3f}")

    plt.xlabel("Photon energy (eV)")
    plt.xlim(0, 5.5)
    plt.ylabel(r"$(\alpha h v)^{1/r}$")
    plt.ylim(0, 80)
    plt.grid(True)
    plt.legend()
    # plt.savefig("Tauc-like plot")
    plt.show()


def main():
    """Main function to take UV-Vis spec data and create Tauc-like plots
    determining band-gap temperature dependence."""
    low_t_data = get_data(LOW_T_FILE)
    high_t_data = get_data(HIGH_T_FILE)
    clean_data, all_baselines = data_cleaner(low_t_data, high_t_data)
    scaled_df = scale_n_differentiate(clean_data)

    # TODO: HW, clean up examples and code,
    #  implement regression and set points,
    #  finalize smoothing and give good examples

    InteractivePlot(scaled_df.iloc[:, [0, 10]])
    plt.show()

    # linear_region_df = linear_region_extrapolater(scaled_df)
    # tauc_plotter(scaled_df.iloc[:, [0,4,5]], linear_region_df)


if __name__ == "__main__":
    main()
