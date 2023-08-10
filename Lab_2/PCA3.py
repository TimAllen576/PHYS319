
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from read_station_data import datetime_array_create
from PCA_artificial_data import Principal_Component_Analysis
from PCA2 import load_all_data

#z = np.ma.masked_values([1.0, 1.e20, 3.0, 4.0], 1.e20)
#val = [0, 0, 5, 6, 5]
#val_i = np.where(val == -99.0, True , False)
#print(f'joe mama{val_i}')

def invalid_data_stripper(tt_df):
    """takes a dataframe, slices to dates between Jan 1990 and Dec 2005 then drops columns with invalid data(-99.0)
    Input: dataframe with datatime indexes and filename columns
    Returns: shortened dataframe with bad data removed
    """
    temperature_time_stripped = tt_df.loc['1990-01-01':'2005-12-01']
    for column in temperature_time_stripped.columns:
        if any(temperature_time_stripped[column].isin([-99.0])):
            temperature_time_stripped = temperature_time_stripped.drop(columns = column)
    return temperature_time_stripped

def main():
    "Does something"
    tt_df= load_all_data()
    temperature_time_good = invalid_data_stripper(tt_df)

    val_i = np.any(np.where(temperature_time_good == -99.0, True , False))
    print(f'Invalid entries: {val_i}')    #check for an invalid entries

    latitude, longitude, station = zip(*temperature_time_good.columns)
    latitude_array, longitude_array, station_array = np. asarray(latitude), np. asarray(longitude), np. asarray(station)
    temperature_array = temperature_time_good.to_numpy()    # Converts all pandas info to numyp arrays

    np.savez(
        "good_station_data",
        latitude_array=latitude_array,
        longitude_array=longitude_array,
        station_array=station_array,
        temperature_array=temperature_array)    # Saves arrays with names
    
main()