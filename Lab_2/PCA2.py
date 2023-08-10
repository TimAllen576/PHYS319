# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:18:33 2023

@author: tal75
"""

import glob
import pandas as pd

from read_station_data import read_station_data_file

def load_all_data():
    "Loads all data in the dir Station_data into arrays lat,lon, and date and temp into a dataframe and returns them"
    start_date  = None
    end_date  = None
    time_ntemp_df = pd.DataFrame()

    for path in glob.glob("Station_data/*/*")[:40]:
        cache = read_station_data_file(path)
        dates = cache[2]
        values = cache[3]

        if start_date  is None or min(dates) < start_date :     # Determines the earliest and latest observations to appropriately size the dataframe
            start_date  = min(dates)
        if end_date  is None or max(dates) > end_date :
            end_date  = max(dates)

        meta_data_tuples = [(cache[0], cache[1], path)]     # Inserting latitude and longitude data into dataframe
        column_data = pd.MultiIndex.from_tuples(meta_data_tuples, names=["Latitude", "Longitude", "Station"])

        cache_df = pd.DataFrame(data=values, index=dates, columns=column_data)   # Create a temporary dataframe for each file and interpolate missing values
        cache_date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        cache_df = cache_df.reindex(cache_date_range)

        time_ntemp_df = pd.concat([time_ntemp_df, cache_df], axis=1) # Merge the temporary dataframe into the main dataframe

    return time_ntemp_df

def main():
    "Do the thingy"
    tt_df= load_all_data() # Overdone cos I misread the question but it works
    #print(f'Temperature dataframe:\n{tt_df}')

main()