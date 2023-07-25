# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:45:56 2023

@author: ajm226
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta


def read_lat_lon_metadata(filename):
    input_file=open(filename)
    for line in input_file:
        if(line[0:4]=='Lat='):
            lat=float(line[5:])
        if(line[0:5]=='Long='):
            lon=float(line[6:])      
    return lat,lon

def datetime_array_create(start_year,end_year):
    # create a datetime array using start and end year information
    start_month=1
    start_day=1
    start_hour=0
    start_minute=0
    start_sec=0
    base=datetime.datetime(start_year,start_month,start_day,start_hour,start_minute,start_sec,0)
    date_array=np.array([base+relativedelta(months=i) for i in range(int((end_year-start_year)*12))])
    return date_array


def read_station_data_file(filename):
    #this code reads the formatted station data file
    # input= filename
    # output is latitude,longitude, datetime_array and temperature_array
    tmp=np.loadtxt(filename,skiprows=22)  #skipping metadata information to get to temperature data
    year_array=tmp[:,0]
    temperature=tmp[:,1:]
    temperature_array=np.reshape(temperature,(temperature.shape[0]*temperature.shape[1],1))
    datetime_array=datetime_array_create(int(year_array[0]),int(year_array[-1])+1)
    [lat,lon]=read_lat_lon_metadata(filename)
    return lat,lon, datetime_array,temperature_array



filename='DAex5_data1.dat'
[lat,lon, datetime_array,temperature_array]=read_station_data_file(filename)

plt.plot(datetime_array,temperature_array)
plt.xlim(datetime.datetime(1980,1,1,0,0,0),datetime.datetime(1985,1,1,0,0,0))
plt.ylim(-10.0,20.0)
plt.xlabel('Date')
plt.ylabel('Temperature ($^o$C)')

