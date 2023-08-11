# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import the relevant packages 
import serial 
import serial.tools.list_ports as port_lists 
import matplotlib.pyplot as plt
import numpy as np

"""
# Examine all the serial port hardware connected to the computer 
ports=list(port_lists.comports()) 
# print out the names of the various ports 
print() 
for p in ports: 
 print(p) 
print() 
# pick the relevant port 
port_string=input('Enter port to be used: ') 
"""

def initializer():
    "gets info for setup"
    # define the relevant serial port parameters and take control of the port 
    serial_connection=serial.Serial(port="COM3",baudrate=9600,timeout=2, 
    bytesize=serial.EIGHTBITS,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE, 
    xonxoff=0,dsrdtr=0,rtscts=0)
    
      # write a string to the port 
    write_check=serial_connection.write("ID?\r".encode())
    # read the string from the port 
    ID_value=serial_connection.readline() 
    #print(ID_value.decode()) 
     
    
    serial_connection.write("*esr?\r".encode())
    ID_value=serial_connection.readline() 
    #print(ID_value.decode())
    serial_connection.write("allev?\r".encode())
    ID_value=serial_connection.readline() 
    #print(ID_value.decode())
    serial_connection.write("data:source?\r".encode())
    ID_value=serial_connection.readline() 
    #print(ID_value.decode())
    serial_connection.write("data:encdg ascii\r".encode())
    ID_value=serial_connection.readline() 
    #print(ID_value.decode())
    serial_connection.write("data:width?\r".encode())
    ID_value=serial_connection.readline() 
    #print(ID_value.decode())
    serial_connection.write("data:start?\r".encode())
    ID_value=serial_connection.readline() 
    #print(ID_value.decode())
    serial_connection.write("data:stop?\r".encode())
    ID_value=serial_connection.readline() 
    #print(ID_value.decode())
    serial_connection.write("wfmpre?\r".encode())
    ID_value=serial_connection.readline() 
    #print(ID_value.decode())
    serial_connection.write("curve?\r".encode())
    ID_value=serial_connection.readline()
    #print(ID_value.decode())
    # close the port 
    serial_connection.close() # note if you do not do this you might hang-up the port 
    return ID_value.decode()

def com_helper(function):
    "Helper func for encode and decode"
    serial_connection=serial.Serial(port="COM3",baudrate=9600,timeout=2, 
    bytesize=serial.EIGHTBITS,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE, 
    xonxoff=0,dsrdtr=0,rtscts=0)
    serial_connection.write(function.encode())
    return serial_connection.readline().decode()


def data_collect():
    "Gets data"
    com_helper("data:encdg ascii\r")
    curve = com_helper("curve?\r")
    samp_int = float(com_helper("wfmpre:xincr?\r")[14:])
    x_unit = com_helper("wfmpre:xunit?\r")[15:-2]
    y_unit = com_helper("wfmpre:yunit?\r")[15:-2]
    y_off = float(com_helper("wfmpre:yoff?\r")[13:])
    y_mult = float(com_helper("wfmpre:ymult?\r")[14:])
    #print(f"y mult = {y_mult}")
    end_x = samp_int*len(curve)
    y = (np.fromstring(curve[7:], sep=','))*y_mult + y_off
    return y, end_x, x_unit, y_unit,samp_int

def plotter(x, y, x_unit, y_unit, xlim= None):
    "plots a graph of the waveform"
    plt.plot(x, y, "b-")
    plt.xlabel(x_unit)
    plt.ylabel(y_unit)
    plt.xlim(0,xlim)
    plt.grid()
    plt.show()

def fourier_trans(x, y, samp_int):
    "Does a fourier transform of the data and gives amplitude and frequency"
    nfft = len(y)
    four = np.fft.fft(y)
    mx = abs(four[0:int(nfft/2)])*(2.0/nfft)
    freq = np.arange(0,int(nfft/2))/(nfft*samp_int)
    return mx, freq

def main():
    "Do the thing"
    y, end_x, x_unit, y_unit, samp_int = data_collect()
    time_data = np.linspace(0, end_x, len(y))
    plotter(time_data, y, x_unit, y_unit)
    amp, freq = fourier_trans(time_data, y, samp_int)
    plotter(freq, amp, "Frequency(Hz)", "Amplitude(V)", 250000)

main()
    
    
    
    