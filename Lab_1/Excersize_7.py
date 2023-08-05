# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 2023

@author: tal75
"""

import numpy as np
import matplotlib.pyplot as plt


def plotter(x, y):
    "makes a plot or smthn of the req place a do"
    plt.plot(x,y, color = "black", linewidth = 0.5)
    #Formatting fluff
    plt.axvline(x = 12, color = 'b', linewidth = 0.5)
    plt.text(9, 13, 'semi-diurnal tide', rotation = 90, verticalalignment='center', c='b')
    plt.axvline(x = 24, color = 'r', label = 'diurnal tide', linewidth = 0.5)
    plt.text(20, 13, 'diurnal tide', rotation = 90, verticalalignment='center', c='r')
    plt.xlabel("Period (hours)")
    plt.ylabel("Amplitude (m/s)")
    plt.xscale("log")
    plt.ylim(bottom= 0)
    plt.xlim(1, max(x))
    plt.tick_params(top = True, which='both')
    plt.tick_params(right = True)
    plt.tick_params(which='both', direction="in")
    #
    plt.show()


def main():
    "Doofenshmirtz evil incorporated"    
    hours, vel = np.genfromtxt("DAex7data.txt")[:,0], np.genfromtxt("DAex7data.txt")[:,1]
    l_fourier = len(hours) # length of FT
    fourier = np.fft.fft(vel)
    fourier_half = abs(fourier[1:int(l_fourier/2)])*(2.0/l_fourier) # Cut off duplicate half of FT and rescale 
    period = l_fourier/np.arange(1,int(l_fourier/2))    # 0 value skipped
    if fourier_half[11] > fourier_half[23]:
        print("The semi-diurnal tide is largest during this period")
    else:
        print("The diurnal tide is largest during this period")
    plotter(period, fourier_half)
    
main()

