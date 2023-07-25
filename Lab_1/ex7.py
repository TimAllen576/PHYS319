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
    "Doofenshmirtz evil incorporatedd"    
    hours, vel = np.genfromtxt("DAex7data.txt")[:,0], np.genfromtxt("DAex7data.txt")[:,1]
    nfft = max(hours)
    H = np.fft.fft(vel)
    mx = abs(H[1:int(nfft/2)])*(2.0/nfft)
    period = nfft/np.arange(1,int(nfft/2))
    if mx[11] > mx[23]:
        print("The semi-diurnal tide is largest during this period")
    else:
        print("The diurnal tide is largest during this period")
    plotter(period, mx)
    
main()

