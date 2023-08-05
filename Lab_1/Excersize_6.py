# -*- coding: utf-8 -*-
"""
Example Fourier Transform code for PHYS381 Data analysis assessment


Created on Sun Jul 12 11:52:42 2020

@author: ajm226
Edited by tal75
"""
import numpy as np
import matplotlib.pyplot as plt


fs =11 # sampling frequency (Hz) 
ts = 1/fs # set sampling rate and interval
period=1 #sampling period (s)
nfft = period/ts # length of DFT
t=np.arange(0,period,ts)
h = ((1.3)*np.sin(2*np.pi*35.0*t) )#+ (1.7)*np.sin((2*np.pi*35.0*t)-0.6) + (2.5)*np.random.normal(0.0,1.0,t.shape))
# combination of a 5 Hz signal a 35Hz signal and Gaussian noise
# 35Hz and noise commented out for clarity, 5->35


H = np.fft.fft(h) # determine the Discrte Fourier Transform


# Take the magnitude of fft of H
mx = abs(H[0:int(nfft/2)])*(2.0/nfft)  # note only need to examine first half of spectrum
# Frequency vector
f = np.arange(0,int(nfft/2))*fs/nfft

#np.int is deprecated, repeat removed


plt.figure(1)
plt.plot(t,h);
t2=np.arange(0,period,0.001);
h2= (1.3)*np.sin(2*np.pi*35.0*t2);
plt.plot(t2, h2, "m-");
plt.title('Sine Wave Signals');
plt.xlabel('Time (s)');
plt.ylabel('Amplitude');
plt.ylim(-2, 2)


plt.figure(2)
plt.scatter(f,mx)
plt.title('Amplitude Spectrum of Sine Wave signals');
plt.xlabel('Frequency (Hz)');
plt.ylabel('Amplitude');
plt.ylim(-2, 2)
plt.show()

"""
For sampling freq in between true and 2 x true the aliased freq is: sample - true
For sampling freq below it folds at 2/N fractions of true, oscillating between inc and dec
"""