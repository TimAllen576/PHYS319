# -*- coding: utf-8 -*-
"""
%   Demonstration script to create some simulated data
%   and then fit a polynomial to it.


Created on Sun Jul 12 11:13:47 2020

@author: ajm226
Edited by tal75
"""



import numpy as np
import matplotlib.pyplot as plt


MAX_ = 3


#
#   generate the data
#
n = 100                   # number of data points
x = np.random.ranf(n)*15-5;        # random x data values in range -5 to 10 uniformly distibuted
xx = np.arange(-5.5,10.501,0.01)       # x vector to use for plotting
p = [1, -10, -10, 20]        # polynomial coefficients

y = np.polyval(p,x)          # y data values
sigma = np.random.rand(n)*25;      # random uniformly distributed uncertainties in range 0 to 25
error=np.random.normal(0.0,1.0,n)*sigma   # add gaussian-distributed random noise
y=y+error



#
#    plot the data points and the cubic polynomial
#


def plotter(p_coeffs):
    "Ill leave the rest but this is too bad"
    plt.figure()
    plt.errorbar(x,y,xerrr=None,yerr=sigma,linestyle="none", marker="*", label="Data")
    plt.plot(xx,np.polyval(p,xx),color="black", label= "Original fit")
    #fit a 4-coefficient polynomial (i.e. a cubic) to the data
    #and quantify the quality of the fit
    y_plot = np.polyval(p_coeffs,xx)
    y_comp = np.polyval(p_coeffs,x)
    chi_sq = np.sum(((y-y_comp)/sigma)**2)
    print(f"This gives a chi squared value of {chi_sq}.\nThe expected value is {len(x)-4} with variance {2*(len(x)-4)}.")
    plt.plot(xx, y_plot,color ="red", linestyle = "--", label= "My fit")
    plt.legend()
    plt.show(block=False)
    
def bas_func(xdata):
    "Returns a vector of the basis functions"
    x_k = [xdata**0]
    for power in range(1, MAX_+1):
        x_k = np.concatenate((x_k, [xdata**power]), axis=0)
    return x_k

def power_params(xdata, ydata):
    """Solves for parameters a using matrix methods
    assumes solution of polynomial form
    """
    x_k = bas_func(xdata)/sigma
    alpha = np.zeros((MAX_+1, MAX_+1))
    beta = np.sum(ydata * x_k/sigma, axis=1)
    for jcount, jvalue in enumerate(x_k):
        for icount, ivalue in enumerate(x_k):
            alpha[icount, jcount] = np.dot(jvalue, ivalue)
    c= np.linalg.inv(alpha)
    return np.matmul(c, beta), np.sqrt(np.diag(c)) 

def main():
    "(Doofenshmirtz intro music)"
    p_coeffs, uncerts = power_params(x, y)
    plotter(np.flip(p_coeffs))
    print("With equation y= a + bx + cx^2 + dx^3\n"
          "The uncertainties of the powers is:\n"
          f"Uncertainty of a: {uncerts[3]}\n"
          f"Uncertainty of b: {uncerts[2]}\n"
          f"Uncertainty of c: {uncerts[1]}\n"
          f"Uncertainty of d: {uncerts[0]}\n"
          )
    plt.show()

main()