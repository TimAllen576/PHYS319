# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:53:16 2023

@author: tal75
"""

import numpy as np

from scipy.special import betainc
from scipy.stats import ttest_ind

def mean_and_num(data):
    "Finds the mean and number of entries then returns both"
    mean = np.mean(data)
    num = data.size
    return mean, num

def stderrormean(data1, data2, mean1, mean2, num1, num2):
    "Finds the std error"
    sum_difs = np.sum((data1-mean1)**2) + np.sum((data2-mean2)**2)
    tot_num  = num1+num2+2
    inv_nums = 1/num1 +1/num2
    s_d = np.sqrt((inv_nums)*sum_difs/tot_num)
    return s_d
    
def t_valsd(mean1, mean2, s_d):
    """
    Computes students t value by difference of means divides by standard error of means
    """
    t = (mean1 - mean2)/s_d
    return t


def incbeta_f(num1,num2,t):
    "Computes the incomplete beta function"
    df= num1 + num2 + 2
    #p_val = betainc(df/(df+t**2), df/2, 0.5)       # equation as written
    p_val = betainc(df/2, 0.5, df/(df+t**2))        # equation corrected for transcription errors
    return p_val

def sig_dif_means(filename1, filename2):
    "Finds probability that the null hypothesis is true for two datasets with students t test and the incomplete beta function"
    SampleA = np.genfromtxt(filename1)
    SampleA = SampleA[~np.isnan(SampleA)] # removes nans
    SampleB = np.genfromtxt(filename2)
    SampleB = SampleB[~np.isnan(SampleB)] # removes nans
    #p_val = ttest_ind(SampleA, SampleB, equal_var= False).pvalue
    #print(f"With Scipy probability functions the p-value that the two samples {filename1}, {filename2} have identical means is:  {p_val}")
    mean1, num1 = mean_and_num(SampleA)
    mean2, num2 = mean_and_num(SampleB)
    s_d = stderrormean(SampleA, SampleB, mean1, mean2, num1, num2)
    t =t_valsd(mean1, mean2, s_d)
    beta_f = incbeta_f(num1,num2,t)
    print(f"Using the given functions the p-value that the two samples have identical means is: {beta_f}")
    """
    print(f'With t-value:{t}')
    SampleA = np.append(SampleA, np.full(len(SampleB)-len(SampleA), np.nan))
    SampleA = SampleA.reshape(len(SampleA), 1)
    SampleB = SampleB.reshape(len(SampleB), 1)
    sample_df = pd.DataFrame(np.concatenate((SampleA, SampleB), axis=1), columns= ('SampleA', 'SampleB'))
    displot(data=sample_df)
    plt.show()
    """

def main():
    "Does the whole thing"
    sig_dif_means("SampleA.dat", "SampleB.dat")

main()
