# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:55:39 2023

@author: HP
"""
import numpy as np

from scipy.integrate import odeint

from matplotlib import pyplot as plt

from scipy.optimize import curve_fit

import pandas as pd

import scipy

#Enter the Filepath for where you stored the original Data Set on your computer

Filepath_dataset = r"C:\Users\HP\Downloads\Code Repository\Fitting the HLC Model to Time Period 1\Lynx_Snowshoehare_dataset.txt"

data = pd.read_csv( Filepath_dataset ,  delim_whitespace=True, header=None , index_col = None , names = ["Year" , "Hare Population (1000s)" , "Lynx Population (1000s)"])

h_pop = data["Hare Population (1000s)"]

l_pop = data["Lynx Population (1000s)"]

h_pop_array = h_pop.values

l_pop_array = l_pop.values

h_pop_0 = np.array((h_pop_array[30:59]))

l_pop_0 = np.array((l_pop_array[30:59]))

ts = np.linspace(1875 , 1903 , 29) #We fit the model to Time Period 2 (1875 - 1903)

scaled_h_pop = h_pop_0/100

scaled_l_pop = l_pop_0/100

y_0 = [ scaled_h_pop[0] , scaled_l_pop[0] , 0.6 , 1.2]

xaxisData = np.linspace(0 , 28 , 29) #Independent Variable

yaxisData = np.array([scaled_h_pop ,scaled_l_pop]) #Dependent Variables

params = [2.27146495e-01 , 1.99368388e+00 , 2.26956970e+00 , 1.30215200e+00 , 
 2.95218273e-01 , 8.81330522e-02 , 2.75923377e-03 , 1.15812176e+01 ,
 2.04623478e+00 , 1.41445023e-01 , 3.28668521e-02 , 1.01391391e-01 ,
 1.98691636e-01 , 3.21956395e-01 , 4.23933841e-03 , 8.82016492e-02 ,
 1.27465801e-01 , 7.52913199e+00 , 6.44173890e-01 , 6.78223581e+00]

def to_be_solved( y , t , param):
    
    """
    Input: The values of the state variables, time and the value of the parameters
    
    Output: The values of dHdt , dLdt ,  dCdt , dTdt at those times
    """
        
    H , L , C , T  = y[0] , y[1] , y[2] , y[3]
                
    a_1, a_2 , b , b_1 , b_2 , d_1 , d_2 , d_3 , h_1 , h_2 , m , m_1 , m_2 , m_3 , r_1, r_2 , u_1 , u_2 , v_1 , v_2  = param[0] , param[1] , param[2] , param[3] , param[4] , param[5] , param[6] , param[7] , param[8] , param[9], param[10] , param[11] , param[12] , param[13] , param[14] , param[15] , param[16] , param[17] , param[18] , param[19]
    
    dHdt = H*(b-m*H-a_1*L/(1+h_1*a_1*H) - a_2*C/(1+h_2*a_2*H) - u_1*T/(1 + v_1*u_1*H + v_2*u_2*L))
    
    dLdt = L*(b_1*a_1*H/(1+h_1*a_1*H) - d_1 - m_1*L - u_2*T/(1+v_1*u_1*H+v_2*u_2*L))
    
    dCdt = C*(a_2*b_2*H/(1 + h_2*a_2*H) - d_2 - m_2*C)

    dTdt = T*((r_1*u_1*H+r_2*u_2*L)/(1 + v_1*u_1*H + v_2*u_2*L) - d_3 - m_3*T)

    return [ dHdt , dLdt ,  dCdt , dTdt ]

ySoln = odeint(to_be_solved,y_0,ts,args = (params,)) # soln for entire xaxisSpan

H , L , C , T = ySoln[:,0] , ySoln[:,1] , ySoln[:,2] , ySoln[:,3]

fig, ax = plt.subplots(2, sharex = 'col', sharey = 'row')

ax[0].plot(ts , C , color = "tab:blue" , label = "Predicted")

ax[0].set(title='Predicted Populations for the General Predator')

ax[1].plot(ts , T , color ="tab:orange" ,label = "Predicted " )

ax[1].set(xlabel="Years" , title='Predicted Population for Coyotes')

fig.text(0.03, 0.5, 'Population (100 000s)', va='center', rotation='vertical')