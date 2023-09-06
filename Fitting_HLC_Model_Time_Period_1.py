# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:29:15 2023

@author: HP
"""

#Importing Relevant Packages and the Original Data Set


import numpy as np

from scipy.integrate import odeint

from matplotlib import pyplot as plt

from scipy.optimize import curve_fit

import pandas as pd

import scipy

#Enter the Filepath for where you stored the original Data Set on your computer

Filepath_dataset = r"C:\Users\HP\Downloads\Code Repository\Fitting the HLC Model to Time Period 1\Lynx_Snowshoehare_dataset.txt"

data = pd.read_csv( Filepath_dataset ,  delim_whitespace=True, header=None , index_col = None , names = ["Year" , "Hare Population (1000s)" , "Lynx Population (1000s)"])

#Parameter Optimisarion for Time Period 1 using the HLC Model

h_pop = data["Hare Population (1000s)"]

l_pop = data["Lynx Population (1000s)"]

h_pop_array = h_pop.values

l_pop_array = l_pop.values

scaled_h_pop = np.array((h_pop_array[55:90])/100) #Choose the data in the last 30 years

scaled_l_pop = np.array((l_pop_array[55:90])/100) #Choose the data in the last 30 years

xaxisData = np.linspace(0 , 34 , 35) #Independent Variable

yaxisData = np.array([scaled_h_pop ,scaled_l_pop]) #Dependent Variables

params0 = [ 5.05 , 8.58 , 0.01 , 0.01 , 0.01, 0.01 , 0.01 , 0.01 , 0.08 , 0.0175 , 2] #Initial Guess for Parameters

new_ts =  np.linspace(1901 , 1935 , 35)

def to_be_solved( y, t , param):
    
    '''
    Input: A state vector, whose components are the populaation of the hare, lynx and coyote; the timeframe and vector includnng all the values of the paraee
    
    Output: The populations of hare and lynx as a single array annually (in a formal accessiblw by curve_fit)

    '''
        
    N , P_1 , P_2  = y[0] , y[1] , y[2] 
            
    a , b , c , d , e , f, g, h, r , k , m = param[0] , param[1] , param[2] , param[3] , param[4] , param[5] , param[6] ,param[7] , param[8] , param[9] , param[10] 
                                              
    
    dNdt = r*N*(1-N/k) -a*N*P_1 - b*N**2*P_2/(1+m*N**2)
    
    dP_1dt = -c*P_1 + d*P_1*N - g*P_1*P_2

    dP_2dt = -e*P_2 + f*N**2*P_2/(1+m*N**2) + h*P_1*P_2

    return [ dNdt , dP_1dt ,  dP_2dt ]

def model(xaxisData,*params):
    
    '''
    Input: The values of parameters and the timeframe for which the model will run for
    
    Output: The populations of the hare at lynx in a single array (format accessible by scipy curvefit)
    
    '''
    y_0 = [ scaled_h_pop[0] , scaled_l_pop[0] , 4.5/100]
    
    numYaxisVariables = 3

    yaxisCalc = np.zeros((xaxisData.size,numYaxisVariables))
    
    for i in np.arange(0,len(xaxisData)):
        
        if xaxisData[i] == 0.0: # should include a decimal
            
            # edit for > 1 dependent variables:            
            yaxisCalc[i,:] = y_0
        
        else:
            xaxisSpan = np.linspace(0.0,xaxisData[i],101)
            ySoln = odeint(to_be_solved,y_0,xaxisSpan,args = (params,)) # soln for entire xaxisSpan
            # edit for > 1 dependent variables:            
            yaxisCalc[i,:] = ySoln[-1,:] # calculated y at the end of the xaxisSpan
            # at this point yaxisCalc is now 2D matrix with the number of columns set as : to include all yvariables
            # curve_fit needs a 1D vector that has the rows in a certain order, which result from the next two commands

    yaxisOutput = np.transpose(yaxisCalc)
    
    yaxisOutput = yaxisOutput[0:2]
    
    yaxisOutput = np.ravel(yaxisOutput)
    
    return yaxisOutput

yaxisCalc= model(xaxisData,*params0)

yaxisCalc = np.reshape(yaxisCalc,(2,35))

pred_h_pop , pred_l_pop = yaxisCalc[0] , yaxisCalc[1]

# bnds = ((0,0,0,0,0,0,0,0,0,0,0), (np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf))

bnds = ((0,0,0,0,0,0,0,0,0,0,0), (100,100,100,100,100,100,100,100,100,100,100)) #Setting the bounds between 0 and 100 give us a better bit than 0 to infinity

parametersoln, pcov = curve_fit(model,xaxisData,np.ravel(yaxisData), p0=params0 , bounds = bnds)

# print(parametersoln)

print(f"a {parametersoln[0]}\nb {parametersoln[1]} \nc {parametersoln[2]} \nd {parametersoln[3]} \ne {parametersoln[4]} \nf {parametersoln[5]}")
print(f"g {parametersoln[6]}\nh {parametersoln[7]} \nr {parametersoln[8]} \nk {parametersoln[9]} \nm {parametersoln[10]}")

yaxisCalc= model(xaxisData,*parametersoln)

yaxisCalc = np.reshape(yaxisCalc,(2,35))

fig, ax = plt.subplots(2, sharex='col', sharey='row')

ax[0].plot(new_ts , yaxisCalc[0] , color = "tab:blue" , label = "Predicted")

ax[0].set(ylabel = "Population (100 000)",title='Experimental and Predicted Populations for Hares')

ax[0].plot(new_ts , scaled_h_pop ,'x' ,label = "Experimental" , color =  "tab:blue")

ax[1].plot(new_ts , yaxisCalc[1] , color ="tab:orange" ,label = "Predicted " )

ax[1].set(xlabel="Years" , ylabel = "Population (100 000)", title='Experimental and Predicted Populations for Lynx')

ax[1].plot(new_ts , scaled_l_pop , 'x',color ="tab:orange" , label = "Experimental")
