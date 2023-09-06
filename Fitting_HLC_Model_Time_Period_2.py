# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:51:11 2023

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

xaxisData = np.linspace(0 , 28 , 29) #Independent Variable

yaxisData = np.array([scaled_h_pop ,scaled_l_pop]) #Dependent Variables

H_0 = 9741.78 #Initial Hare Population

L_0 = 93.79 #Initial Lynx Population

T_0 = 0.31729 #Initial Trapper or Hunter Population

params0 = (0.00688 ,  2.05204 , 0.05070 ,  1.08531 , 5.80864 , 0.02299 ,  0.00008 , 0.00068 ,  0.10828 , 0.00191 , 0.02673 , 0.12834 , 8.44143  , 0.00024 , 0.00345) #Initial Guess for Parameters

y_0  = [H_0 , L_0 , T_0 ]

def to_be_solved( y, t , param):
    
    """
    Input: The values of the state variables, time and the value of the parameters
    
    Output: The values of dHdt , dLdt ,  dCdt , dTdt at those times
    
    """
        
    H , L  , T  = y[0] , y[1] , y[2] 
                
    a_1 , b, b_1 , d_1  , d_3 , h_1 ,  m, m_1  , m_3 , r_1, r_2 , u_1 , u_2 , v_1 , v_2  = param[0] , param[1] , param[2] , param[3] , param[4] , param[5] , param[6] , param[7] , param[8] , param[9], param[10] , param[11] , param[12] , param[13] , param[14] 
    
    dHdt = H*(b-m*H-a_1*L/(1+h_1*a_1*H) - u_1*T/(1 + v_1*u_1*  H + v_2*u_2*L))
    
    dLdt = L*(b_1*a_1*H/(1+h_1*a_1*H) - d_1 - m_1*L - u_2*T/(1+v_1*u_1*H+v_2*u_2*L))
    
    dTdt = T*((r_1*u_1*H+r_2*u_2*L)/(1 + v_1*u_1*H + v_2*u_2*L) - d_3 - m_3*T)

    return [ dHdt , dLdt , dTdt ]



#The value of parameters given from the original study

def model(xaxisData,*params):
    
    '''
    Input: The values of parameters and the timeframe for which the model will run for
    
    Output: The populations of the hare at lynx in a single array (format accessible by scipy curvefit)
    
    '''
    y_0 = [ scaled_h_pop[0] , scaled_l_pop[0] , 0.31729 ]
    
    numYaxisVariables = 3

    yaxisCalc = np.zeros((xaxisData.size,numYaxisVariables))
    
    for i in np.arange(0,len(xaxisData)):
        
        if xaxisData[i] == 0.0: # should include a decimal
            
            # edit for > 1 dependent variables:            
            yaxisCalc[i,:] = y_0
        
        else:
            xaxisSpan = np.linspace(0.0 , xaxisData[i], 101)
            ySoln = odeint(to_be_solved,y_0,xaxisSpan,args = (params,)) # soln for entire xaxisSpan
            # edit for > 1 dependent variables:            
            yaxisCalc[i,:] = ySoln[-1,:] # calculated y at the end of the xaxisSpan
            # at this point yaxisCalc is now 2D matrix with the number of columns set as : to include all yvariables
            # curve_fit needs a 1D vector that has the rows in a certain order, which result from the next two commands

    yaxisOutput = np.transpose(yaxisCalc)
    
    yaxisOutput = yaxisOutput[0:2]
    
    yaxisOutput = np.ravel(yaxisOutput)
    
    return yaxisOutput

# bnds = ((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), (100,100,100,100,100,100,100,100,100,100,100,100,100,100,100))

bnds = ((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf))

parametersoln, pcov = curve_fit(model , xaxisData , np.ravel(yaxisData), p0=params0 , bounds = bnds)

yaxisCalc= model(xaxisData,*parametersoln)

yaxisCalc = np.reshape(yaxisCalc,(2,29))

print(f"a {parametersoln[0]}\nb {parametersoln[1]} \nc {parametersoln[2]} \nd {parametersoln[3]} \ne {parametersoln[4]} \nf {parametersoln[5]}")
print(f"g {parametersoln[6]}\nh {parametersoln[7]} \nr {parametersoln[8]} \nk {parametersoln[9]} \nm {parametersoln[10]}")

fig, ax = plt.subplots(2, sharex='col', sharey='row')

ax[0].plot(ts , yaxisCalc[0] , color = "tab:blue" , label = "Predicted")

ax[0].set(ylabel = "Population (100 000)",title='Experimental and Predicted Populations for the Hare')

ax[0].plot(ts , scaled_h_pop ,'x' , label = "Experimental" , color =  "tab:blue")

ax[1].plot(ts , yaxisCalc[1] , color ="tab:orange" ,label = "Predicted " )

ax[1].set(xlabel="Years" , ylabel = "Population (100 000)", title='Experimental and Predicted Populations for the Lynx')

ax[1].plot(ts , scaled_l_pop , 'x' , color = "tab:orange" , label = "Experimental")