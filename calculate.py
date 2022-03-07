# Purpose:  In this program, calculate Chay model and Riz model.
#
# Author:   K.Koyama
#
# Function:
#   - ion_currents
#           Output power and current for energy consumption calculation of Chay model.
#   - cal_chay_input
#           Calculate Chay model with digital on-demand input and noise applied.
#   - cal_riz_input
#           Calculate Riz model with digital on-demand input and noise applied.


import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit
import copy, math

from model import Chaymodel, Rizmodel
from numerical_solution import runge_kutta, stochastic_euler



@jit(nopython = True)
def ion_currents(variables, t, params):
    ipt, V_K, V_I, V_L, V_C, g_I_star, g_KV_star, g_KC_star, g_L_star, rho, k_C = params

    V = variables[0]
    n = variables[1]
    C = variables[2]
    
    a_m = 0.1*(25+V)/(1-math.exp(-0.1*V-2.5))
    b_m = 4*math.exp(-(V+50)/18)
    a_h = 0.07*math.exp(-0.05*V-2.5)
    b_h = 1/(1+math.exp(-0.1*V-2))
    a_n = 0.01*(20+V)/(1-math.exp(-0.1*V-2))
    b_n = 0.125*math.exp(-(V+30)/80)

    tau_n = 1/(230*(a_n+b_n))

    m_inf = a_m/(a_m+b_m)
    h_inf = a_h/(a_h+b_h)
    n_inf = a_n/(a_n+b_n)
    
    I_I  = g_I_star*(m_inf**3)*h_inf*(V-V_I)
    I_KV = g_KV_star*(n**4)*(V-V_K)
    I_KC = g_KC_star*(C/(1+C))*(V-V_K)
    I_L  = g_L_star*(V-V_L)
    
    P = abs(I_KV*V_K)+abs(I_KC*V_K)+abs(I_L*V_L)-abs(I_I*V_I)
    
    return  (np.array([I_I,I_KV,I_KC,I_L])),P/1000



@jit(nopython=True)
def cal_chay_input(cellout, step_info, params, input_info, noise_info):

    # Step
    ttime = 0
    tau, NSKIP, NSTEPS = step_info[0], int(step_info[1]), int(step_info[2])
    NSTEPS += NSKIP
    
    # Output
    output_array = np.zeros((3,NSTEPS-NSKIP))
    V_array = np.zeros(NSTEPS-NSKIP)
    input_array = np.zeros(NSTEPS-NSKIP)
    P_array = np.zeros(NSTEPS-NSKIP)
    
    # Info of Burst
    timing_or_zero_array = np.zeros(NSTEPS-NSKIP)
    threshold = -42
    V_before  = 0
    
    # Info of design
    a = input_info[0]
    l = input_info[1] 
    V_th = input_info[2]
    
    # noise_info
    D = noise_info[0]
    sd = int(noise_info[1]) # noise_seed(int)

    ltime = 0
    input_c = 0

    # stochastic system (noise input)
    np.random.seed(sd)
    gwn_array = np.random.normal(0, 1, NSTEPS)*(tau**(1/2))

    for i in range(NSTEPS):

        # stochastic system
        input_noise = D*gwn_array[i]

        params[0] = input_c
        if D > 0:
            cellout = stochastic_euler(cellout, tau, Chaymodel, params, input_noise)
        else:
            cellout = runge_kutta(cellout, ttime, tau, params, Chaymodel)
        
        currents, P = ion_currents(cellout, ttime, params)
        
        if i >= NSKIP:
            output_array[0][i-NSKIP] = cellout[0]
            output_array[1][i-NSKIP] = cellout[1]
            output_array[2][i-NSKIP] = cellout[2]
            input_array[i-NSKIP] = input_c
            P_array[i-NSKIP] = P
        
            # Get spike(or burst) timing
            if cellout[0] > threshold and V_before < threshold:
                timing_or_zero_array[i-NSKIP] = i*tau
            
        # ***** Design Input *****
        if cellout[0] >= V_th and V_before < V_th: # event occurence (pass threshold)
            ltime = ttime + l
        if ttime < ltime:
            input_c = a
        else:
            input_c, ltime = 0, 0
        # *************************

        V_before = cellout[0]
        ttime += tau

    # get ISIs (or IBIs)
    timing_array = timing_or_zero_array[timing_or_zero_array.nonzero()] # gain spike timing.
    isis_array = timing_array[1:]-timing_array[:-1] # gain ISIs
    ibis_array = isis_array[isis_array > 2] # gain IBIs

    return output_array, input_array, P_array, isis_array, ibis_array



@jit(nopython=True)
def cal_riz_input(cellout, step_info, conductance, input_info, noise_info):
    
    ttime = 0
    tau, NSKIP, NSTEPS = step_info[0], int(step_info[1]), int(step_info[2])
    NSTEPS += NSKIP
    output_array = np.zeros((14,NSTEPS-NSKIP))
    input_array = np.zeros(NSTEPS-NSKIP)
    
    # Info of Burst
    timing_or_zero_array = np.zeros(NSTEPS-NSKIP)
    threshold = -42
    V_before  = 0
    
    # Info of Design
    a = input_info[0]
    l = input_info[1] 
    V_th = input_info[2]
    
    # noise_info
    D  = noise_info[0]
    sd = int(noise_info[1])
    
    ltime = 0
    input_c = 0
    
    # Stochastic System
    np.random.seed(sd)
    gwn_array = np.random.normal(0,1,NSTEPS)*(tau**(1/2))

    for i in range(NSTEPS):

        # stochastic system
        input_noise = D*gwn_array[i]

        conductance[0] = input_c
        if D > 0:
            cellout = stochastic_euler(cellout, tau, Rizmodel, conductance, input_noise)
        else:
            cellout = runge_kutta(cellout, ttime, tau, conductance, Rizmodel)
        if i >= NSKIP:
            output_array[0][i-NSKIP] = cellout[0]
            output_array[1][i-NSKIP] = cellout[1]
            output_array[2][i-NSKIP] = cellout[2]
            output_array[3][i-NSKIP] = cellout[3]
            output_array[4][i-NSKIP] = cellout[4]
            output_array[5][i-NSKIP] = cellout[5]
            output_array[6][i-NSKIP] = cellout[6]
            output_array[7][i-NSKIP] = cellout[7]
            output_array[8][i-NSKIP] = cellout[8]
            output_array[9][i-NSKIP] = cellout[9]
            output_array[10][i-NSKIP] = cellout[10]
            output_array[11][i-NSKIP] = cellout[11]
            output_array[12][i-NSKIP] = cellout[12]
            output_array[13][i-NSKIP] = cellout[13]
            input_array[i-NSKIP] = input_c

            # Get spike(or burst) timing
            if cellout[0] > threshold and V_before < threshold:
                timing_or_zero_array[i-NSKIP] = i*tau
            
        # ***** Design Input *****
        if cellout[0] >= V_th and V_before < V_th: # event occurence (pass threshold)
            ltime = ttime + l
        if ttime < ltime:
            input_c = a
        else:
            input_c, ltime = 0, 0
        # *************************

        V_before = cellout[0]
        ttime += tau

    # get ISIs (or IBIs)
    timing_array = timing_or_zero_array[timing_or_zero_array.nonzero()] # gain spike timing.
    isis_array = timing_array[1:]-timing_array[:-1] # gain ISIs
    ibis_array = isis_array[isis_array > 200] # gain IBIs

    return output_array, input_array, isis_array, ibis_array


# Test for correct calculations.
if __name__ =="__main__":
    
    # select_model = "chay"
    select_model = "riz"
    # select_model = ""

    if select_model == "chay":
        # set Parameters
        # V_K, V_I, V_L, V_C, g_I_star, g_KV_star, g_KC_star, g_L_star, rho, k_C
        params = np.array([0, -75, 100, -40, 100, 1800, 1700, 11, 7, 0.27, 0.183333333333333])

        # set Step
        step_info = np.array([0.005,0,20000])
        # set Initial Variables
        cellout = np.array([-40, 0.4, 0.4])

        # set input_info
        input_info = np.array([5, 0.25,-45.7]) # a, l, V_th

        # set noise
        noise_info = np.array([0,0])

        # Calculate
        output_array, _, __, ___, ____ = cal_chay_input(cellout, step_info, params, input_info, noise_info)
        print('Calculation completed.')
        
        # Plot
        plt.plot(output_array[0])
        plt.title("Chay")
        plt.ylabel("$V$")
        plt.show()

    # ****************
    
    if select_model == "riz":
        # set conductnace
        conductance = np.array([0, 0.01, 0.02, 0.25, 0.01, 0.42, 0.4, 0.14, 0.17, 0.05, 0.015, -10])
        # set step
        step_info = np.array([0.01,0,1000000])
        
        # initial_value
        # (BBRC, Riz et al., 2015)
        ini_V    = -63
        ini_Ca_m = 0.3
        ini_Ca_c = 0.17
        ini_G6PF6P = 1.935191
        ini_FBP = 0.000090
        ini_DHAPG3P = 0.000407 
        ini_a = 0.050591
        ini_m_BK = 0.002
        ini_m_KV = 0.02
        ini_m_HERG = 0.012630
        ini_h_HERG = 0.858772
        ini_h_Na   = 0.97
        ini_h_CaL  = 0.98
        ini_h_CaT  = 0.52
        ini_cellout = np.array([ini_V, ini_Ca_m, ini_Ca_c, ini_G6PF6P, ini_FBP, ini_DHAPG3P, ini_a, \
                            ini_m_BK, ini_m_KV, ini_m_HERG, ini_h_HERG, ini_h_Na, ini_h_CaL,ini_h_CaT])

        cellout = copy.deepcopy(ini_cellout)

        # set input_info
        input_info = np.array([10, 0.25, -52.0])
        # set noise_info
        #noise_info = np.array([0.0003, 4])
        noise_info = np.array([0.003, 4])
        output_array, _, __, ___ = cal_riz_input(cellout, step_info, conductance, input_info, noise_info)

        print('Calculation completed.')
        
        # Plot
        plt.plot(output_array[0])
        plt.title("Riz")
        plt.ylabel("$V$")
        plt.show()



