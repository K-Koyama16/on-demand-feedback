# Purpose:  In this program, simulate digital on-demand input to the stochastic Riz model.
#
# Author:   K.Koyama

import os
import numpy as np
import copy
from tqdm import tqdm

from calculate import *

# Set "ax0_array (a)", "ax2_array (V_th)", l.
# * ax1_array is noise_seed.
ax0_array = np.array([round(0 + 0.5*i,2) for i in range(41)]) # a
l = 0.25 
ax2_array = np.array([round(-65 +0.5*i,2) for i in range(41)]) # V_th

# Set "save_dir" and "name".
save_dir = "stochastic_riz"
name = "d3_seed0-4"


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

# Step Info
step_info = np.array([0.01, 10000, 1000000])

# Parameters (input + conductance)
conductance = np.array([0, 0.01, 0.02, 0.25, 0.01, 0.42, 0.4, 0.14, 0.17, 0.05, 0.015, -10])

# Noise info
seed_array = np.array([0,1,2,3,4])
ax1_array = copy.deepcopy(seed_array)

# Output
result_num_spike = np.zeros((len(ax0_array), len(ax1_array), len(ax2_array)))
result_num_burst = np.zeros((len(ax0_array), len(ax1_array), len(ax2_array)))
result_cv = np.ones((len(ax0_array), len(ax1_array), len(ax2_array)))*100
result_std_ibis = np.ones((len(ax0_array), len(ax1_array), len(ax2_array)))*100

# ***** Simulation *****

for idx0, ax0 in enumerate(tqdm(ax0_array)):
    for idx1, ax1 in enumerate(ax1_array):
        for idx2, ax2 in enumerate(ax2_array):
            
            cellout_copy = copy.deepcopy(ini_cellout)

            input_info = np.array([ax0, l, ax2])
            noise_info = np.array([0.003, int(ax1)]) # noise strength D = 0.003

            output, input_current, isis, ibis = cal_riz_input(cellout_copy, step_info, conductance, input_info, noise_info)

            # Record
            if len(isis) > 0: result_num_spike[idx0][idx1][idx2] = len(isis) + 1
            if len(ibis) > 0: result_num_burst[idx0][idx1][idx2] = len(ibis) + 1
            if len(ibis) > 1: result_cv[idx0][idx1][idx2] = np.std(ibis)/np.mean(ibis)
            if len(ibis) > 1: result_std_ibis[idx0][idx1][idx2] = np.std(ibis)
            
# Save
os.makedirs(os.path.join('result', save_dir), exist_ok=True)
np.save(f'./result/{save_dir}/num_spike_{name}' , result_num_spike)
np.save(f'./result/{save_dir}/num_burst_{name}' , result_num_burst)
np.save(f'./result/{save_dir}/cv_{name}' , result_cv)
np.save(f'./result/{save_dir}/std_ibis_{name}' , result_std_ibis)