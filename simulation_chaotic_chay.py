# Purpose:  In this program, simulate digital on-demand input to the chaotic Chay model.
#
# Author:   K.Koyama

import os
import numpy as np
import copy
from tqdm import tqdm
from scipy import integrate

from calculate import *

# Set "ax0_array (a)", "ax1_array (l)", "ax2_array (V_th)".
ax0_array = np.array([round(0 + 0.1*i,1) for i in range(101)]) # a
ax1_array = np.array([0.25]) # l
ax2_array = np.array([round(-47.5 +0.02*i,2) for i in range(226)]) # V_th

# Set "save_dir" and "name".
save_dir = "chaotic_chay"
name = "l0p25_allregion"

# initial_value
cellout = np.array([-40,0.4,0.4])

# Output
result_num_spike = np.zeros((len(ax0_array), len(ax1_array), len(ax2_array)))
result_num_burst = np.zeros((len(ax0_array), len(ax1_array), len(ax2_array)))
result_cv = np.ones((len(ax0_array), len(ax1_array), len(ax2_array)))*100
result_std_ibis = np.ones((len(ax0_array), len(ax1_array), len(ax2_array)))*100

result_input_energy  = np.zeros((len(ax0_array), len(ax1_array), len(ax2_array)))
result_output_energy = np.zeros((len(ax0_array), len(ax1_array), len(ax2_array))) 
result_ratio_energy  = np.zeros((len(ax0_array), len(ax1_array), len(ax2_array)))

# Step Info
step_info = np.array([0.005, 60000, 60001]) # 300[s] (Transient:300[s])
t = np.arange(0, step_info[0]*step_info[2],step_info[0])

# Parameters
params   = np.array([0, -75, 100, -40, 100, 1800, 1700, 11, 7, 0.27, 0.183333333333333])

# Noise info
noise_info = np.array([0,0])

# ***** Simulation *****

for idx0, ax0 in enumerate(tqdm(ax0_array)):
    for idx1, ax1 in enumerate(ax1_array):
        for idx2, ax2 in enumerate(ax2_array):
            cellout_copy = copy.deepcopy(cellout)

            input_info = np.array([ax0, ax1, ax2])

            output, input_current, power, isis, ibis = cal_chay_input(cellout_copy, step_info, params, input_info, noise_info)
            input_E = integrate.cumtrapz(abs(input_current),t,initial=0)
            output_E = integrate.cumtrapz(abs(power),t,initial=0)
            
            # Record
            if len(isis) > 0: result_num_spike[idx0][idx1][idx2] = len(isis) + 1
            if len(ibis) > 0: result_num_burst[idx0][idx1][idx2] = len(ibis) + 1
            if len(ibis) > 1: result_cv[idx0][idx1][idx2] = np.std(ibis)/np.mean(ibis)
            if len(ibis) > 1: result_std_ibis[idx0][idx1][idx2] = np.std(ibis)
            
            result_input_energy[idx0][idx1][idx2]  = np.max(input_E)
            result_output_energy[idx0][idx1][idx2] = np.max(output_E)
            if np.max(input_E) > 0:result_ratio_energy[idx0][idx1][idx2]  = np.max(output_E)/np.max(input_E)

# Save
os.makedirs(os.path.join('result', save_dir), exist_ok=True)
np.save(f'./result/{save_dir}/num_spike_{name}' , result_num_spike)
np.save(f'./result/{save_dir}/num_burst_{name}' , result_num_burst)
np.save(f'./result/{save_dir}/cv_{name}' , result_cv)
np.save(f'./result/{save_dir}/std_ibis_{name}' , result_std_ibis)
np.save(f'./result/{save_dir}/input_energy_{name}' , result_input_energy)
np.save(f'./result/{save_dir}/output_energy_{name}', result_output_energy)
np.save(f'./result/{save_dir}/ratio_energy_{name}' , result_ratio_energy)