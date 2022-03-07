# Purpose:  In this program, define functions to output the heatmap.
#           Figures in the paper are available at "ShowHeatmap.ipynb".            
#
# Author:   K.Koyama
#
# Function:
#   - create_df_fixed_l
#           Load simulation results (.npy) and convert to DataFrame.
#   - periodic_InputEnergy
#           Extract input energy in the periodic regions.
#   - plot_heatmap
#           Draw heatmap using DataFrame.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['mathtext.fontset'] = 'stix' 
plt.rcParams["font.size"] = 15 
plt.rcParams['xtick.labelsize'] = 15 
plt.rcParams['ytick.labelsize'] = 15 
plt.rcParams['xtick.direction'] = 'in' 
plt.rcParams['ytick.direction'] = 'in'


def create_df_fixed_l(save_dir, save_name, idx, ax0_array, ax2_array):

    ns_array = np.load(f"./result/{save_dir}/num_spike_{save_name}.npy")
    nb_array = np.load(f"./result/{save_dir}/num_burst_{save_name}.npy")
    cv_array = np.load(f"./result/{save_dir}/cv_{save_name}.npy")
    std_array = np.load(f"./result/{save_dir}/std_ibis_{save_name}.npy")

    
    df_ns = pd.DataFrame(ns_array[:,idx,:], index=ax0_array, columns=ax2_array)
    df_nb = pd.DataFrame(nb_array[:,idx,:], index=ax0_array, columns=ax2_array)
    df_cv = pd.DataFrame(cv_array[:,idx,:], index=ax0_array, columns=ax2_array)
    df_std = pd.DataFrame(std_array[:,idx,:], index=ax0_array, columns=ax2_array)

    try: # Chay
        ie_array = np.load(f"./result/{save_dir}/input_energy_{save_name}.npy")
        oe_array = np.load(f"./result/{save_dir}/output_energy_{save_name}.npy")
        ratio_array = np.load(f"./result/{save_dir}/ratio_energy_{save_name}.npy")
        df_ie = pd.DataFrame(ie_array[:,idx,:], index=ax0_array, columns=ax2_array)
        df_oe = pd.DataFrame(oe_array[:,idx,:], index=ax0_array, columns=ax2_array)
        df_ratio = pd.DataFrame(ratio_array[:,idx,:], index=ax0_array, columns=ax2_array)
    except: # Riz
        df_ie = pd.DataFrame()
        df_oe = pd.DataFrame()
        df_ratio = pd.DataFrame()
    

    return df_ns, df_nb, df_cv, df_std, df_ie, df_oe, df_ratio


def periodic_InputEnergy(df_cv, df_ie):

    f_period = lambda x: 0 if x<0.01 else -10000
    f_select = lambda x: x if x > 0 else 0
    f_0 = lambda x: x if x!=0 else 100000 
    
    df_cv = df_cv.applymap(f_period)
    df_ie = df_ie + df_cv
    df_ie = df_ie.applymap(f_select)
    df_ie = df_ie.applymap(f_0)
    df = df_ie.copy()
    
    return df

def plot_heatmap(df, title, bar_label, typ, v, xtick=2):

    """
    Args:
        df: 描画用DataFrame
        title
        bar_label
        typ：magmaの順序選択
        v: list [v_min, v_max]
        xtick
    """
    
    plt.figure(facecolor="white")
    nb_max = max(df.max())
    # if v == 0:
    #     if typ == 1: ax = sns.heatmap(df, cmap = "magma", xticklabels=1, yticklabels=1)
    #     if typ == 2: ax = sns.heatmap(df, cmap = "magma_r", xticklabels=1, yticklabels=1)   
    # elif v == 1:
    #     if typ == 1: ax = sns.heatmap(df, vmin=v_min, vmax=v_max, cmap = "magma", xticklabels=1, yticklabels=1)
    #     if typ == 2: ax = sns.heatmap(df, vmin=v_min, vmax=v_max, cmap = "magma_r", xticklabels=1, yticklabels=1)

    if v:
        if typ == 1: ax = sns.heatmap(df, vmin=v[0], vmax=v[1], cmap = "magma", xticklabels=1, yticklabels=1)
        if typ == 2: ax = sns.heatmap(df, vmin=v[0], vmax=v[1], cmap = "magma_r", xticklabels=1, yticklabels=1)
    else:
        if typ == 1: ax = sns.heatmap(df, cmap = "magma", xticklabels=1, yticklabels=1)
        if typ == 2: ax = sns.heatmap(df, cmap = "magma_r", xticklabels=1, yticklabels=1)    
    
    ax.collections[0].colorbar.set_label(bar_label)
    x_labels = list(ax.get_xticklabels())
    y_labels = list(ax.get_yticklabels())
    x_labels_ex = [] 
    y_labels_ex = []
    
    for idx, x in enumerate(x_labels):
        if idx%((len(x_labels)-1)/xtick) == 0:
            x_labels_ex.append(x)
        else:
            x_labels_ex.append("")
    for idx, y in enumerate(y_labels):
        if idx%((len(y_labels)-1)/2) == 0:
            y_labels_ex.append(y)
        else:
            y_labels_ex.append("")

    ax.set_xticklabels(x_labels_ex, rotation=0)
    ax.set_yticklabels(y_labels_ex)
    ax.tick_params(length=0, which='major')
    ax.invert_yaxis()

    # fixed "l"
    plt.xlabel('$V_{th}$')
    plt.ylabel('$a$')
    
    plt.title(title, fontsize = 18)           
    plt.show()



# Example of heatmap output
if __name__ =="__main__":
    
    name = 'l0p25_allregion'
    save_dir = 'chaotic_chay'

    ax0 = np.array([round(0 + 0.1*i,1) for i in range(101)]) # a
    ax1 = np.array([0.25]) # l
    ax2 = np.array([round(-47.5 +0.02*i,2) for i in range(226)]) # V_th

    df_ns, _, df_cv, _, df_ie, _, _ = create_df_fixed_l(save_dir, name, 0, ax0, ax2)

    df_periodicie  = periodic_InputEnergy(df_cv, df_ie)
    epsilon = 0.1
    df_ais = df_ns/(df_cv+epsilon)
    
    # plot_heatmap(df_cv, "", "CV", typ=2, v=[0,0.2])
    # plot_heatmap(df_periodicie, "", "Input Energy", typ=1, v=[0,300])
    plot_heatmap(df_ais, "", "Ability of Insulin Secretion", typ=1, v=[1300,3400])