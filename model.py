# Purpose:  In this program, define the cell model.
#
# Author:   K.Koyama
#
# Function:
#   - HRmodel
#           Hindmarsh-Rose Model
#   - Chaymodel
#           Chay Model
#   - Rizmodel
#            Riz Model
#   - electrical_coupling_term
#           Electrical coupling term between neurons (limited)



# Import of pyrhon libraries.
import numpy as np
import numba
from numba import jit
import math

@jit(nopython = True)
def HRmodel(variables, params): 
    """ Hindmarsh-Rose Model
    Args:
        variables (np.ndarray 1-dim): 3 variables in HRModel
        t (float): time
        params (np.ndarray 1-dim): 8 parameters in HRModel
    
    Returns:
        deriv (np.ndarray 1-dim): Model output
    """
    
    I, a, b, c, d, epsilon, s, x_0 = params
    x = variables[0]
    y = variables[1]
    z = variables[2] 

    deriv_0 = y - a * x**3 + b * x**2 + I - z
    deriv_1 = c - d * x**2 - y 
    deriv_2 = epsilon * (s * (x - x_0) - z)

    return(np.array([deriv_0, deriv_1, deriv_2]))


@jit(nopython = True)
def Chaymodel(variables, params):
    """ Chay Model
    Args:
        variables (np.ndarray 1-dim): 3 variables in ChayModel
        t (float): time
        params (np.ndarray 1-dim): 10 parameters in ChayModel
    
    Returns:
        deriv (np.ndarray 1-dim): Model output
    """

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

    deriv_0 = g_I_star*(m_inf**3)*h_inf*(V_I-V) + g_KV_star*(n**4)*(V_K-V) \
            + g_KC_star*(C/(1+C))*(V_K-V) + g_L_star*(V_L-V) + ipt
    deriv_1 = (n_inf-n)/tau_n
    deriv_2 = rho*((m_inf**3)*h_inf*(V_C-V)-k_C*C)

    return (np.array([deriv_0, deriv_1, deriv_2]))


##### Riz model #####
##### Activation Variavles and Steady-State Voltage-Dependent Activation (InActivation) ################
@jit(nopython = True)
def cal_m_inf(V, V_mx, n_mx):
    return 1/(1+math.exp((V-V_mx)/n_mx))

@jit(nopython = True)
def cal_h_inf(V, V_hx, n_hx):
    return 1/(1+math.exp((V-V_hx)/n_hx))

@jit(nopython = True)
def deriv_m(mx, mx_inf, tau_mx):
    return (mx_inf - mx)/tau_mx

@jit(nopython = True)
def deriv_h(hx, hx_inf, tau_hx):
    return (hx_inf - hx)/tau_hx


@jit(nopython = True)
def Rizmodel(variables, conductance):
    
    ##### Variables #############################
    V    = variables[0]
    Ca_m = variables[1]
    Ca_c = variables[2]
    
    G6PF6P  = variables[3]
    FBP     = variables[4]
    DHAPG3P = variables[5]
    
    a = variables[6]
    
    m_BK = variables[7]
    m_KV = variables[8]
    m_HERG = variables[9]
    h_HERG = variables[10]
    h_Na   = variables[11]
    h_CaL  = variables[12]
    h_CaT  = variables[13]

    ##### Default model Parameters ################################

    # Voltage
    V_K  = -75
    V_Na = 70 
    V_Ca = 65
    V_Cl = -40

    # SK
    g_SK = 0.1
    K_SK = 0.57
    n    = 5.2

    # BK
    g_BK = 0.02
    tau_m_BK = 2
    V_m_BK   = 0
    n_m_BK   = -10    
    B_BK = 20

    # KV
    g_KV = 1.0
    tau_m_KV_0 = 2.0
    if V >= 26.6: # eq.(31)
        tau_m_KV = tau_m_KV_0 + 10*math.exp((-20-V)/6)
    else:
        tau_m_KV = tau_m_KV_0 + 30
    V_m_KV   = 0
    n_m_KV   = -10

    # HERG
    g_HERG = 0.0
    V_m_HERG   = -30
    n_m_HERG   = -10
    V_h_HERG   = -42
    n_h_HERG   = 17.5
    tau_m_HERG = 100
    tau_h_HERG = 50

    # Na
    g_Na   = 0.4
    tau_h_Na = 2
    V_m_Na   = -18
    n_m_Na   = -5
    V_h_Na   = -42
    n_h_Na   = 6

    # CaL
    g_CaL   = 0.14
    tau_h_CaL = 20
    V_m_CaL = -25
    n_m_CaL = -6

    # CaPQ
    g_CaPQ   = 0.17
    V_m_CaPQ = -10
    n_m_CaPQ = -6

    # CaT
    g_CaT     = 0.05
    tau_h_CaT = 7
    V_m_CaT   = -40
    n_m_CaT   = -4
    V_h_CaT   = -64
    n_h_CaT   = 8

    # ATP
    g_K_ATP_current = 0.01
    g_GABAR = 0

    # Leak
    g_leak = 0.015
    V_leak = -30

    # Ca2+ -fluxes
    J_SERCA_max = 0.06 #0.15
    K_SERCA = 0.27
    J_PMCA_max = 0.021
    K_PMCA = 0.5
    J_leak  = 0.00094
    J_NCX_0  = 0.01867

    # (2),(3)
    f = 0.01 
    B = 0.1
    C_m = 10 # パラメータ表にないため，本文中のParameterを採用
    Vol_c = 1.15*(10**(-12)) #※
    Vol_m = 0.1*(10**(-12))  #※
    alpha = 5.18*(10**(-15))

    # GK
    V_GK_max = 0.0000556
    K_GK = 8
    hGK  = 1.7
    G    = 10#10

    # PFK
    V_PFK_max = 0.000556
    K_PFK   = 4.0
    h_PFK = 2.5
    h_act = 1.0
    X_PFK   = 0.01
    h_X   = 2.5
    alpha_G = 5.0

    # FBA
    V_FBA_max = 0.000139
    K_FBA = 0.005
    P_FBA = 0.5
    Q_FBA = 0.275

    # GADPH
    V_GADPH_max = 0.00139
    K_GADPH = 0.005

    # GPI
    K_GPI = 0.3

    # TPI
    K_TPI = 0.045455

    # KATP
    k_A = 0.0001
    g_KATP_hat = 0.05
    
    #### ★ Change Parameters ★ ############

    ### コンダクタンス ###
    #g_SK, g_BK, g_KV, g_K_ATP_current, g_HERG, g_NA, g_CaL, g_CaPQ, g_CaT, g_leak, n_m_CaPQ = change_conductance(conductance)
    ipt, g_SK, g_BK, g_KV, g_K_ATP_current, g_HERG, g_NA, g_CaL, g_CaPQ, g_CaT, g_leak, n_m_CaPQ = conductance

    ##### Current ######################
    
    # 固定しないケース
    #g_KATP_current  = g_KATP_hat/(1+a)

    I_CaL  = g_CaL*cal_m_inf(V, V_m_CaL, n_m_CaL)*h_CaL*(V-V_Ca)
    I_CaPQ = g_CaPQ*cal_m_inf(V, V_m_CaPQ, n_m_CaPQ)*(V-V_Ca)
    I_CaT  = g_CaT*cal_m_inf(V, V_m_CaT, n_m_CaT)*h_CaT*(V-V_Ca)
    I_Ca = I_CaL + I_CaPQ + I_CaT

    I_SK = g_SK*(Ca_m**n)/(K_SK**n + Ca_m**n)*(V-V_K)
    I_BK = g_BK*m_BK*(-1*I_Ca + B_BK)*(V-V_K) # どのタイミングで？？？
    I_KV = g_KV*m_KV*(V-V_K)
    I_HERG = g_HERG*m_HERG*h_HERG*(V - V_K)
    I_KATP = g_K_ATP_current*(V-V_K)
    I_Na = g_Na*cal_m_inf(V, V_m_Na, n_m_Na)*h_Na*(V-V_Na)
    I_leak  = g_leak*(V-V_leak)
    I_GABAR = g_GABAR*(V-V_Cl)

    ##### Ca2+ -flues ##############

    J_SERCA = J_SERCA_max*(Ca_c**2)/((K_SERCA**2)+(Ca_c**2))
    J_PMCA = J_PMCA_max*Ca_m/(K_PMCA+Ca_m)
    J_NCX    = J_NCX_0*Ca_m

    ##### Glycolysis ###############
    
    # (44)-(47)
    F6P   = G6PF6P*K_GPI/(1+K_GPI)
    G3P   = DHAPG3P*K_TPI/(1+K_TPI)
    DHAP  = DHAPG3P - G3P
    h_FBP = h_PFK - ((h_PFK-h_act)*FBP/(K_FBA+FBP))

    ## (39) ※ 微分方程式に影響なし
    g_KATP  = g_KATP_hat/(1+a)

    # V_GK (40)
    V_GK = V_GK_max*(G**hGK)/(K_GK**hGK + G**hGK)
    # V_PFK (41)
    V_PFK_numerator1   = (F6P/K_PFK)**h_FBP
    V_PFK_denominator1 = V_PFK_numerator1 
    V_PFK_denominator2 = (1+(FBP/X_PFK)**h_X)/((1+(FBP/X_PFK)**h_X)*alpha_G**h_FBP) 
    V_PFK = V_PFK_max*V_PFK_numerator1/(V_PFK_denominator1 + V_PFK_denominator2)
    # V_FBA (42)
    V_FBA_fraction1 = FBP/K_FBA
    V_FBA_fraction2 = DHAP/Q_FBA
    V_FBA_fraction3 = G3P/P_FBA
    V_FBA = V_FBA_max*(V_FBA_fraction1 - V_FBA_fraction2*V_FBA_fraction3/K_FBA)\
            /(1 + V_FBA_fraction1 + V_FBA_fraction2 + V_FBA_fraction2*V_FBA_fraction3)
    # V_GADPH (43)
    V_GAPDH = V_GADPH_max*G3P/(K_GADPH + G3P)

    ##### Steady-State Voltage-Dependent Activation(Inactivation) #####
    m_BK_inf   = cal_m_inf(V, V_m_BK, n_m_BK)
    m_KV_inf   = cal_m_inf(V, V_m_KV, n_m_KV)
    m_HERG_inf = cal_m_inf(V, V_m_HERG, n_m_HERG)
    h_HERG_inf = cal_h_inf(V, V_h_HERG, n_h_HERG)
    h_Na_inf   = cal_h_inf(V, V_h_Na, n_h_Na)
    m_CaL_inf  = cal_m_inf(V, V_m_CaL, n_m_CaL)
    h_CaL_inf  = max(0,min(1,1 + (m_CaL_inf*(V-V_Ca))/57)) # eq.(30)
    h_CaT_inf  = cal_h_inf(V, V_h_CaT, n_h_CaT)
    
    ##### differential equation #######################

    # main 3 Equations
    deriv_V = -(I_SK + I_BK + I_KV + I_HERG + I_KATP + I_Na + I_CaL +I_CaPQ +I_CaT + I_leak +I_GABAR) + ipt
    deriv_Ca_m = f*alpha*C_m*((-I_CaL-I_CaPQ-I_CaT))/Vol_m - f*(Vol_c/Vol_m)*(B*(Ca_m-Ca_c)+(J_PMCA+J_NCX))
    deriv_Ca_c = f*(B*(Ca_m-Ca_c)-J_SERCA + J_leak)

    # Glycolysis
    deriv_G6PF6P  = V_GK -V_PFK
    deriv_FBP     = V_PFK - V_FBA
    deriv_DHAPG3P = 2*V_FBA - V_GAPDH

    # ATP-mimetic, K(ATP)-channels
    deriv_a = V_GAPDH - k_A*a
    # Activation Variables
    deriv_m_BK   = deriv_m(m_BK, m_BK_inf, tau_m_BK)
    deriv_m_KV   = deriv_m(m_KV, m_KV_inf, tau_m_KV)
    deriv_m_HERG = deriv_m(m_HERG, m_HERG_inf, tau_m_HERG)
    deriv_h_HERG = deriv_h(h_HERG, h_HERG_inf, tau_h_HERG)
    deriv_h_Na   = deriv_h(h_Na, h_Na_inf, tau_h_Na)
    deriv_h_CaL  = deriv_h(h_CaL, h_CaL_inf, tau_h_CaL)
    deriv_h_CaT  = deriv_h(h_CaT, h_CaT_inf, tau_h_CaT)

    return (np.array([deriv_V, deriv_Ca_m, deriv_Ca_c, deriv_G6PF6P, deriv_FBP, deriv_DHAPG3P, deriv_a, \
                      deriv_m_BK, deriv_m_KV, deriv_m_HERG, deriv_h_HERG, deriv_h_Na, deriv_h_CaL,deriv_h_CaT]))



@jit(nopython=True)
def electrical_coupling_term(i, cellsout, k, full_connected=1):
    """Electrical Coupling
    Args:
        i (int): cell No.
        cellsout (np.ndarray 2-dim): Output of cells
        k (float): Coupling Strngth

    Returns:
        K_i (float): electrical coupling term
    """
    N = cellsout.shape[0]
    K_i = 0 # Coupling Term
    r_i = 0 # number of cells coupled cell i

    if full_connected == 1:
        r_i = N-1
        for j in range(N):
            K_i += k*(cellsout[j][0]-cellsout[i][0])
        K_i = K_i/r_i
    else:
        #Implement as needed.
        pass
    
    return K_i