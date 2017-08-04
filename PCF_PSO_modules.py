from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
import CoolProp
from scipy.optimize import minimize

CP = CoolProp.CoolProp

"""
    This code performs a post-simulation optimization using the pair correlation function
"""
     
compound='Ethane'

Temp = np.array([178, 197, 217, 236, 256, 275, 110, 135, 160, 290]) #[K]
rho_v = np.array([0.02915, 0.07315, 0.16168, 0.30783, 0.56606, 0.98726, 4.89203E-05, 0.001216635, 0.0097938, 1.54641093]) #[1/nm**3]
rho_L = np.array([11.06996, 10.57738, 10.01629, 9.42727, 8.71778, 7.89483, 12.61697633, 12.06116887, 11.48692334, 7.036023817]) #[1/nm**3]

V_v = 1./rho_v #[nm**3]
V_L = 1./rho_L #[nm**3]

T_c_RP = CP.PropsSI('TCRIT','REFPROP::'+compound)
rho_c_RP = CP.PropsSI('RHOCRIT','REFPROP::'+compound)
M_w = CP.PropsSI('M','REFPROP::'+compound) #[kg/mol]
T_low = CP.PropsSI('TMIN','REFPROP::'+compound)
RP_rho_L = CP.PropsSI('D','T',Temp,'Q',0,'REFPROP::'+compound) #[kg/m3]
RP_rho_v = CP.PropsSI('D','T',Temp,'Q',1,'REFPROP::'+compound) #[kg/m3]
RP_P_v = CP.PropsSI('P','T',Temp,'Q',1,'REFPROP::'+compound)/1000 #[kPa]
RP_H_L = CP.PropsSI('HMOLAR','T',Temp,'Q',0,'REFPROP::'+compound) / 1000 #[kJ/mol]
RP_H_v = CP.PropsSI('HMOLAR','T',Temp,'Q',1,'REFPROP::'+compound) / 1000 #[kJ/mol]
RP_HVP = RP_H_v - RP_H_L #[kJ/mol]
RP_V_L = M_w/RP_rho_L #[m3/mol]
RP_V_v = M_w/RP_rho_v #[m3/mol]
RP_deltaV = RP_V_v - RP_V_L #[m3/mol]
RP_deltaU = RP_HVP - RP_P_v*RP_deltaV #[kJ/mol]
RP_U_L = CP.PropsSI('UMOLAR','T',Temp,'Q',0,'REFPROP::'+compound) / 1000 #[kJ/mol]
RP_U_v = CP.PropsSI('UMOLAR','T',Temp,'Q',1,'REFPROP::'+compound) / 1000 #[kJ/mol] 
RP_deltaU_alt = RP_U_v - RP_U_L
RP_U_ig = CP.PropsSI('UMOLAR','T',Temp,'D',0,'REFPROP::'+compound) / 1000 #[kJ/mol]
RP_U_L_res = RP_U_L - RP_U_ig
RP_U_v_res = RP_U_v - RP_U_ig

T_YE_tab = np.array([170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280])
YE_U_L_tab = np.array([-9.305, -8.587, -7.86, -7.121, -6.367, -5.596, -4.805, -3.988, -3.14, -2.254, -1.316, -0.3061]) #Alternative source
YE_U_v_tab = np.array([4.554, 4.839, 5.117 ,5.388 ,5.647 ,5.893 ,6.12 ,6.324 ,6.498 ,6.632, 6.709,6.7])

Temp_sorted = np.sort(Temp)

YE_U_L_fit = np.polyval(np.polyfit(T_YE_tab,YE_U_L_tab,2),Temp_sorted)
YE_U_v_fit = np.polyval(np.polyfit(T_YE_tab,YE_U_v_tab,2),Temp_sorted)
YE_deltaU_fit = YE_U_v_fit - YE_U_L_fit

#RP_U_L = YE_U_L_fit
#RP_U_v = YE_U_v_fit

#RP_deltaU = YE_deltaU_fit

U_intra = np.array([5.506943842, 5.878684617, 6.289459224, 6.693660331, 7.130462861, 7.553929439, 4.467419639, 4.772479078, 5.177295002, 7.893015148])

R_g = 8.314472e-3 # [kJ/mol/K]

N = 400 # Number of molecules
N_sites = 2
N_pair = N_sites**2

reference='TraPPE'

if reference == 'TraPPE':

    # TraPPE
    eps_ref = 98. #[K]
    sig_ref = 0.375 #[nm]
    lam_ref = 12;
    
    fname = 'H:/PCF-PSO/RDF_TraPPE_all_temps.txt'
    RDFs_ref = np.loadtxt(fname,delimiter='\t')
    
    deltaU_ens = np.array([12.26633, 11.50548, 10.63125, 9.6665, 8.465, 6.98475, 14.68024786, 13.79634868, 12.91341433, 5.421409895]) # [kJ/mol] Ensemble averages for TraPPE
    U_L_ens = np.array([-12.3225, -11.6325, -10.89, -10.13, -9.2675, -8.3125, -14.68044749, -13.79952918, -12.93440837, -7.39177347]) # [kJ/mol] Ensemble averages for TraPPE
    U_v_ens = np.array([-0.056175, -0.127025,-0.25875,-0.4635,-0.8025,-1.32775, -0.000199626, -0.003180493, -0.020994042,-1.970363575]) # [kJ/mol] Ensemble averages for TraPPE

elif reference == 'Potoff':
                    
    # Potoff
    eps_ref = 121.25 #[K]
    sig_ref = 0.3783 #[nm]
    lam_ref = 16
    
    fname = 'H:/PCF-PSO/RDF_Potoff.txt'
    RDFs_ref = np.loadtxt(fname,delimiter='\t')   
    
    deltaU_ens = np.array([13.47817937, 12.60056654, 11.58723646, 10.49408513, 9.136582368, 7.504864688]) # [kJ/mol] Ensemble averages for Potoff

elif reference == 'Iteration_3':                     
                     
    # Iteration_3
    eps_ref = 117.426 #[K]
    sig_ref = 0.377629 #[nm]
    lam_ref = 15.0561
    
    fname = 'H:/PCF-PSO/RDF_PCF_PSO_Iteration_3_all_Temps.txt'
    RDFs_ref = np.loadtxt(fname,delimiter='\t')   
    
    # Wrong tail corrections
    #deltaU_ens = np.array([13.30261018, 12.44742398, 11.45416561, 10.37973227, 9.04093736, 7.430165499, 16.04817995, 15.04421521, 14.03175751, 5.703436554]) # [kJ/mol] Ensemble averages for Iteration 3
    # Correct tail corrections
    deltaU_ens = np.array([13.565469, 12.69750703, 11.68878262, 10.59684646, 9.235012225, 7.594619838, 16.34856183, 15.3313366, 14.30500321, 5.834132324]) # [kJ/mol] Ensemble averages for Iteration 3
    U_L_ens = np.array([-13.63266466, -12.84714095, -11.98980705, -11.12923475, -10.14797751 ,-9.09372066 ,-16.34856315 ,-15.33136885 ,-14.33118704 ,-8.071015752]) # [kJ/mol] Ensemble averages for TraPPE
    U_v_ens = np.array([-0.067195657, -0.149633913, -0.301024437, -0.532388298, -0.91296529, -1.499100823,-0.000252822,-0.004209754,-0.026183824,-2.236883428]) # [kJ/mol] Ensemble averages for TraPPE

elif reference == 'Iteration_4':                                          
                     
    # Iteration_4
    eps_ref = 119.6115 #[K]
    sig_ref = 0.377761 #[nm]
    lam_ref = 15.091
        
    fname = 'H:/PCF-PSO/RDF_PCF_PSO_Iteration_4.txt'
    RDFs_ref = np.loadtxt(fname,delimiter='\t')   
    
    deltaU_ens = np.array([13.58024314, 12.70553103, 11.6871413, 10.58667247, 9.225330852, 7.573248935]) # [kJ/mol] Ensemble averages for Iteration 4                  

elif reference == 'Iteration_5_Int':                                          
                     
    # Iteration_5_Int
    eps_ref = 118.935 #[K]
    sig_ref = 0.377525 #[nm]
    lam_ref = 15.
        
    fname = 'H:/PCF-PSO/RDF_Iteration_5_Int_all_Temps.txt'
    RDFs_ref = np.loadtxt(fname,delimiter='\t')   
    
    deltaU_ens = np.array([13.58024314, 12.70553103, 11.6871413, 10.58667247, 9.225330852, 7.573248935, 16.27853971, 15.27048735, 14.24386152, 5.806201543]) # [kJ/mol] Ensemble averages for Iteration 4                  

      
def r_min_calc(sig, n=12., m=6.):
    r_min = (n/m*sig**(n-m))**(1./(n-m))
    return r_min
                      
r_min_ref = r_min_calc(sig_ref,lam_ref)
r_avg_ref = (sig_ref + r_min_ref)/2
                      
bond_length = 0.154 #[nm]

# Simulation constants
r_c = 1.4 #[nm]

r = np.linspace(0.014,1.386,num=50) #[nm]

dr = r[1] - r[0]

r_c_plus_ref = r_c / sig_ref

r_plus_ref = r/sig_ref

dr_plus_ref = dr/sig_ref

def U_Mie(r, e_over_k, sigma, n = 12., m = 6.):
    """
    The Mie potential calculated in [K]. 
    Note that r and sigma must be in the same units. 
    The exponents (n and m) are set to default values of 12 and 6    
    """
    C = (n/(n-m))*(n/m)**(m/(n-m)) # The normalization constant for the Mie potential
    U = C*e_over_k*((r/sigma)**-n - (r/sigma)**-m)
    return U

def U_Corr(e_over_k, sigma, r_c_plus, n = 12., m = 6.): #I need to correct this for m != 6
    C = (n/(n-m))*(n/m)**(m/(n-m)) # The normalization constant for the Mie potential
    U = C*e_over_k*((1./(n-3))*r_c_plus**(3-n) - (1./3) * r_c_plus **(-3))*sigma**3 #[K * nm^3]
    return U

def U_total(r, e_over_k, sigma, r_c_plus, RDF, dr,  n = 12., m = 6.):
    U_int = (U_Mie(r, e_over_k, sigma,n,m)*RDF*r**2*dr).sum() # [K*nm^3]
    U_total = U_int + U_Corr(e_over_k,sigma,r_c_plus,n,m)
    U_total *= 2*math.pi
    return U_total

def r_min_calc(sig, n=12., m=6.):
    r_min = (n/m*sig**(n-m))**(1./(n-m))
    return r_min

def U_hat(eps_pred,sig_pred,lam_pred,RDF_Temp,r_ref=r,sig_ref=sig_ref,r_c=r_c,r_plus_ref = r_plus_ref, dr_plus_ref = dr_plus_ref, r_c_plus_ref = r_c_plus_ref):
    
    U_hat = 0 
                      
    for n in range(0,N_pair):
        
        RDF = RDF_Temp[:,n]
                
        U_hat += U_total(r_plus_ref*sig_pred,eps_pred,sig_pred,r_c_plus_ref,RDF,dr_plus_ref*sig_pred,lam_pred) # Constant r_plus
    return U_hat
    
    
def deltaU(eps,sig,lam,RDF_all=RDFs_ref,rho_v=rho_v,rho_L=rho_L,Temp=Temp):
    
    U_L = np.empty(len(Temp))
    U_v = U_L.copy()
    
    for t in range(0, len(Temp)):
        rhov_Temp = rho_v[t]
        rhoL_Temp = rho_L[t]
        
        RDF_Temp_L = RDF_all[:,8*t:8*t+N_pair]
        RDF_Temp_v = RDF_all[:,8*t+N_pair:8*t+2*N_pair]
        
        U_L_Temp = U_hat(eps,sig,lam,RDF_Temp_L)
        U_v_Temp = U_hat(eps,sig,lam,RDF_Temp_v)
        
        U_L_Temp *= R_g * rhoL_Temp
        U_v_Temp *= R_g * rhov_Temp
        
        U_L[t] = U_L_Temp
        U_v[t] = U_v_Temp
           
    deltaU = U_v - U_L
    return deltaU, U_L, U_v

deltaU_error = deltaU_ens - deltaU(eps_ref,sig_ref,lam_ref)[0]
U_L_error = U_L_ens - deltaU(eps_ref,sig_ref,lam_ref)[1]
U_v_error = U_v_ens - deltaU(eps_ref,sig_ref,lam_ref)[2]

def deltaU_hat(eps,sig,lam):
        
    deltaU_hat = deltaU(eps,sig,lam)[0] + deltaU_error
    return deltaU_hat

def U_L_hat(eps,sig,lam):
            
    U_L_hat = deltaU(eps,sig,lam)[1] + U_L_error #Easier to just compare to residual
    #U_L_hat = deltaU(eps,sig,lam)[1] + U_intra + U_L_error
    return U_L_hat

def U_v_hat(eps,sig,lam):
            
    U_v_hat = deltaU(eps,sig,lam)[2] + U_v_error # Easier to just compare to residual
    #U_v_hat = deltaU(eps,sig,lam)[2] + U_intra + U_v_error
    return U_v_hat

def objective(params):
    eps = params[0]
    sig = params[1]
    lam = params[2]
    dev_deltaU = deltaU_hat(eps,sig,lam) - RP_deltaU
    dev_U_L = U_L_hat(eps,sig,lam) - RP_U_L_res
    dev_U_v = U_v_hat(eps,sig,lam) - RP_U_v_res
    dev_dev_U = dev_U_L - dev_U_v
    #dev_deltaU = dev_deltaU[Temp>110]
    #dev_deltaU = dev_deltaU[0:-1]
    SSE_deltaU = np.sum(np.power(dev_deltaU,2))
    SSE_U_L = np.sum(np.power(dev_U_L,2))
    SSE_U_v = np.sum(np.power(dev_U_v,2))
    SSE_dev_dev = np.sum(np.power(dev_dev_U,2))
    SSE = 0
    SSE += SSE_deltaU #+ SSE_dev_dev
    #SSE += SSE_U_L + SSE_U_v
    #SSE += SSE_U_L
    print('eps = '+str(eps)+', sig = '+str(sig)+', lam = '+str(lam)+' SSE ='+str(SSE))
    return SSE

def constraint1(params):
    sig = params[1]
    lam = params[2]
    if lam < lam_ref:
        return sig_ref - sig
    elif lam >= lam_ref:
        return sig - sig_ref
    
def constraint2(params):
    sig = params[1]
    lam = params[2]
    if lam < lam_ref:
        return r_min_calc(sig,lam) - r_min_ref
    elif lam >= lam_ref:
        return r_min_ref - r_min_calc(sig,lam)

sig_TraPPE = 0.375 #[nm]
lam_TraPPE = 12
r_min_TraPPE = r_min_calc(sig_TraPPE,lam_TraPPE)

def constraint3(params):
    sig = params[1]
    lam = params[2]
    if lam < lam_TraPPE:
        return sig_TraPPE - sig
    elif lam >= lam_TraPPE:
        return sig - sig_TraPPE
    
def constraint4(params):
    sig = params[1]
    lam = params[2]
    if lam < lam_TraPPE:
        return r_min_calc(sig,lam) - r_min_TraPPE
    elif lam >= lam_TraPPE:
        return r_min_TraPPE - r_min_calc(sig,lam)
    
lam_fixed = 15    

def constraint5(params):
    lam = params[2]
    return lam - lam_fixed

def constraint6(params):
    lam = params[2]
    return lam % 2
    
guess = [eps_ref, sig_ref, 14.]
con1 = {'type':'ineq','fun':constraint1}
con2 = {'type':'ineq','fun':constraint2}
con3 = {'type':'ineq','fun':constraint3}
con4 = {'type':'ineq','fun':constraint4}
con5 = {'type':'eq','fun':constraint5}
con6 = {'type':'eq','fun':constraint6}
cons = [con3,con4]
#cons = []
bnds = ((50,200),(0.3,0.45),(10,20))

sol = minimize(objective,guess,method='SLSQP',bounds=bnds,constraints=cons)#,options={'eps':1e-6})
params_opt = sol.x
eps_opt = sol.x[0]
sig_opt = sol.x[1]
lam_opt = sol.x[2]
r_min_opt = r_min_calc(sig_opt,lam_opt)

print(params_opt)

#Taken from optimization_Mie_ITIC
guess_scaled = [1.,1.,1.]
objective_scaled = lambda params_scaled: objective(params_scaled*guess)
bnds_scaled = ((50/eps_ref,200/eps_ref),(0.3/sig_ref,0.45/sig_ref),(10./14.,20./14.)) 
sol = minimize(objective_scaled,guess_scaled,method='SLSQP',bounds=bnds_scaled,options={'eps':1e-5,'maxiter':50}) #'eps' accounts for the algorithm wanting to take too small of a step change for the Jacobian that Gromacs does not distinguish between the different force fields
params_scaled_opt = sol.x
eps_opt = sol.x[0]*eps_ref
sig_opt = sol.x[1]*sig_ref
lam_opt = sol.x[2]*14.

eps_It_1 = 122.94636444
sig_It_1 = 0.37620538
lam_It_1 = 14.59923016

#print(deltaU_hat(eps_ref,sig_ref,lam_ref))
#print(deltaU(eps_ref,sig_ref,lam_ref))

r_plot = np.linspace(0.8*sig_ref,2*sig_ref,num=1000)

Mie_ref = U_Mie(r_plot,eps_ref,sig_ref,lam_ref)
Mie_opt = U_Mie(r_plot,eps_opt,sig_opt,lam_opt)
Mie_Potoff = U_Mie(r_plot,121.25,0.3783,16)

plt.plot(r_plot,Mie_ref,label='Initial')
plt.plot(r_plot,Mie_opt,label='Optimal')
plt.plot(r_plot,Mie_Potoff,label='Potoff')
plt.xlabel('r (nm)')
plt.ylabel('U (K)')
plt.ylim([-1.1*np.max(np.array([eps_ref,eps_opt])),1.1*np.max(np.array([eps_ref,eps_opt]))])
plt.xlim([np.min(r_plot),np.max(r_plot)])
plt.legend()
plt.show()

T_plot = np.linspace(Temp.min(), Temp.max())
RP_HVP_plot = (CP.PropsSI('HMOLAR','T',T_plot,'Q',1,'REFPROP::'+compound) - CP.PropsSI('HMOLAR','T',T_plot,'Q',0,'REFPROP::'+compound)) / 1000 #[kJ/mol]
RP_deltaU_plot = (CP.PropsSI('UMOLAR','T',T_plot,'Q',1,'REFPROP::'+compound) - CP.PropsSI('UMOLAR','T',T_plot,'Q',0,'REFPROP::'+compound)) / 1000 #[kJ/mol]
RP_U_L_plot = CP.PropsSI('UMOLAR','T',T_plot,'Q',0,'REFPROP::'+compound) / 1000 #[kJ/mol]
RP_U_v_plot = CP.PropsSI('UMOLAR','T',T_plot,'Q',1,'REFPROP::'+compound) / 1000 #[kJ/mol]
RP_U_ig_plot = CP.PropsSI('UMOLAR','T',T_plot,'D',0,'REFPROP::'+compound) / 1000 #[kJ/mol]
RP_U_L_res_plot = RP_U_L_plot - RP_U_ig_plot
RP_U_v_res_plot = RP_U_v_plot - RP_U_ig_plot

deltaU_opt = deltaU_hat(eps_opt,sig_opt,lam_opt)
deltaU_It_1 = deltaU_hat(eps_It_1,sig_It_1,lam_It_1)
deltaU_ref = deltaU_hat(eps_ref,sig_ref,lam_ref)
U_L_opt = U_L_hat(eps_opt,sig_opt,lam_opt)
U_L_ref = U_L_hat(eps_ref,sig_ref,lam_ref)
U_v_opt = U_v_hat(eps_opt,sig_opt,lam_opt)
U_v_ref = U_v_hat(eps_ref,sig_ref,lam_ref)

plt.scatter(Temp,deltaU_ref,label='Initial')
plt.scatter(Temp,deltaU_opt,label='Optimal')
plt.plot(T_plot,RP_deltaU_plot,label='RefProp')
#plt.plot(Temp,YE_deltaU_fit,label='YE')
plt.xlabel('T (K)')
plt.ylabel(r'$\Delta U_v \left(\frac{kJ}{mol}\right)$')
plt.legend()
plt.show()

plt.scatter(Temp,U_L_ref,label='Initial')
plt.scatter(Temp,U_L_opt,label='Optimal')
plt.plot(T_plot,RP_U_L_res_plot,label='RefProp')
#plt.plot(Temp,YE_U_L_fit,label='YE')
plt.xlabel('T (K)')
plt.ylabel(r'$U_l \left(\frac{kJ}{mol}\right)$')
plt.legend()
plt.show()

plt.scatter(Temp,U_v_ref,label='Initial')
plt.scatter(Temp,U_v_opt,label='Optimal')
plt.plot(T_plot,RP_U_v_res_plot,label='RefProp')
#plt.plot(Temp,YE_U_v_fit,label='YE')
plt.xlabel('T (K)')
plt.ylabel(r'$U_v \left(\frac{kJ}{mol}\right)$')
plt.legend()
plt.show()