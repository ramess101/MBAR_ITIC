from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
import CoolProp
from scipy.optimize import minimize
     
# Simulation constants

# Gromacs uses a different format
    
N_sites = 1
N_pair = N_sites**2
N_columns = N_pair * 2

# RDF bins

r_c = 1.4 #[nm]

r = np.linspace(0.002,1.4,num=700) #[nm]

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

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt","--optimizer",type=str,choices=['fsolve','steep','LBFGSB','leapfrog','scan','points','SLSQP'],help="choose which type of optimizer to use")
    parser.add_argument("-prop","--properties",type=str,nargs='+',choices=['rhoL','Psat','rhov','P','U','Z'],help="choose one or more properties to use in optimization" )
    args = parser.parse_args()
    if args.optimizer:
        eps_opt, sig_opt, lam_opt = call_optimizers(args.optimizer,args.properties)
    else:
        print('Please specify an optimizer type')
        eps_opt = 0.
        sig_opt = 0.
        lam_opt = 0.

if __name__ == '__main__':
    
    main()