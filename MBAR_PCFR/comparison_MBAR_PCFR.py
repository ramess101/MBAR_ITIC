# -*- coding: utf-8 -*-
"""
Compares the MBAR predictions with direct simulation for U and P

@author: ram9
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize  

fpathroot = 'parameter_space_LJ/'
nReruns = 441
nStates = 19

def get_parameter_sets():
    
    eps_sig_all = np.loadtxt(fpathroot+'eps_sig_lam12_all',skiprows=2)
    
    eps_matrix = np.zeros([nStates,nReruns])
    sig_matrix = np.zeros([nStates,nReruns])
    
    eps_all = eps_sig_all[:,0]
    sig_all = eps_sig_all[:,1]
    
    for iState in range(nStates):
        eps_matrix[iState,:] = eps_all
        sig_matrix[iState,:] = sig_all

    return eps_all, sig_all, eps_matrix, sig_matrix

def compile_data(model_type):
    U_compiled = np.zeros([nStates,nReruns])
    dU_compiled = np.zeros([nStates,nReruns])
    P_compiled = np.zeros([nStates,nReruns])
    dP_compiled = np.zeros([nStates,nReruns])
    Z_compiled = np.zeros([nStates,nReruns])
    Neff_compiled = np.zeros([nStates,nReruns])
    
    for iRerun in range(nReruns):
        iRerun += 1
        if model_type == 'Direct_simulation':
            fpath = fpathroot+model_type+'_rr'+str(iRerun)
            UPZ = np.loadtxt(fpath)            
        else:
            fpath = fpathroot+model_type+'rr'+str(iRerun)+'_lam12'
            UPZ = np.loadtxt(fpath)
            
        U_compiled[:,iRerun-1] = UPZ[:,0]
        dU_compiled[:,iRerun-1] = UPZ[:,1]
        P_compiled[:,iRerun-1] = UPZ[:,2]
        dP_compiled[:,iRerun-1] = UPZ[:,3]
        Z_compiled[:,iRerun-1] = UPZ[:,4]
        Neff_compiled[:,iRerun-1] = UPZ[:,6]
        
    dZ_compiled = dP_compiled * (Z_compiled/P_compiled)
    
    return U_compiled, dU_compiled, P_compiled, dP_compiled, Z_compiled, dZ_compiled, Neff_compiled

def parity_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,Neff,dprop_direct,dprop_hat1):
    parity = np.array([np.min(np.array([np.min(prop_direct),np.min(prop_hat1),np.min(prop_hat2)])),np.max(np.array([np.max(prop_direct),np.max(prop_hat1),np.max(prop_hat2)]))])
    
    if prop == 'U':
        units = '(kJ/mol)'
        title = 'Residual Energy'
    elif prop == 'P':
        units = '(bar)'
        title = 'Pressure'
    elif prop == 'Z':
        units = ''
        title = 'Compressibility Factor'
    elif prop == 'Pdep':
        units = '(bar)'
        title = 'Pressure - Ideal Gas'

    plt.plot(prop_direct,prop_hat3,'b.',label='Constant PCF',alpha=0.2)    
    plt.plot(prop_direct,prop_hat1,'r.',label='MBAR',alpha=0.2)
    plt.plot(prop_direct,prop_hat2,'g.',label='PCFR',alpha=0.2)
    #plt.plot(prop_direct,prop_hat3,'bx',label='MBAR, Neff > 10')
    plt.plot(parity,parity,'k',label='Parity')
    plt.xlabel('Direct Simulation '+units)
    plt.ylabel('Predicted '+units)
    plt.title(title)
    plt.legend()
    plt.show()
    
    #Error bar did not seem useful because the errors are so small
#    plt.errorbar(prop_direct,prop_hat1,xerr=dprop_direct,yerr=dprop_hat1,fmt='ro',label='MBAR')
#    #plt.plot(prop_direct,prop_hat2,'gx',label='PCFR')
#    #plt.plot(prop_direct,prop_hat3,'bx',label='MBAR, Neff > 10')
#    plt.plot(parity,parity,'k',label='Parity')
#    plt.xlabel('Direct Simulation '+units)
#    plt.ylabel('Predicted '+units)
#    plt.title(title)
#    plt.legend()
#    plt.show()
#       
    p = plt.scatter(prop_direct[Neff.argsort()],prop_hat1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='cool',label='MBAR',alpha=0.5)
    plt.plot(parity,parity,'k',label='Parity')
    plt.xlabel('Direct Simulation '+units)
    plt.ylabel('Predicted with MBAR '+units)
    plt.title(title)
    #plt.legend()
    cb = plt.colorbar(p)
    cb.set_label('log$_{10}(N_{eff})$')
    plt.show()
    
def residual_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,Neff,dprop_direct):
    
    abs_dev = lambda hat, direct: (hat - direct)
    rel_dev = lambda hat, direct: abs_dev(hat,direct)/direct * 100.
    rel_dev_dprop = lambda hat, direct: abs_dev(hat,direct)/dprop_direct
                                         
    abs_dev1 = abs_dev(prop_hat1, prop_direct)
    abs_dev2 = abs_dev(prop_hat2,prop_direct)
    abs_dev3 = abs_dev(prop_hat3,prop_direct)
           
    rel_dev1 = rel_dev(prop_hat1,prop_direct)
    rel_dev2 = rel_dev(prop_hat2,prop_direct)
    rel_dev3 = rel_dev(prop_hat3,prop_direct)
    
    rel_dev_dprop1 = rel_dev_dprop(prop_hat1,prop_direct)
    rel_dev_dprop2 = rel_dev_dprop(prop_hat2,prop_direct)
    rel_dev_dprop3 = rel_dev_dprop(prop_hat3,prop_direct)
               
    if prop == 'U':
        units = ''
        title = 'Residual Energy'
        dev1 = rel_dev1
        dev2 = rel_dev2
        dev3 = rel_dev3
        dev_type = 'Percent'
    elif prop == 'P':
        units = '(bar)'
        title = 'Pressure'
        dev1 = abs_dev1
        dev2 = abs_dev2
        dev3 = abs_dev3
        dev_type = 'Absolute'
    elif prop == 'Z':
        units = ''
        title = 'Compressibility Factor'
        dev1 = abs_dev1
        dev2 = abs_dev2
        dev3 = abs_dev3
        dev_type = 'Absolute'
    elif prop == 'Pdep':
        units = '(bar)'
        title = 'Pressure - Ideal Gas'
        dev1 = abs_dev1
        dev2 = abs_dev2
        dev3 = abs_dev3
        dev_type = 'Absolute'

    plt.plot(prop_direct,dev3,'bx',label='Constant PCF',alpha=0.2)        
    plt.plot(prop_direct,dev1,'rx',label='MBAR',alpha=0.2)
    plt.plot(prop_direct,dev2,'gx',label='PCFR',alpha=0.2)
    #plt.plot(prop_direct,dev3,'bx',label='MBAR, Neff > 10')
    plt.xlabel('Direct Simulation '+units)
    plt.ylabel(dev_type+' Deviation '+units)
    plt.title(title)
    plt.legend()
    plt.show()
    
    plt.plot(prop_direct,rel_dev_dprop3,'bx',label='Constant PCF')
    plt.plot(prop_direct,rel_dev_dprop1,'rx',label='MBAR')
    plt.plot(prop_direct,rel_dev_dprop2,'gx',label='PCFR')
    #plt.plot(prop_direct,rel_dev_dprop3,'bx',label='MBAR, Neff > 10')
    plt.xlabel('Direct Simulation '+units)
    plt.ylabel('Coverage Factor')
    plt.title(title)
    plt.legend()
    plt.show()
    
    p = plt.scatter(prop_direct[Neff.argsort()],dev1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='cool',label='MBAR',alpha=0.5)
    plt.xlabel('Direct Simulation '+units)
    plt.ylabel(dev_type+' Deviation '+units)
    plt.title(title)
    #plt.legend()
    cb = plt.colorbar(p)
    cb.set_label('log$_{10}(N_{eff})$')
    plt.show()
    
    p = plt.scatter(prop_direct,rel_dev_dprop1,c=np.log10(Neff),cmap='cool',label='MBAR')
    plt.xlabel('Direct Simulation '+units)
    plt.ylabel('Coverage Factor')
    plt.title(title)
    #plt.legend()
    cb = plt.colorbar(p)
    cb.set_label('log$_{10}(N_{eff})$')
    plt.show()
    
    plt.plot(Neff,dev1,'rx',label='MBAR')
    plt.xlabel('Number of Effective Samples')
    plt.ylabel(dev_type+' Deviation '+units)
    plt.title(title)
    plt.legend()
    plt.show()
    
    plt.plot(Neff,rel_dev_dprop1,'rx',label='MBAR')
    plt.xlabel('Number of Effective Samples')
    plt.ylabel('Coverage Factor')
    plt.title(title)
    plt.legend()
    plt.show()
    
def contour_plot(prop,eps_all,sig_all,prop_direct,prop_hat1,prop_hat2,prop_hat3):
        
    SSE_1 = np.sum((prop_direct-prop_hat1)**2,axis=0).reshape(21,21)
    SSE_2 = np.sum((prop_direct-prop_hat2)**2,axis=0).reshape(21,21)
    SSE_3 = np.sum((prop_direct-prop_hat3)**2,axis=0).reshape(21,21)
    
    RMS_1 = np.sqrt(SSE_1/len(prop_direct))
    RMS_2 = np.sqrt(SSE_2/len(prop_direct))
    RMS_3 = np.sqrt(SSE_3/len(prop_direct))
    
    AAD_1 = np.mean(np.abs((prop_direct-prop_hat1)/prop_direct*100.),axis=0).reshape(21,21)
    AAD_2 = np.mean(np.abs((prop_direct-prop_hat2)/prop_direct*100.),axis=0).reshape(21,21)
    
    logSSE_1 = np.log10(SSE_1)
    logSSE_2 = np.log10(SSE_2)
    
    eps_plot = np.unique(eps_all)
    sig_plot = np.unique(sig_all)
                  
    if prop == 'U':
        units = '(kJ/mol)'
        title = 'Residual Energy'
        contour_lines = [25,50,75,100,125,150,200,250,300,400,500,600,700,800]
    elif prop == 'P':
        units = '(bar)'
        title = 'Pressure'
        contour_lines = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
    elif prop == 'Z':
        units = ''
        title = 'Compressibility Factor'
        contour_lines = [0.5, 1., 1.5, 2., 2.5,3.,3.5]
    elif prop == 'Pdep':
        units = '(bar)'
        title = 'Pressure'
        contour_lines = [100,200,300,400,500]
        
    plt.figure()
    CS = plt.contour(sig_plot,eps_plot,RMS_1,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title('RMS '+units+' of '+prop+' for MBAR')
    plt.show()
    
    plt.figure()
    CS = plt.contour(sig_plot,eps_plot,RMS_2,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title('RMS '+units+' of '+prop+' for PCFR')
    plt.show()
    
    plt.figure()
    CS = plt.contour(sig_plot,eps_plot,RMS_3,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title('RMS '+units+' of '+prop+' for Constant PCF')
    plt.show()
    
#    plt.figure()
#    CS = plt.contour(sig_plot,eps_plot,prop_hat1[0,:])
#    plt.clabel(CS, inline=1,fontsize=10)
#    plt.xlabel(r'$\sigma$ (nm)')
#    plt.ylabel(r'$\epsilon$ (K)')
#    plt.title(prop+''+units+' for MBAR')
#    plt.show()
#    
#    plt.figure()
#    CS = plt.contour(sig_plot,eps_plot,prop_hat2[0,:])
#    plt.clabel(CS, inline=1,fontsize=10)
#    plt.xlabel(r'$\sigma$ (nm)')
#    plt.ylabel(r'$\epsilon$ (K)')
#    plt.title(prop+''+units+' for PCFR')
#    plt.show()

def Pdep_calc(P,Z):
    Zdep = Z-1.
    C_rhoT = P/Z #Constant that converts Z to P
    Pdep = Zdep * C_rhoT
    return Pdep
        
def main():
    
    eps_all, sig_all, eps_matrix, sig_matrix = get_parameter_sets()
    
    for model_type in ['Direct_simulation', 'MBAR_ref0', 'PCFR_ref0','Constant_']:
        if model_type == 'Direct_simulation':
            U_direct, dU_direct, P_direct, dP_direct, Z_direct, dZ_direct, Neff_direct = compile_data(model_type)
        elif model_type == 'MBAR_ref0':
            U_MBAR, dU_MBAR, P_MBAR, dP_MBAR, Z_MBAR, dZ_MBAR, Neff_MBAR = compile_data(model_type)
        elif model_type == 'PCFR_ref0':
            U_PCFR, dU_PCFR, P_PCFR, dP_PCFR, Z_PCFR, dZ_PCFR, Neff_PCFR = compile_data(model_type)
        elif model_type == 'Constant_':
            U_W1, dU_W1, P_W1, dP_W1, Z_W1, dZ_W1, Neff_W1 = compile_data(model_type)
     
    # Taking an average of the PCFR and constant PCF approachs works quite well for P and Z
    #U_PCFR = (U_PCFR + U_W1)/2.
    P_PCFR = (P_PCFR + P_W1)/2.
    Z_PCFR = (Z_PCFR + Z_W1)/2.
    
    plt.scatter(sig_matrix,Neff_MBAR)
    plt.show()
    
    #In my original analysis I forgot to correct for the error associated with ensembles versus integrating histograms
        
    U_ref = U_direct[:,220]
    P_ref = P_direct[:,220]
    
    U_error = U_ref - U_PCFR[:,220]
    P_error = P_ref - P_PCFR[:,220]
                    
    U_PCFR = (U_PCFR.T + U_error).T
    Z_PCFR *= (P_PCFR.T + P_error).T/P_PCFR
    P_PCFR = (P_PCFR.T + P_error).T  
             
    U_W1 = (U_W1.T + U_error).T
    Z_W1 *= (P_W1.T + P_error).T/P_W1
    P_W1 = (P_W1.T + P_error).T  
    
    Neff_min = 50.
    sig_min = 0.373
    sig_max = 0.378
    
    mask_MBAR = Neff_MBAR >= Neff_min
    mask_PCFR = (sig_max >= sig_matrix) & (sig_matrix >= sig_min)
    mask_none = Neff_MBAR > 0.
    
    mask = mask_none
    
    for prop in ['U','P','Z']:
        if prop == 'U':
            prop_direct = U_direct[mask]
            prop_hat1 = U_MBAR[mask]
            prop_hat2 = U_PCFR[mask]
            prop_hat3 = U_W1[mask]
            dprop_direct = dU_direct[mask]
            dprop_MBAR = dU_MBAR[mask]
        
        elif prop == 'P':
            prop_direct = P_direct[mask]
            prop_hat1 = P_MBAR[mask]
            prop_hat2 = P_PCFR[mask]
            prop_hat3 = P_W1[mask]
            dprop_direct = dP_direct[mask]
            dprop_MBAR = dP_MBAR[mask]

        elif prop == 'Z':
            prop_direct = Z_direct[mask]
            prop_hat1 = Z_MBAR[mask] 
            prop_hat2 = Z_PCFR[mask]
            prop_hat3 = Z_W1[mask]
            dprop_direct = dZ_direct[mask]
            dprop_MBAR = dZ_MBAR[mask]
            
        elif prop == 'Pdep':
            Pdep_direct = Pdep_calc(P_direct,Z_direct)
            Pdep_MBAR = Pdep_calc(P_MBAR,Z_MBAR)
            Pdep_PCFR = Pdep_calc(P_PCFR,Z_PCFR)
            prop_direct = Pdep_direct[mask]
            prop_hat1 = Pdep_MBAR[mask]
            prop_hat2 = Pdep_PCFR[mask]
            dprop_direct = dP_direct[mask]
            dprop_MBAR = dP_MBAR[mask]
            
        dprop = np.sqrt(dprop_direct**2. + dprop_MBAR ** 2.) # I need to verify how MBAR reports uncertainties. Are these standard errors? Standard deviations?

        Neff = Neff_MBAR[mask]
        parity_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,Neff,dprop_direct,dprop_MBAR)
        residual_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,Neff,dprop)
        
    contour_plot('U',eps_all,sig_all,U_direct,U_MBAR,U_PCFR,U_W1)
    contour_plot('P',eps_all,sig_all,P_direct,P_MBAR,P_PCFR,P_W1)
    contour_plot('Z',eps_all,sig_all,Z_direct,Z_MBAR,Z_PCFR,Z_W1)
    #contour_plot('Pdep',eps_all,sig_all,Pdep_direct,Pdep_MBAR,Pdep_PCFR)
    
if __name__ == '__main__':
    
    main()