# -*- coding: utf-8 -*-
"""
Compares the MBAR predictions with direct simulation for U and P

@author: ram9
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy import stats
from scipy.optimize import minimize

reference = 'TraPPE'  

fpathroot = 'parameter_space_Mie16/'
nReruns = 441
nStates = 19

T_rho = np.loadtxt('Temp_rho.txt')

sig_PCFs = [0.375, 0.370, 0.3725, 0.365, 0.3675, 0.3775, 0.380, 0.3825, 0.385]
eps_PCFs = [98]*9

def get_parameter_sets():
    
    if fpathroot == 'parameter_space_LJ/':
        eps_sig_all = np.loadtxt(fpathroot+'eps_sig_lam12_all',skiprows=2)
    elif fpathroot == 'parameter_space_Mie16/':
        eps_sig_all = np.loadtxt(fpathroot+'eps_sig_lam16_highEps_all',skiprows=2)
    
    eps_matrix = np.zeros([nStates,nReruns])
    sig_matrix = np.zeros([nStates,nReruns])
    
    eps_all = eps_sig_all[:,0]
    sig_all = eps_sig_all[:,1]
    
    for iState in range(nStates):
        eps_matrix[iState,:] = eps_all
        sig_matrix[iState,:] = sig_all

    return eps_all, sig_all, eps_matrix, sig_matrix

def merge_PCFR(sig_all):
    ending = '_lam12'
    
    U_compiled = np.zeros([nStates,nReruns])
    dU_compiled = np.zeros([nStates,nReruns])
    P_compiled = np.zeros([nStates,nReruns])
    dP_compiled = np.zeros([nStates,nReruns])
    Z_compiled = np.zeros([nStates,nReruns])
    Neff_compiled = np.zeros([nStates,nReruns])
    
    for iRerun, sigRerun in enumerate(sig_all):
        iUPZ = iRerun
        #print(iRerun)
        #print(sigRerun)
        
#        sig_match = str(np.argmin(np.abs(sigRerun - sig_PCFs)))
#        model_type = 'PCFR_ref'+sig_match
#        
#        #print(model_type)
#        
#        fpath = fpathroot+model_type+'rr'+str(iRerun)+ending
#        UPZ = np.loadtxt(fpath)

        dev_sig = np.abs(sig_PCFs - sigRerun)

        Wrefs = np.zeros(len(sig_PCFs))
        
        for iRef, devRef in enumerate(dev_sig):
            if devRef < 0.0025:
                Wrefs[iRef] = 0.5 

        #Wrefs = np.abs(sig_PCFs - sigRerun)/(0.385-0.365)
        
        NConstant = np.sum(Wrefs)
        UPZ = np.zeros([nStates,7]) #7 is the number of properties
        
        for iRef, Wref in enumerate(Wrefs):
            if iRef == 0:
                iRerun += 1
            elif iRef == 1:
                iRerun -= 1
            model_type = 'PCFR_ref'+str(iRef)
            fpath = fpathroot+model_type+'rr'+str(iRerun)+ending
            UPZref = np.loadtxt(fpath)
            UPZ += UPZref * Wref / NConstant
                
        U_compiled[:,iUPZ] = UPZ[:,0]
        dU_compiled[:,iUPZ] = UPZ[:,1]
        P_compiled[:,iUPZ] = UPZ[:,2]
        dP_compiled[:,iUPZ] = UPZ[:,3]
        Z_compiled[:,iUPZ] = UPZ[:,4]
        Neff_compiled[:,iUPZ] = UPZ[:,6]
            
    dZ_compiled = dP_compiled * (Z_compiled/P_compiled)
    
    return U_compiled, dU_compiled, P_compiled, dP_compiled, Z_compiled, dZ_compiled, Neff_compiled
        
        
def compile_data(model_type):
    
    if (reference == 'TraPPE' and fpathroot == 'parameter_space_Mie16/'):
        ending = '_lam16_highEps'
    elif model_type == 'MBAR_ref1' or model_type == 'MBAR_ref8':
        ending = ''
    elif fpathroot == 'parameter_space_LJ/':
        ending = '_lam12'    
    else:
        ending = ''
    
    if model_type == 'TraPPE' or model_type == 'Potoff':
        UPZ = np.loadtxt(model_type)
        U_compiled = UPZ[:,0]
        dU_compiled = UPZ[:,1]
        P_compiled = UPZ[:,2]
        dP_compiled = UPZ[:,3]
        Z_compiled = UPZ[:,4]
        Neff_compiled = UPZ[:,5]
    else:
        U_compiled = np.zeros([nStates,nReruns])
        dU_compiled = np.zeros([nStates,nReruns])
        P_compiled = np.zeros([nStates,nReruns])
        dP_compiled = np.zeros([nStates,nReruns])
        Z_compiled = np.zeros([nStates,nReruns])
        Neff_compiled = np.zeros([nStates,nReruns])
        
        for iRerun in range(nReruns):
            iUPZ = iRerun
            if model_type == 'Direct_simulation' or model_type == 'MBAR_ref0' or fpathroot == 'parameter_space_LJ/' or (model_type == 'PCFR_ref0' and reference == 'TraPPE'):
                iRerun += 1
            if model_type == 'MBAR_ref1':
                iRerun += 1
            if model_type == 'MBAR_ref8':
                iRerun += 8
            if model_type == 'Direct_simulation':
                fpath = fpathroot+model_type+'_rr'+str(iRerun)
                UPZ = np.loadtxt(fpath)            
            else:
                fpath = fpathroot+model_type+'rr'+str(iRerun)+ending
                UPZ = np.loadtxt(fpath)
                
            U_compiled[:,iUPZ] = UPZ[:,0]
            dU_compiled[:,iUPZ] = UPZ[:,1]
            P_compiled[:,iUPZ] = UPZ[:,2]
            dP_compiled[:,iUPZ] = UPZ[:,3]
            Z_compiled[:,iUPZ] = UPZ[:,4]
            Neff_compiled[:,iUPZ] = UPZ[:,6]
            
    dZ_compiled = dP_compiled * (Z_compiled/P_compiled)
    
    return U_compiled, dU_compiled, P_compiled, dP_compiled, Z_compiled, dZ_compiled, Neff_compiled

def parity_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,prop_hat4,Neff,dprop_direct,dprop_hat1):
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
        
    f = plt.figure()

    plt.plot(prop_direct,prop_hat3,'b.',label='Constant PCF',alpha=0.2)    
    plt.plot(prop_direct,prop_hat1,'r.',label='MBAR',alpha=0.2)
    plt.plot(prop_direct,prop_hat2,'g.',label='PCFR',alpha=0.2)
    plt.plot(prop_direct,prop_hat4,'c.',label='Recommended',alpha=0.2)
    #plt.plot(prop_direct,prop_hat3,'bx',label='MBAR, Neff > 10')
    plt.plot(parity,parity,'k',label='Parity')
    plt.xlabel('Direct Simulation '+units)
    plt.ylabel('Predicted '+units)
    plt.title(title)
    plt.legend()
    plt.show()
    
    f.savefig('Parity_comparison_'+prop+'.pdf')
    
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

    f = plt.figure()

    p = plt.scatter(prop_direct[Neff.argsort()],prop_hat1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR') 
    #p = plt.scatter(prop_direct[Neff.argsort()],prop_hat1[Neff.argsort()],c=Neff[Neff.argsort()],cmap='cool',label='MBAR',norm=col.LogNorm())
    plt.plot(parity,parity,'k',label='Parity')
    plt.xlabel('Direct Simulation '+units)
    plt.ylabel('Predicted with MBAR '+units)
    plt.title(title)
    #plt.legend()
    cb = plt.colorbar(p)
    cb.set_label('log$_{10}(N_{eff})$')
    plt.show()
    
    f.savefig('Parity_MBAR_'+prop+'.pdf')
    
def residual_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,prop_hat4,Neff,dprop_direct):
    
    abs_dev = lambda hat, direct: (hat - direct)
    rel_dev = lambda hat, direct: abs_dev(hat,direct)/direct * 100.
    rel_dev_dprop = lambda hat, direct: abs_dev(hat,direct)/dprop_direct
                                         
    abs_dev1 = abs_dev(prop_hat1, prop_direct)
    abs_dev2 = abs_dev(prop_hat2,prop_direct)
    abs_dev3 = abs_dev(prop_hat3,prop_direct)
    abs_dev4 = abs_dev(prop_hat4,prop_direct)
           
    rel_dev1 = rel_dev(prop_hat1,prop_direct)
    rel_dev2 = rel_dev(prop_hat2,prop_direct)
    rel_dev3 = rel_dev(prop_hat3,prop_direct)
    rel_dev4 = rel_dev(prop_hat4,prop_direct)
    
    rel_dev_dprop1 = rel_dev_dprop(prop_hat1,prop_direct)
    rel_dev_dprop2 = rel_dev_dprop(prop_hat2,prop_direct)
    rel_dev_dprop3 = rel_dev_dprop(prop_hat3,prop_direct)
    rel_dev_dprop4 = rel_dev_dprop(prop_hat4,prop_direct)
               
    if prop == 'U':
        units = '(kJ/mol)'
        title = 'Residual Energy'
        dev1 = rel_dev1
        dev2 = rel_dev2
        dev3 = rel_dev3
        dev4 = rel_dev4
        dev_type = 'Percent'
    elif prop == 'P':
        units = '(bar)'
        title = 'Pressure'
        dev1 = abs_dev1
        dev2 = abs_dev2
        dev3 = abs_dev3
        dev4 = abs_dev4
        dev_type = 'Absolute'
    elif prop == 'Z':
        units = ''
        title = 'Compressibility Factor'
        dev1 = abs_dev1
        dev2 = abs_dev2
        dev3 = abs_dev3
        dev4 = abs_dev4
        dev_type = 'Absolute'
    elif prop == 'Pdep':
        units = '(bar)'
        title = 'Pressure - Ideal Gas'
        dev1 = abs_dev1
        dev2 = abs_dev2
        dev3 = abs_dev3
        dev4 = abs_dev4
        dev_type = 'Absolute'
        
    f = plt.figure()

    plt.plot(prop_direct,dev3,'bx',label='Constant PCF',alpha=0.2)        
    plt.plot(prop_direct,dev1,'rx',label='MBAR',alpha=0.2)
    plt.plot(prop_direct,dev2,'gx',label='PCFR',alpha=0.2)
    plt.plot(prop_direct,dev4,'cx',label='Recommended',alpha=0.2)
    #plt.plot(prop_direct,dev3,'bx',label='MBAR, Neff > 10')
    plt.xlabel('Direct Simulation '+units)
    if dev_type == 'Percent':
        plt.ylabel(dev_type+' Deviation ')
    else:
        plt.ylabel(dev_type+' Deviation '+units)
    plt.title(title)
    plt.legend()
    plt.show()
    
    f.savefig('Residual_comparison_'+prop+'.pdf')
    
    f,ax = plt.subplots()
    
    plt.boxplot([dev1,dev2,dev3,dev4],labels=['MBAR','PCFR','Constant PCF','Recommended'])
    if dev_type == 'Percent':
        plt.ylabel(dev_type+' Deviation ')
    else:
        plt.ylabel(dev_type+' Deviation '+units)
    #ax.yaxis.grid(True)
    ax.axhline(0.,linestyle='--',color='b')
    plt.title(title)
    plt.show()
    
    f.savefig('Boxplot_'+prop+'.pdf')
    
#    plt.plot(prop_direct,rel_dev_dprop3,'bx',label='Constant PCF')
#    plt.plot(prop_direct,rel_dev_dprop1,'rx',label='MBAR')
#    plt.plot(prop_direct,rel_dev_dprop2,'gx',label='PCFR')
#    #plt.plot(prop_direct,rel_dev_dprop3,'bx',label='MBAR, Neff > 10')
#    plt.xlabel('Direct Simulation '+units)
#    plt.ylabel('Coverage Factor')
#    plt.title(title)
#    plt.legend()
#    plt.show()
    
    f = plt.figure()
    
    p = plt.scatter(prop_direct[Neff.argsort()],dev1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR')
    plt.xlabel('Direct Simulation '+units)
    if dev_type == 'Percent':
        plt.ylabel(dev_type+' Deviation ')
    else:
        plt.ylabel(dev_type+' Deviation '+units)
    plt.title(title)
    #plt.legend()
    cb = plt.colorbar(p)
    cb.set_label('log$_{10}(N_{eff})$')
    plt.show()
    
    f.savefig('Residual_MBAR_'+prop+'.pdf')
    
#    p = plt.scatter(prop_direct,rel_dev_dprop1,c=np.log10(Neff),cmap='cool',label='MBAR')
#    plt.xlabel('Direct Simulation '+units)
#    plt.ylabel('Coverage Factor')
#    plt.title(title)
#    #plt.legend()
#    cb = plt.colorbar(p)
#    cb.set_label('log$_{10}(N_{eff})$')
#    plt.show()
    
    plt.plot(Neff,dev1,'rx',label='MBAR')
    plt.xlabel('Number of Effective Samples')
    if dev_type == 'Percent':
        plt.ylabel(dev_type+' Deviation ')
    else:
        plt.ylabel(dev_type+' Deviation '+units)
    plt.title(title)
    plt.legend()
    plt.show()
    
#    plt.plot(Neff,rel_dev_dprop1,'rx',label='MBAR')
#    plt.xlabel('Number of Effective Samples')
#    plt.ylabel('Coverage Factor')
#    plt.title(title)
#    plt.legend()
#    plt.show()
    
def contour_plot(prop,eps_all,sig_all,prop_direct,prop_hat1,prop_hat2,prop_hat3,prop_hat4):
        
    SSE_1 = np.sum((prop_direct-prop_hat1)**2,axis=0).reshape(21,21)
    SSE_2 = np.sum((prop_direct-prop_hat2)**2,axis=0).reshape(21,21)
    SSE_3 = np.sum((prop_direct-prop_hat3)**2,axis=0).reshape(21,21)
    SSE_4 = np.sum((prop_direct-prop_hat4)**2,axis=0).reshape(21,21)
    
    RMS_1 = np.sqrt(SSE_1/len(prop_direct))
    RMS_2 = np.sqrt(SSE_2/len(prop_direct))
    RMS_3 = np.sqrt(SSE_3/len(prop_direct))
    RMS_4 = np.sqrt(SSE_4/len(prop_direct))
    
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
        contour_lines = [0.1, 0.2, 0.3, 0.4, 0.5, 1., 1.5, 2., 2.5,3.,3.5]
    elif prop == 'Pdep':
        units = '(bar)'
        title = 'Pressure'
        contour_lines = [100,200,300,400,500]
        
    if False:
        xlabel = r'$\sigma$ (nm)'
        x_plot = sig_plot
    else:
        if fpathroot == 'parameter_space_Mie16/':
            lam = 16.
        elif fpathroot == 'parameter_space_LJ/':
            lam = 12.
        xlabel = r'r$_{min}$ (nm)'
        x_plot = calc_rmin(sig_plot,lam)
   
    f = plt.figure()
    
    CS = plt.contour(x_plot,eps_plot,RMS_1,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title('RMS '+units+' of '+prop+' for MBAR')
    plt.show()
    
    f.savefig('Contour_MBAR_'+prop+'.pdf')
    
    f = plt.figure()
    
    CS = plt.contour(x_plot,eps_plot,RMS_2,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title('RMS '+units+' of '+prop+' for PCFR')
    plt.show()
    
    f.savefig('Contour_PCFR_'+prop+'.pdf')
    
    f = plt.figure()
    
    CS = plt.contour(x_plot,eps_plot,RMS_3,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title('RMS '+units+' of '+prop+' for Constant PCF')
    plt.show()
    
    f.savefig('Contour_constant_PCF_'+prop+'.pdf')
    
    f = plt.figure()
    
    CS = plt.contour(x_plot,eps_plot,RMS_4,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title('RMS '+units+' of '+prop+' for Recommended')
    plt.show()
    
    f.savefig('Contour_recommended_'+prop+'.pdf')
    
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

def uncertainty_check(prop_direct, prop_MBAR,u_direct, u_MBAR,Neff_MBAR):
    
    dev_MBAR = np.abs(prop_MBAR - prop_direct)
    Neff_acc = []
    Neff_rej = []
    u_total = np.sqrt(u_direct**2 + u_MBAR**2)
    
    for i, dev in enumerate(dev_MBAR):
        if dev < u_total[i]:
            Neff_acc.append(Neff_MBAR[i])
        else:
            Neff_rej.append(Neff_MBAR[i])
            
    Neff_acc.sort()
    Neff_rej.sort()
    
    plt.plot(Neff_acc,label='Accurate error estimate')
    plt.show()
    
    plt.plot(Neff_rej,label='Inaccurate error estimate')
    plt.show()
    
    plt.boxplot(Neff_acc)
    plt.show()
    
    plt.boxplot(Neff_rej)
    plt.show()
    
    plt.hist(Neff_acc)
    plt.show()
    
    plt.hist(Neff_rej)
    plt.show()
    
def PCFR_error(U_ref, P_ref,ref):
    if ref == 'TraPPE':
        UPZ_PCFR = np.loadtxt('parameter_space_LJ/PCFR_ref0rr221_lam12') 
    elif ref == 'Potoff':
        UPZ_PCFR = np.loadtxt('parameter_space_Mie16/PCFR_ref0rr286')
        
    U_PCFR = UPZ_PCFR[:,0]
    P_PCFR = UPZ_PCFR[:,2]
    
    U_error = U_ref - U_PCFR
    P_error = P_ref - P_PCFR
    
#    if fpathroot == 'parameter_space_Mie16/' and reference == 'TraPPE':
#           
#        U_error = 0.
#        P_error = 0.
    
    return U_error, P_error

def box_bar_state_plots(Neff_MBAR,Neff_min,Neff_small,mask_MBAR,mask_poor):
    ''' 
    Some miscellaneous plots that provide insight into what state points typically
    have a large enough Neff.
    '''
    #plt.boxplot(Neff_MBAR.T)
    plt.boxplot(np.log10(Neff_MBAR.T))
    #plt.ylabel('$N_{eff}$')
    plt.ylabel('log$_{10}(N_{eff})$')
    plt.xlabel('State Point')
    plt.show()
    
    #    fig, ax = plt.subplots()
#    for iState in range(nStates):
#        ax.boxplot(Neff_MBAR[iState,:],positions=[iState])
#    ax.set_xlim([-0.5,18.5])
#    plt.show()
    
    plt.bar(range(nStates),sum(mask_MBAR.T)/float(nReruns)*100.)
    plt.ylabel('Percent N$_{eff}>$'+str(int(Neff_min)))
    plt.xlabel('State Point')
    plt.show()
    
    plt.bar(range(nStates),sum(mask_poor.T)/float(nReruns)*100.)
    plt.ylabel('Percent N$_{eff}$<'+str(int(Neff_small)))
    plt.xlabel('State Point')
    plt.show()
    
    plt.plot(T_rho[:,1],T_rho[:,0],'bo')
    plt.show()
    
    plt.scatter(T_rho[:,1],T_rho[:,0],s=sum(mask_MBAR.T)/float(nReruns)*100.)
    plt.xlabel(r'Density $\left(\frac{kg}{m^3}\right)$')
    plt.ylabel('Temperature (K)')
    plt.title('Percent N$_{eff}>$'+str(int(Neff_min)))
    plt.show()
    
    plt.scatter(T_rho[:,1],T_rho[:,0],s=sum(mask_poor.T)/float(nReruns)*100.)
    plt.xlabel(r'Density $\left(\frac{kg}{m^3}\right)$')
    plt.ylabel('Temperature (K)')
    plt.title('Percent N$_{eff}$<'+str(int(Neff_small)))
    plt.show()
    
    p = plt.scatter(T_rho[:,1],T_rho[:,0],c=sum(mask_MBAR.T)/float(nReruns)*100.,cmap='cool')
    plt.xlabel(r'Density $\left(\frac{kg}{m^3}\right)$')
    plt.ylabel('Temperature (K)')
    cb = plt.colorbar(p)
    cb.set_label('Percent N$_{eff}>$'+str(int(Neff_min)))
    plt.show()
    
    p = plt.scatter(T_rho[:,1],T_rho[:,0],c=sum(mask_poor.T)/float(nReruns)*100.,cmap='cool')
    plt.xlabel(r'Density $\left(\frac{kg}{m^3}\right)$')
    plt.ylabel('Temperature (K)')
    cb = plt.colorbar(p)
    cb.set_label('Percent N$_{eff}$<'+str(int(Neff_small)))
    plt.show()
    
def calc_rmin(sigma,n,m=6.):
    """ Calculates the rmin for LJ potential """
    rmin = (n/m*sigma**(n-m))**(1./(n-m))
    return rmin

def contours_Neff(Neff_MBAR,sig_all,eps_all):
    
    sig_plot = np.unique(sig_all)
    eps_plot = np.unique(eps_all)
    
    if True:
        xlabel = r'$\sigma$ (nm)'
        x_plot = sig_plot
    else:
        if fpathroot == 'parameter_space_Mie16/':
            lam = 16.
        elif fpathroot == 'parameter_space_LJ/':
            lam = 12.
        xlabel = r'r$_{min}$ (nm)'
        x_plot = calc_rmin(sig_plot,lam)
        
    #CS = plt.contour(x_plot,eps_plot,np.mean(Neff_MBAR,axis=0).reshape(21,21),[0,1,2,3,4,5,6,7,8,9,10,11,12])
    CS = plt.contour(x_plot,eps_plot,np.mean(Neff_MBAR,axis=0).reshape(21,21))
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title('Average N$_{eff}$')
    plt.show()
   
    for iState in range(nStates):
            
        CS = plt.contour(x_plot,eps_plot,Neff_MBAR[iState,:].reshape(21,21),[1,2,5,10,20,50,100,200,500,900,1500,2000])
        plt.clabel(CS, inline=1,fontsize=10)
        plt.xlabel(xlabel)
        plt.ylabel(r'$\epsilon$ (K)')
        plt.title('N$_{eff}$')
        plt.show()
        
def RMS_contours(eps_all,sig_all):
    
    eps_plot = np.unique(eps_all)
    sig_plot = np.unique(sig_all)
    
#    contour_lines = [5,10,20,30,40,50,60,70,80,90,100,150,200]
#    
#    RMSrhoL = np.loadtxt(fpathroot+'Direct_simulation_rr_RMS_rhoL_all').reshape(21,21)
#    RMSrhoL_MBAR = np.loadtxt(fpathroot+'/Potoff/MBAR/MBAR_ref0rr_RMS_rhoL_all').reshape(21,21)
#    CS1 = plt.contour(sig_plot,eps_plot,RMSrhoL,contour_lines,label='Direct Simulation',colors='r')
#    CS2 = plt.contour(sig_plot,eps_plot,RMSrhoL_MBAR,contour_lines,label='MBAR Simulation',colors='b')
#    plt.clabel(CS1, inline=1,fontsize=10,colors='k',fmt='%1.0f')
#    plt.xlabel(r'$\sigma$ (nm)')
#    plt.ylabel(r'$\epsilon$ (K)')
#    plt.title(r'RMS of $\rho_l  \left(\frac{kg}{m^3}\right)$')
#    plt.show()
#    
#    RMSPsat = np.loadtxt(fpathroot+'Direct_simulation_rr_RMS_Psat_all').reshape(21,21)
#        
#    contour_lines = [0.8,1.6,2.4,3.2,4,5,6,7,8,9,10,12,15,20]
#    
#    CS1 = plt.contour(sig_plot,eps_plot,RMSPsat,contour_lines,label='Direct Simulation',colors='r')
#    plt.clabel(CS1, inline=0,fontsize=10,colors='k',fmt='%1.1f')
#    plt.xlabel(r'$\sigma$ (nm)')
#    plt.ylabel(r'$\epsilon$ (K)')
#    plt.title(r'RMS of $P_v  \left(bar\right)$')    
#    plt.show()
#    
#    RMS_Z = np.loadtxt(fpathroot+'Direct_simulation_rr_RMS_Z_all').reshape(21,21)
#    CS1 = plt.contour(sig_plot,eps_plot,RMS_Z,label='Direct Simulation',colors='r')
#    plt.clabel(CS1, inline=1,fontsize=10,colors='k',fmt='%1.1f')
#    plt.xlabel(r'$\sigma$ (nm)')
#    plt.ylabel(r'$\epsilon$ (K)')
#    plt.title(r'RMS of Z')
#    plt.show()
#    
#    RMS_U = np.loadtxt(fpathroot+'Direct_simulation_rr_RMS_U_all').reshape(21,21)
#    CS1 = plt.contour(sig_plot,eps_plot,RMS_U,label='Direct Simulation',colors='r')
#    plt.clabel(CS1, inline=1,fontsize=10,colors='k',fmt='%1.1f')
#    plt.xlabel(r'$\sigma$ (nm)')
#    plt.ylabel(r'$\epsilon$ (K)')
#    plt.title(r'RMS of U')
#    plt.show()
#    
#    return
    
    RMSrhoL = np.loadtxt(fpathroot+'Direct_simulation_rr_RMS_rhoL_all').reshape(21,21)
    RMSrhoL_MBAR = np.loadtxt(fpathroot+'MBAR_ref0rr_RMS_rhoL_all').reshape(21,21)
    RMSrhoL_PCFR = np.loadtxt(fpathroot+'PCFR_ref0rr_RMS_rhoL_all').reshape(21,21)
    RMSrhoL_constant = np.loadtxt(fpathroot+'Constant_rr_RMS_rhoL_all').reshape(21,21)
    RMSrhoL_MBAR_9refs = np.loadtxt(fpathroot+'MBAR_ref8rr_RMS_rhoL_all').reshape(21,21)
    
    contour_lines = [5,10,20,30,40,50,60,70,80,90,100,150,200]
    
    f = plt.figure(figsize=(8,6))
    
    CS3 = plt.contour(sig_plot,eps_plot,RMSrhoL_MBAR,[5,10],label='MBAR with single reference',colors='g')
    CS4 = plt.contour(sig_plot,eps_plot,RMSrhoL_PCFR,[5,10],label='PCFR with single reference',colors='c')
    CS1 = plt.contour(sig_plot,eps_plot,RMSrhoL,contour_lines,label='Direct Simulation',colors='r')
    CS2 = plt.contour(sig_plot,eps_plot,RMSrhoL_MBAR_9refs,contour_lines,label='MBAR with multiple references',colors='b')
    plt.clabel(CS2, inline=1,fontsize=10,colors='w',fmt='%1.0f')
    plt.clabel(CS4, inline=1,fontsize=10,colors='c',fmt='%1.0f',manual=[(0,-10),(5,10)])
    plt.clabel(CS3, inline=1,fontsize=10,colors='g',fmt='%1.0f')
    plt.clabel(CS1, inline=1,fontsize=10,colors='k',fmt='%1.0f')
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $\rho_l  \left(\frac{kg}{m^3}\right)$')
    plt.plot([],[],'r',label='Direct Simulation')
    plt.plot([],[],'b',label='MBAR multiple references')
    plt.plot([],[],'g',label='MBAR single reference')
    plt.plot([],[],'c',label='PCFR single reference')
    plt.plot(sig_PCFs,eps_PCFs,'mx',label='References')
    plt.legend()
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    f.savefig('RMS_rhol_comparison.pdf')
    
    
    CS = plt.contour(sig_plot,eps_plot,RMSrhoL)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $\rho_l$ with Direct Simulation')
    plt.plot(sig_all,eps_all,'rx',label='Simulations')
    plt.show()
    
    CS = plt.contour(sig_plot,eps_plot,RMSrhoL_MBAR,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $\rho_l$ with MBAR single reference')
    plt.plot(sig_PCFs[0],eps_PCFs[0],'rx',label='References')
    plt.show()
    
    CS = plt.contour(sig_plot,eps_plot,RMSrhoL_PCFR,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $\rho_l$ with PCFR single reference')
    plt.plot(sig_PCFs[0],eps_PCFs[0],'rx',label='References')
    plt.show()
    
    CS = plt.contour(sig_plot,eps_plot,RMSrhoL_constant,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $\rho_l$ with constant PCF single reference')
    plt.show()
    
    CS = plt.contour(sig_plot,eps_plot,RMSrhoL_MBAR_9refs)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $\rho_l$ with MBAR multiple references')
    plt.plot(sig_PCFs,eps_PCFs,'rx',label='References')
    plt.legend()
    plt.show()
    
    RMSPsat = np.loadtxt(fpathroot+'Direct_simulation_rr_RMS_Psat_all').reshape(21,21)
    RMSPsat_MBAR = np.loadtxt(fpathroot+'MBAR_ref0rr_RMS_Psat_all').reshape(21,21)
    RMSPsat_PCFR = np.loadtxt(fpathroot+'PCFR_ref0rr_RMS_Psat_all').reshape(21,21)
    RMSPsat_constant = np.loadtxt(fpathroot+'Constant_rr_RMS_Psat_all').reshape(21,21)
    RMSPsat_MBAR_9refs = np.loadtxt(fpathroot+'MBAR_ref8rr_RMS_Psat_all').reshape(21,21)
    
    contour_lines = [0.8,1.6,2.4,3.2,4,5,6,7,8,9,10,12,15,20]
    
    f = plt.figure(figsize=(8,6))
    
    CS3 = plt.contour(sig_plot,eps_plot,RMSPsat_MBAR,[0.8,1.6],label='MBAR with single reference',colors='g')
    CS4 = plt.contour(sig_plot,eps_plot,RMSPsat_PCFR,[0.8,1.6],label='PCFR with single reference',colors='c')
    CS1 = plt.contour(sig_plot,eps_plot,RMSPsat,contour_lines,label='Direct Simulation',colors='r')
    CS2 = plt.contour(sig_plot,eps_plot,RMSPsat_MBAR_9refs,contour_lines,label='MBAR with multiple references',colors='b')
    #plt.clabel(CS2, inline=1,fontsize=10,colors='w',fmt='%1.1f')
    plt.clabel(CS4, inline=1,fontsize=10,colors='c',fmt='%1.1f')
    plt.clabel(CS3, inline=1,fontsize=10,colors='g',fmt='%1.1f')
    plt.clabel(CS1, inline=0,fontsize=10,colors='k',fmt='%1.1f')
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $P_v  \left(bar\right)$')
    plt.plot([],[],'r',label='Direct Simulation')
    plt.plot([],[],'b',label='MBAR multiple references')
    plt.plot([],[],'g',label='MBAR single reference')
    plt.plot([],[],'c',label='PCFR single reference')
    plt.plot(sig_PCFs,eps_PCFs,'mx',label='References')
    plt.legend()
    plt.show()
    
    f.savefig('RMS_Psat_comparison.pdf')
        
    CS = plt.contour(sig_plot,eps_plot,RMSPsat)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $P_v$ with Direct Simulation')
    plt.show()
    
    CS = plt.contour(sig_plot,eps_plot,RMSPsat_MBAR,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $P_v$ with MBAR single reference')
    plt.show()
    
    CS = plt.contour(sig_plot,eps_plot,RMSPsat_PCFR,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $P_v$ with PCFR single reference')
    plt.show()
    
    CS = plt.contour(sig_plot,eps_plot,RMSPsat_constant,contour_lines)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $P_v$ with constant PCF single reference')
    plt.show()
    
    CS = plt.contour(sig_plot,eps_plot,RMSPsat_MBAR_9refs)
    plt.clabel(CS, inline=1,fontsize=10)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of $P_v$ with MBAR multiple references')
    plt.show()
        
def main():
    
    eps_all, sig_all, eps_matrix, sig_matrix = get_parameter_sets()
    
    RMS_contours(eps_all,sig_all)
    
    return
    
    for model_type in [reference,'Direct_simulation', 'MBAR_ref8', 'PCFR_mult_ref','Constant_']:
        if model_type == 'TraPPE' or model_type == 'Potoff':
            U_ref, dU_ref, P_ref, dP_ref, Z_ref, dZ_ref, Neff_ref = compile_data(model_type)
            # Now I call a function that should calculate the error in the proper manner
            U_error, P_error = PCFR_error(U_ref,P_ref,model_type)
        elif model_type == 'Direct_simulation':
            U_direct, dU_direct, P_direct, dP_direct, Z_direct, dZ_direct, Neff_direct = compile_data(model_type)
        elif model_type == 'MBAR_ref0' or model_type == 'MBAR_ref1' or model_type == 'MBAR_ref8':
            U_MBAR, dU_MBAR, P_MBAR, dP_MBAR, Z_MBAR, dZ_MBAR, Neff_MBAR = compile_data(model_type)
        elif model_type == 'PCFR_ref0':
            U_PCFR, dU_PCFR, P_PCFR, dP_PCFR, Z_PCFR, dZ_PCFR, Neff_PCFR = compile_data(model_type)
        elif model_type == 'PCFR_mult_ref':
            U_PCFR, dU_PCFR, P_PCFR, dP_PCFR, Z_PCFR, dZ_PCFR, Neff_PCFR = merge_PCFR(sig_all)
        elif model_type == 'Constant_':
            U_W1, dU_W1, P_W1, dP_W1, Z_W1, dZ_W1, Neff_W1 = compile_data(model_type)

#    plt.scatter(sig_matrix,Neff_MBAR)
#    plt.show()

    #In my original analysis I forgot to correct for the error associated with ensembles versus integrating histograms
    #Should actually be 221, I believe. Depends on how the matrices are built. I know it is rr221, but is that index 220?     
    #U_ref = U_direct[:,220]
    #P_ref = P_direct[:,220]
    
    #U_error = U_ref - U_PCFR[:,220]
    #P_error = P_ref - P_PCFR[:,220]
                
    U_PCFR = (U_PCFR.T + U_error).T
    Z_PCFR *= (P_PCFR.T + P_error).T/P_PCFR
    P_PCFR = (P_PCFR.T + P_error).T  
             
    U_W1 = (U_W1.T + U_error).T
    Z_W1 *= (P_W1.T + P_error).T/P_W1
    P_W1 = (P_W1.T + P_error).T  
           
    # Taking an average of the PCFR and constant PCF approachs works quite well for P and Z
    #U_PCFR = (U_PCFR + U_W1)/2.
    #P_PCFR = (P_PCFR + P_W1)/2.
    #Z_PCFR = (Z_PCFR + Z_W1)/2.
             
    # Just testing something out
#    U_PCFR = (U_PCFR + U_MBAR)/2.
#    P_PCFR = (P_PCFR + P_MBAR)/2.
#    Z_PCFR = (Z_PCFR + Z_MBAR)/2.
#    

               
    Neff_min = 30.
    Neff_small = 2.
    sig_min = 0.373
    sig_max = 0.378
    
    mask_MBAR = Neff_MBAR >= Neff_min
    mask_PCFR = (sig_max >= sig_matrix) & (sig_matrix >= sig_min)
    mask_none = Neff_MBAR > 0.
    mask_poor = Neff_MBAR <= Neff_small
    
    mask = mask_none
    
    # Recommended values
#
    U_rec = U_MBAR.copy()
    P_rec = P_MBAR.copy()
    Z_rec = Z_MBAR.copy()

    U_rec[~mask_MBAR] = (U_PCFR[~mask_MBAR] + U_MBAR[~mask_MBAR])/2.
    P_rec[~mask_MBAR] = (P_PCFR[~mask_MBAR] + P_MBAR[~mask_MBAR])/2.
    Z_rec[~mask_MBAR] = (Z_PCFR[~mask_MBAR] + Z_MBAR[~mask_MBAR])/2.
         
#    U_rec[~mask_MBAR] = U_PCFR[~mask_MBAR]
#    U_rec[~mask_MBAR] = (U_PCFR[~mask_MBAR] + U_MBAR[~mask_MBAR])/2.
#    P_rec[~mask_MBAR] = (P_PCFR[~mask_MBAR] + P_W1[~mask_MBAR])/2.
#    Z_rec[~mask_MBAR] = (Z_PCFR[~mask_MBAR] + Z_W1[~mask_MBAR])/2.
   
#   #Alternatively this could be useful
#    U_W1 = (U_PCFR + U_W1)/2.
#    P_W1 = (P_PCFR + P_W1)/2.
#    Z_W1 = (Z_PCFR + Z_W1)/2.
# 
#    U_MBAR[~mask_MBAR] = U_PCFR[~mask_MBAR]
#    P_MBAR[~mask_MBAR] = P_PCFR[~mask_MBAR]
#    Z_MBAR[~mask_MBAR] = Z_PCFR[~mask_MBAR]
#    

#    U_MBAR[~mask_MBAR] = (U_PCFR[~mask_MBAR] + U_W1[~mask_MBAR])/2.
#    P_MBAR[~mask_MBAR] = (P_PCFR[~mask_MBAR] + P_W1[~mask_MBAR])/2.
#    Z_MBAR[~mask_MBAR] = (Z_PCFR[~mask_MBAR] + Z_W1[~mask_MBAR])/2.
#          
#    U_MBAR[~mask_MBAR] = (U_PCFR[~mask_MBAR]*(Neff_min-Neff_MBAR[~mask_MBAR])/Neff_min + U_MBAR[~mask_MBAR]*(Neff_MBAR[~mask_MBAR])/Neff_min)
#    P_MBAR[~mask_MBAR] = (P_PCFR[~mask_MBAR]*(Neff_min-Neff_MBAR[~mask_MBAR])/Neff_min + P_MBAR[~mask_MBAR]*(Neff_MBAR[~mask_MBAR])/Neff_min)
#    Z_MBAR[~mask_MBAR] = (Z_PCFR[~mask_MBAR]*(Neff_min-Neff_MBAR[~mask_MBAR])/Neff_min + Z_MBAR[~mask_MBAR]*(Neff_MBAR[~mask_MBAR])/Neff_min)
    
    
    for prop in ['U','P','Z']:
        if prop == 'U':
            prop_direct = U_direct[mask]
            prop_hat1 = U_MBAR[mask]
            prop_hat2 = U_PCFR[mask]
            prop_hat3 = U_W1[mask]
            prop_hat4 = U_rec[mask]
            dprop_direct = dU_direct[mask]
            dprop_MBAR = dU_MBAR[mask]
        
        elif prop == 'P':
            prop_direct = P_direct[mask]
            prop_hat1 = P_MBAR[mask]
            prop_hat2 = P_PCFR[mask]
            prop_hat3 = P_W1[mask]
            prop_hat4 = P_rec[mask]
            dprop_direct = dP_direct[mask]
            dprop_MBAR = dP_MBAR[mask]

        elif prop == 'Z':
            prop_direct = Z_direct[mask]
            prop_hat1 = Z_MBAR[mask] 
            prop_hat2 = Z_PCFR[mask]
            prop_hat3 = Z_W1[mask]
            prop_hat4 = Z_rec[mask]
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
        parity_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,prop_hat4,Neff,dprop_direct,dprop_MBAR)
        residual_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,prop_hat4,Neff,dprop)
        #uncertainty_check(prop_direct,prop_hat1,dprop_direct,dprop_MBAR,Neff)
        
    contour_plot('U',eps_all,sig_all,U_direct,U_MBAR,U_PCFR,U_W1,U_rec)
    contour_plot('P',eps_all,sig_all,P_direct,P_MBAR,P_PCFR,P_W1,P_rec)
    contour_plot('Z',eps_all,sig_all,Z_direct,Z_MBAR,Z_PCFR,Z_W1,Z_rec)
    #contour_plot('Pdep',eps_all,sig_all,Pdep_direct,Pdep_MBAR,Pdep_PCFR)
    
    #box_bar_state_plots(Neff_MBAR,Neff_min,Neff_small,mask_MBAR,mask_poor)
    #contours_Neff(Neff_MBAR,sig_all,eps_all)
    
if __name__ == '__main__':
    
    main()