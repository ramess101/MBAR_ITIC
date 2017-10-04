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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

reference = 'TraPPE'  

nReruns = 441
nStates = 19

T_rho = np.loadtxt('Temp_rho.txt')

sig_PCFs = [0.375, 0.370, 0.3725, 0.365, 0.3675, 0.3775, 0.380, 0.3825, 0.385]
eps_PCFs = [98]*9

def get_parameter_sets(fpathroot):
    
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

def merge_PCFR(sig_all,fpathroot):
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
        
        
def compile_data(model_type,fpathroot):
    
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
            if (model_type == 'MBAR_ref8' and fpathroot == 'parameter_space_Mie16/'):
                iRerun += 1
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
        fmt_prop = '%1.0f'
    elif prop == 'P':
        units = '(bar)'
        title = 'Pressure'
        contour_lines = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
        fmt_prop = '%1.0f'
    elif prop == 'Z':
        units = ''
        title = 'Compressibility Factor'
        contour_lines = [0.1, 0.5, 1., 1.5, 2., 2.5,3.,3.5]
        fmt_prop = '%1.1f'
    elif prop == 'Pdep':
        units = '(bar)'
        title = 'Pressure'
        contour_lines = [100,200,300,400,500]
        
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
        
    f = plt.figure()
    
    CS1 = plt.contour(x_plot,eps_plot,RMS_1,contour_lines,colors='r')
    CS2 = plt.contour(x_plot,eps_plot,RMS_2,contour_lines,colors='g')
    #CS3 = plt.contour(x_plot,eps_plot,RMS_1,contour_lines,colors='b')
    plt.clabel(CS1, inline=1,fontsize=10,colors='r',fmt=fmt_prop)
    plt.clabel(CS2, inline=1,fontsize=10,colors='g',fmt=fmt_prop)
    #plt.clabel(CS3, inline=1,fontsize=10,colors='b')
    plt.xlabel(xlabel)
    plt.ylabel(r'$\epsilon$ (K)')
    plt.plot([],[],'r',label='MBAR')
    plt.plot([],[],'g',label='PCFR')
    plt.plot(sig_PCFs[0],eps_PCFs[0],'mx',label='References')
    plt.title('RMS '+units+' of '+prop+' for MBAR')
    plt.legend()
    plt.show()
    
    f.savefig('Contour_comparison_'+prop+'.pdf')
   
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

def contours_Neff(Neff_MBAR,sig_all,eps_all,fpathroot):
    
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
        
def RMS_contours(eps_all,sig_all,fpathroot):
    
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

def RMS_contours_combined(eps_all,sig_all,fpathroot):
    
    eps_plot = np.unique(eps_all)
    sig_plot = np.unique(sig_all)
    
    RMSrhoL = np.loadtxt(fpathroot+'Direct_simulation_rr_RMS_rhoL_all').reshape(21,21)
    RMSrhoL_MBAR = np.loadtxt(fpathroot+'MBAR_ref0rr_RMS_rhoL_all').reshape(21,21)
    RMSrhoL_PCFR = np.loadtxt(fpathroot+'PCFR_ref0rr_RMS_rhoL_all').reshape(21,21)
    RMSrhoL_constant = np.loadtxt(fpathroot+'Constant_rr_RMS_rhoL_all').reshape(21,21)
    RMSrhoL_MBAR_9refs = np.loadtxt(fpathroot+'MBAR_ref8rr_RMS_rhoL_all').reshape(21,21)
    
    contour_lines = [5,10,20,30,40,50,60,70,80,90,100,150,200]

    f, axarr = plt.subplots(nrows=2,ncols=1,figsize=(8,12))
    
    plt.text(10,0,'a)') 
    plt.text(-5,0,'b)')
    
    CS3 = axarr[0].contour(sig_plot,eps_plot,RMSrhoL_MBAR,[5,10],label='MBAR with single reference',colors='g')
    CS4 = axarr[0].contour(sig_plot,eps_plot,RMSrhoL_PCFR,[5,10],label='PCFR with single reference',colors='c')
    CS1 = axarr[0].contour(sig_plot,eps_plot,RMSrhoL,contour_lines,label='Direct Simulation',colors='r')
    CS2 = axarr[0].contour(sig_plot,eps_plot,RMSrhoL_MBAR_9refs,contour_lines,label='MBAR with multiple references',colors='b')
    axarr[0].clabel(CS2, inline=1,fontsize=10,colors='w',fmt='%1.0f')
    axarr[0].clabel(CS4, inline=1,fontsize=10,colors='c',fmt='%1.0f',manual=[(0,-10),(5,10)])
    axarr[0].clabel(CS3, inline=1,fontsize=10,colors='g',fmt='%1.0f')
    axarr[0].clabel(CS1, inline=1,fontsize=10,colors='k',fmt='%1.0f')
    axarr[0].set_xlabel(r'$\sigma$ (nm)')
    axarr[0].set_ylabel(r'$\epsilon$ (K)')
    axarr[0].set_title(r'RMS of $\rho_l  \left(\frac{kg}{m^3}\right)$')
    axarr[0].plot([],[],'r',label='Direct Simulation')
    axarr[0].plot([],[],'b',label='MBAR multiple references')
    axarr[0].plot([],[],'g',label='MBAR single reference')
    axarr[0].plot([],[],'c',label='PCFR single reference')
    axarr[0].plot(sig_PCFs,eps_PCFs,'mx',label='References')
    axarr[0].legend()
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
    RMSPsat = np.loadtxt(fpathroot+'Direct_simulation_rr_RMS_Psat_all').reshape(21,21)
    RMSPsat_MBAR = np.loadtxt(fpathroot+'MBAR_ref0rr_RMS_Psat_all').reshape(21,21)
    RMSPsat_PCFR = np.loadtxt(fpathroot+'PCFR_ref0rr_RMS_Psat_all').reshape(21,21)
    RMSPsat_constant = np.loadtxt(fpathroot+'Constant_rr_RMS_Psat_all').reshape(21,21)
    RMSPsat_MBAR_9refs = np.loadtxt(fpathroot+'MBAR_ref8rr_RMS_Psat_all').reshape(21,21)
    
    contour_lines = [0.8,1.6,2.4,3.2,4,5,6,7,8,9,10,12,15,20]
    
    CS3 = axarr[1].contour(sig_plot,eps_plot,RMSPsat_MBAR,[0.8,1.6],label='MBAR with single reference',colors='g')
    CS4 = axarr[1].contour(sig_plot,eps_plot,RMSPsat_PCFR,[0.8,1.6],label='PCFR with single reference',colors='c')
    CS1 = axarr[1].contour(sig_plot,eps_plot,RMSPsat,contour_lines,label='Direct Simulation',colors='r')
    CS2 = axarr[1].contour(sig_plot,eps_plot,RMSPsat_MBAR_9refs,contour_lines,label='MBAR with multiple references',colors='b')
    #axarr[1].clabel(CS2, inline=1,fontsize=10,colors='w',fmt='%1.1f')
    axarr[1].clabel(CS4, inline=1,fontsize=10,colors='c',fmt='%1.1f')
    axarr[1].clabel(CS3, inline=1,fontsize=10,colors='g',fmt='%1.1f')
    axarr[1].clabel(CS1, inline=0,fontsize=10,colors='k',fmt='%1.1f')
    axarr[1].set_xlabel(r'$\sigma$ (nm)')
    axarr[1].set_ylabel(r'$\epsilon$ (K)')
    axarr[1].set_title(r'RMS of $P_v  \left(bar\right)$')
    axarr[1].plot([],[],'r',label='Direct Simulation')
    axarr[1].plot([],[],'b',label='MBAR multiple references')
    axarr[1].plot([],[],'g',label='MBAR single reference')
    axarr[1].plot([],[],'c',label='PCFR single reference')
    axarr[1].plot(sig_PCFs,eps_PCFs,'mx',label='References')
    axarr[1].legend()
        
    f.savefig('RMS_rhol_Psat_comparison.pdf')
    
def embed_parity_residual_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,prop_hat4,Neff,dprop_direct,dprop_hat1,dprop):
    parity = np.array([np.min(np.array([np.min(prop_direct),np.min(prop_hat1),np.min(prop_hat2)])),np.max(np.array([np.max(prop_direct),np.max(prop_hat1),np.max(prop_hat2)]))])

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
        xmin = -7000
        xmax = -500
        ymin = -7000
        ymax = -500
        embed_comparison = [0.58,0.16,0.31,0.31]
        embed_MBAR = [0.45,0.13,0.28,0.28]
    elif prop == 'P':
        units = '(bar)'
        title = 'Pressure'
        dev1 = abs_dev1
        dev2 = abs_dev2
        dev3 = abs_dev3
        dev4 = abs_dev4
        dev_type = 'Absolute'
        xmin = np.min(parity) 
        xmax = np.max(parity)
        ymin = np.min(parity)
        ymax = np.max(parity)
        embed_comparison = [0.58,0.16,0.31,0.31]
        embed_MBAR = [0.45,0.13,0.28,0.28]
    elif prop == 'Z':
        units = ''
        title = 'Compressibility Factor'
        dev1 = abs_dev1
        dev2 = abs_dev2
        dev3 = abs_dev3
        dev4 = abs_dev4
        dev_type = 'Absolute'
        xmin = -4 
        xmax = 10
        ymin = -8
        ymax = 1.01*np.max(parity)
        embed_comparison = [0.58,0.14,0.31,0.31]
        embed_MBAR = [0.43,0.16,0.3,0.3]
    elif prop == 'Pdep':
        units = '(bar)'
        title = 'Pressure - Ideal Gas'
        dev1 = abs_dev1
        dev2 = abs_dev2
        dev3 = abs_dev3
        dev4 = abs_dev4
        dev_type = 'Absolute'
        
#    f = plt.figure(figsize=(8,6))
#
#    plt.plot(prop_direct,prop_hat3,'b.',alpha=0.2)    
#    plt.plot(prop_direct,prop_hat1,'r.',alpha=0.2)
#    plt.plot(prop_direct,prop_hat2,'g.',alpha=0.2)
#    plt.plot(prop_direct,prop_hat4,'c.',alpha=0.2)
#    #plt.plot(prop_direct,prop_hat3,'bx',label='MBAR, Neff > 10')
#    plt.plot(parity,parity,'k',label='Parity')
#    plt.xlabel('Direct Simulation '+units)
#    plt.ylabel('Predicted '+units)
#    plt.xlim([xmin,xmax])
#    plt.ylim([ymin,ymax])
#    plt.title(title)
#    plt.plot([],[],'bo',label='Constant PCF')
#    plt.plot([],[],'ro',label='MBAR')
#    plt.plot([],[],'go',label='PCFR')
#    plt.plot([],[],'co',label='Recommended')
#    plt.legend()
#    a = plt.axes(embed_comparison)
#    plt.plot(prop_direct,dev3,'b.',label='Constant PCF',alpha=0.2)   
#    plt.plot(prop_direct,dev1,'r.',label='MBAR',alpha=0.2)
#    plt.plot(prop_direct,dev2,'g.',label='PCFR',alpha=0.2)
#    plt.plot(prop_direct,dev4,'c.',label='Recommended',alpha=0.2)
#    plt.xticks([])
#    #plt.xlabel('Direct Simulation '+units)
#    if dev_type == 'Percent':
#        plt.ylabel(dev_type+' Deviation ')
#    else:
#        plt.ylabel(dev_type+' Deviation '+units)
#    plt.show()
#    
#    f.savefig('Parity_residual_comparison_'+prop+'.pdf')
#    
#    f = plt.figure(figsize=(8,6))
#
#    p = plt.scatter(prop_direct[Neff.argsort()],prop_hat1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR') 
#    #p = plt.scatter(prop_direct[Neff.argsort()],prop_hat1[Neff.argsort()],c=Neff[Neff.argsort()],cmap='cool',label='MBAR',norm=col.LogNorm())
#    plt.plot(parity,parity,'k',label='Parity')
#    plt.xlabel('Direct Simulation '+units)
#    plt.ylabel('Predicted with MBAR '+units)
#    plt.xlim([xmin,xmax])
#    plt.ylim([ymin,ymax])
#    plt.title(title)
#    #plt.legend()
#    cb = plt.colorbar(p)
#    cb.set_label('log$_{10}(N_{eff})$')
#    
#    a = plt.axes(embed_MBAR)
#    plt.scatter(prop_direct[Neff.argsort()],dev1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR')
#    plt.xticks([])
#    #plt.xlabel('Direct Simulation '+units)
#    if dev_type == 'Percent':
#        plt.ylabel(dev_type+' Deviation ')
#    else:
#        plt.ylabel(dev_type+' Deviation '+units)
#    
#    plt.show()
#    
#    f.savefig('Parity_residual_MBAR_'+prop+'.pdf')
    
    f, axarr = plt.subplots(nrows=2,ncols=1,figsize=(8,12))
    
    axarr[0].plot([],[],'bo',label='Constant PCF')
    axarr[0].plot([],[],'ro',label='MBAR')
    axarr[0].plot([],[],'go',label='PCFR')
    axarr[0].plot([],[],'co',label='Recommended')
    axarr[0].plot(parity,parity,'k',label='Parity')
    axarr[0].plot(prop_direct,prop_hat3,'b.',alpha=0.2)    
    axarr[0].plot(prop_direct,prop_hat1,'r.',alpha=0.2)
    axarr[0].plot(prop_direct,prop_hat2,'g.',alpha=0.2)
    axarr[0].plot(prop_direct,prop_hat4,'c.',alpha=0.2)
    axarr[0].plot(parity,parity,'k',label='Parity')
    axarr[0].set_xlabel('Direct Simulation '+units)
    axarr[0].set_ylabel('Predicted '+units)
    axarr[0].set_xlim([xmin,xmax])
    axarr[0].set_ylim([ymin,ymax])
    axarr[0].set_title(title)
    axarr[0].legend(['Constant PCF','MBAR','PCFR','Recommended','Parity'])
    a = inset_axes(axarr[0],width=2.0,height=2.0,loc=4)
    a.plot(prop_direct,dev3,'b.',label='Constant PCF',alpha=0.2)   
    a.plot(prop_direct,dev1,'r.',label='MBAR',alpha=0.2)
    a.plot(prop_direct,dev2,'g.',label='PCFR',alpha=0.2)
    a.plot(prop_direct,dev4,'c.',label='Recommended',alpha=0.2)
    a.set_xticks([])
    #plt.xlabel('Direct Simulation '+units)
    if dev_type == 'Percent':
        a.set_ylabel(dev_type+' Deviation ')
    else:
        a.set_ylabel(dev_type+' Deviation '+units)
        
    p = axarr[1].scatter(prop_direct[Neff.argsort()],prop_hat1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR')    
    axarr[1].plot(parity,parity,'k',label='Parity')
    axarr[1].set_xlabel('Direct Simulation '+units)
    axarr[1].set_ylabel('Predicted with MBAR '+units)
    axarr[1].set_xlim([xmin,xmax])
    axarr[1].set_ylim([ymin,ymax])
    #axarr[1].set_title(title)
    cb = plt.colorbar(p,ax=axarr[1],pad=0.02)
    cb.set_label('log$_{10}(N_{eff})$')
    #cax = cb.ax
    #cax.set_position([0.5,0.5,0.5,0.5])
    a = inset_axes(axarr[1],width=2.0,height=2.0,loc=4)
    a.scatter(prop_direct[Neff.argsort()],dev1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR')   
    a.set_xticks([])
    #plt.xlabel('Direct Simulation '+units)
    if dev_type == 'Percent':
        a.set_ylabel(dev_type+' Deviation ')
    else:
        a.set_ylabel(dev_type+' Deviation '+units)
    
    f.savefig('Combined_parity_residual_comparison_MBAR'+prop+'.pdf')
    
def multi_prop_plot(U_direct,U_MBAR,U_PCFR,U_W1,U_rec,Z_direct,Z_MBAR,Z_PCFR,Z_W1,Z_rec,Neff,fpathroot):
    
    abs_dev = lambda hat, direct: (hat - direct)
    rel_dev = lambda hat, direct: abs_dev(hat,direct)/direct * 100.
                                         
    jfig = 0  
                                                                
    f, axarr = plt.subplots(nrows=2,ncols=2,figsize=(16,12)) 

    if fpathroot == 'parameter_space_LJ/':

        plt.text(-26.5,32,'a)') 
        plt.text(-26.5,10,'b)')
        plt.text(-6,32,'c)')
        plt.text(-6,10,'d)') 

    elif fpathroot == 'parameter_space_Mie16/':

        plt.text(-30,63,'a)') 
        plt.text(-30,20,'b)')
        plt.text(-6,63,'c)')
        plt.text(-6,20,'d)')                                
                                             
    for prop in ['U','Z']:
        
        ifig = 0
        
        if prop == 'U':
            prop_direct = U_direct
            prop_hat1 = U_MBAR
            prop_hat2 = U_PCFR
            prop_hat3 = U_W1
            prop_hat4 = U_rec

        elif prop == 'Z':
            prop_direct = Z_direct
            prop_hat1 = Z_MBAR
            prop_hat2 = Z_PCFR
            prop_hat3 = Z_W1
            prop_hat4 = Z_rec 

        parity = np.array([np.min(np.array([np.min(prop_direct),np.min(prop_hat1),np.min(prop_hat2)])),np.max(np.array([np.max(prop_direct),np.max(prop_hat1),np.max(prop_hat2)]))])
                                         
        abs_dev1 = abs_dev(prop_hat1, prop_direct)
        abs_dev2 = abs_dev(prop_hat2,prop_direct)
        abs_dev3 = abs_dev(prop_hat3,prop_direct)
        abs_dev4 = abs_dev(prop_hat4,prop_direct)
               
        rel_dev1 = rel_dev(prop_hat1,prop_direct)
        rel_dev2 = rel_dev(prop_hat2,prop_direct)
        rel_dev3 = rel_dev(prop_hat3,prop_direct)
        rel_dev4 = rel_dev(prop_hat4,prop_direct)
                   
        if prop == 'U':
            units = '(kJ/mol)'
            title = 'Residual Energy'
            dev1 = rel_dev1
            dev2 = rel_dev2
            dev3 = rel_dev3
            dev4 = rel_dev4
            dev_type = 'Percent'
            xmin = -7000
            xmax = -500
            ymin = -7000
            ymax = -500
            embed_comparison = [0.58,0.16,0.31,0.31]
            embed_MBAR = [0.45,0.13,0.28,0.28]
        elif prop == 'Z':
            units = ''
            title = 'Compressibility Factor'
            dev1 = abs_dev1
            dev2 = abs_dev2
            dev3 = abs_dev3
            dev4 = abs_dev4
            dev_type = 'Absolute'
            if fpathroot == 'parameter_space_LJ/':
                
                xmin = -4 
                xmax = 10
                ymin = -8
                ymax = 1.01*np.max(parity)
                
            elif fpathroot == 'parameter_space_Mie16/':
                
                xmin = -4 
                xmax = 12
                ymin = -15
                ymax = 1.01*np.max(parity)
                
            embed_comparison = [0.58,0.14,0.31,0.31]
            embed_MBAR = [0.43,0.16,0.3,0.3]
        
        axarr[ifig,jfig].plot([],[],'bo',label='Constant PCF')
        axarr[ifig,jfig].plot([],[],'ro',label='MBAR')
        axarr[ifig,jfig].plot([],[],'go',label='PCFR')
        axarr[ifig,jfig].plot([],[],'co',label='Recommended')
        axarr[ifig,jfig].plot(parity,parity,'k',label='Parity')
        axarr[ifig,jfig].plot(prop_direct,prop_hat3,'b.',alpha=0.2)    
        axarr[ifig,jfig].plot(prop_direct,prop_hat1,'r.',alpha=0.2)
        axarr[ifig,jfig].plot(prop_direct,prop_hat2,'g.',alpha=0.2)
        axarr[ifig,jfig].plot(prop_direct,prop_hat4,'c.',alpha=0.2)
        axarr[ifig,jfig].plot(parity,parity,'k',label='Parity')
        axarr[ifig,jfig].set_xlabel('Direct Simulation '+units)
        axarr[ifig,jfig].set_ylabel('Predicted '+units)
        axarr[ifig,jfig].set_xlim([xmin,xmax])
        axarr[ifig,jfig].set_ylim([ymin,ymax])
        axarr[ifig,jfig].set_title(title)
        #axarr[ifig,jfig].text(2,0.65,panels[ifig,jfig])
        if ifig == 0 and jfig == 0:
            axarr[ifig,jfig].legend(['Constant PCF','MBAR','PCFR','Recommended','Parity'])
        a = inset_axes(axarr[ifig,jfig],width=2.0,height=2.0,loc=4)
        a.plot(prop_direct,dev3,'b.',label='Constant PCF',alpha=0.2)   
        a.plot(prop_direct,dev1,'r.',label='MBAR',alpha=0.2)
        a.plot(prop_direct,dev2,'g.',label='PCFR',alpha=0.2)
        a.plot(prop_direct,dev4,'c.',label='Recommended',alpha=0.2)
        a.set_xticks([])
        #plt.xlabel('Direct Simulation '+units)
        if dev_type == 'Percent':
            a.set_ylabel(dev_type+' Deviation ')
        else:
            a.set_ylabel(dev_type+' Deviation '+units)
            
        ifig += 1
            
        p = axarr[ifig,jfig].scatter(prop_direct[Neff.argsort()],prop_hat1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR')    
        axarr[ifig,jfig].plot(parity,parity,'k',label='Parity')
        axarr[ifig,jfig].set_xlabel('Direct Simulation '+units)
        axarr[ifig,jfig].set_ylabel('Predicted with MBAR '+units)
        axarr[ifig,jfig].set_xlim([xmin,xmax])
        axarr[ifig,jfig].set_ylim([ymin,ymax])
        #axarr[1].set_title(title)
        
        if jfig == 1:
        
            cb = plt.colorbar(p,ax=axarr[ifig,jfig],pad=0.02)
            cb.set_label('log$_{10}(N_{eff})$')
        #cax = cb.ax
        #cax.set_position([0.5,0.5,0.5,0.5])
        a = inset_axes(axarr[ifig,jfig],width=2.0,height=2.0,loc=4)
        a.scatter(prop_direct[Neff.argsort()],dev1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR')   
        a.set_xticks([])
        #plt.xlabel('Direct Simulation '+units)
        if dev_type == 'Percent':
            a.set_ylabel(dev_type+' Deviation ')
        else:
            a.set_ylabel(dev_type+' Deviation '+units)
            
        jfig += 1
        
    f.savefig('Multi_prop_combined.pdf')
    
def multi_prop_multi_ref_plot(U_direct,U_MBAR,Z_direct,Z_MBAR,Neff,fpathroot):
    
    abs_dev = lambda hat, direct: (hat - direct)
    rel_dev = lambda hat, direct: abs_dev(hat,direct)/direct * 100.
                                         
    ifig = 0                                     
                                                                
    f, axarr = plt.subplots(nrows=1,ncols=2,figsize=(16,6)) 

    if fpathroot == 'parameter_space_LJ/':

        plt.text(-23.5,7.5,'a)') 
        plt.text(-5.5,7.5,'b)')

    elif fpathroot == 'parameter_space_Mie16/':

        plt.text(-30,63,'a)') 
        plt.text(-30,20,'b)')                            
                                             
    for prop in ['U','Z']:
        
        if prop == 'U':
            prop_direct = U_direct
            prop_hat1 = U_MBAR
            
        elif prop == 'Z':
            prop_direct = Z_direct
            prop_hat1 = Z_MBAR

        parity = np.array([np.min(np.array([np.min(prop_direct),np.min(prop_hat1)])),np.max(np.array([np.max(prop_direct),np.max(prop_hat1)]))])
                                         
        abs_dev1 = abs_dev(prop_hat1, prop_direct)
               
        rel_dev1 = rel_dev(prop_hat1,prop_direct)
                   
        if prop == 'U':
            units = '(kJ/mol)'
            title = 'Residual Energy'
            dev1 = rel_dev1
            dev_type = 'Percent'
            xmin = -7000
            xmax = -500
            ymin = -7000
            ymax = -500
        elif prop == 'Z':
            units = ''
            title = 'Compressibility Factor'
            dev1 = abs_dev1
            dev_type = 'Absolute'
            if fpathroot == 'parameter_space_LJ/':
                
                xmin = -4 
                xmax = 8
                ymin = -4
                ymax = 8
                
            elif fpathroot == 'parameter_space_Mie16/':
                
                xmin = -4 
                xmax = 12
                ymin = -15
                ymax = 1.01*np.max(parity)
                            
        p = axarr[ifig].scatter(prop_direct[Neff.argsort()],prop_hat1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR')    
        axarr[ifig].plot(parity,parity,'k',label='Parity')
        axarr[ifig].set_xlabel('Direct Simulation '+units)
        axarr[ifig].set_ylabel('Predicted with MBAR '+units)
        axarr[ifig].set_xlim([xmin,xmax])
        axarr[ifig].set_ylim([ymin,ymax])
        #axarr[1].set_title(title)
        
        if ifig == 1:
        
            cb = plt.colorbar(p,ax=axarr[ifig],pad=0.02)
            cb.set_label('log$_{10}(N_{eff})$')
        #cax = cb.ax
        #cax.set_position([0.5,0.5,0.5,0.5])
        
        if prop == 'U':
            a = inset_axes(axarr[ifig],width=2.3,height=2.3,loc=4)       
        elif prop == 'Z':        
            a = inset_axes(axarr[ifig],width=2.0,height=2.0,loc=4)
        a.scatter(prop_direct[Neff.argsort()],dev1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR')   
        a.set_xticks([])
        #plt.xlabel('Direct Simulation '+units)
        if dev_type == 'Percent':
            a.set_ylabel(dev_type+' Deviation ')
        else:
            a.set_ylabel(dev_type+' Deviation '+units)
            
        ifig += 1
        
    f.savefig('Multi_prop_MBAR_multi_ref_combined.pdf')
    
def multi_ref_LJ_Mie_plot(U_direct_LJ,U_MBAR_LJ,U_direct_Mie,U_MBAR_Mie,Z_direct_LJ,Z_MBAR_LJ,Z_direct_Mie,Z_MBAR_Mie,Neff_LJ,Neff_Mie):

    abs_dev = lambda hat, direct: (hat - direct)
    rel_dev = lambda hat, direct: abs_dev(hat,direct)/direct * 100.
    
    dev1 = rel_dev(U_MBAR_LJ,U_direct_LJ)
    dev2 = abs_dev(Z_MBAR_LJ,Z_direct_LJ)
    dev3 = rel_dev(U_MBAR_Mie,U_direct_Mie)
    dev4 = abs_dev(Z_MBAR_Mie,Z_direct_Mie)                                     
                                         
    parity_1 = np.array([np.min(np.array([np.min(U_direct_LJ),np.min(U_MBAR_LJ)])),np.max(np.array([np.max(U_direct_LJ),np.max(U_MBAR_LJ)]))])
    parity_2 = np.array([np.min(np.array([np.min(Z_direct_LJ),np.min(Z_MBAR_LJ)])),np.max(np.array([np.max(Z_direct_LJ),np.max(Z_MBAR_LJ)]))])
    parity_3 = np.array([np.min(np.array([np.min(U_direct_Mie),np.min(U_MBAR_Mie)])),np.max(np.array([np.max(U_direct_Mie),np.max(U_MBAR_Mie)]))])
    parity_4 = np.array([np.min(np.array([np.min(Z_direct_Mie),np.min(Z_MBAR_Mie)])),np.max(np.array([np.max(Z_direct_Mie),np.max(Z_MBAR_Mie)]))])      
    
    my_figure = plt.figure(figsize=(16,12))
    subplot_1 = my_figure.add_subplot(2,2,1)
    subplot_1.scatter(U_direct_LJ[Neff_LJ.argsort()],U_MBAR_LJ[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',label='MBAR')
    subplot_1.plot(parity_1,parity_1,'k',label='Parity')
    subplot_1.set_xlabel('Direct Simulation (kJ/mol) ')
    subplot_1.set_ylabel('Predicted with MBAR (kJ/mol)')
    subplot_1.set_xlim([-7000,-500])
    subplot_1.set_ylim([-7000,-500])
    
    a = inset_axes(subplot_1,width=2.2,height=2.2,loc=4)          
    a.scatter(U_direct_LJ[Neff_LJ.argsort()],dev1[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',label='MBAR')   
    a.plot(parity_1,[0,0],'k--')
    a.set_xticks([])
    a.set_ylabel('Percent Deviation')
    
    subplot_2 = my_figure.add_subplot(2,2,2)
    p = subplot_2.scatter(Z_direct_LJ[Neff_LJ.argsort()],Z_MBAR_LJ[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',label='MBAR')
    subplot_2.plot(parity_2,parity_2,'k',label='Parity')
    subplot_2.set_xlabel('Direct Simulation')
    subplot_2.set_ylabel('Predicted with MBAR')
    subplot_2.set_xlim([-4,8])
    subplot_2.set_ylim([-6,8])
    
    a = inset_axes(subplot_2,width=2.2,height=2.2,loc=4)          
    a.scatter(Z_direct_LJ[Neff_LJ.argsort()],dev2[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',label='MBAR')   
    a.plot(parity_2,[0,0],'k--')
    a.set_xticks([])
    a.set_ylabel('Absolute Deviation')
    
    cb = plt.colorbar(p,ax=subplot_2,pad=0.02)
    cb.set_label('log$_{10}(N_{eff})$')
    
    subplot_3 = my_figure.add_subplot(2,2,3)
    subplot_3.scatter(U_direct_Mie[Neff_Mie.argsort()],U_MBAR_Mie[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',label='MBAR')
    subplot_3.plot(parity_3,parity_3,'k',label='Parity')
    subplot_3.set_xlabel('Direct Simulation (kJ/mol) ')
    subplot_3.set_ylabel('Predicted with MBAR (kJ/mol)')
    subplot_3.set_xlim([-7000,-500])
    subplot_3.set_ylim([-7000,-500])
    
    a = inset_axes(subplot_3,width=2.2,height=2.2,loc=4)          
    a.scatter(U_direct_Mie[Neff_Mie.argsort()],dev3[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',label='MBAR')   
    a.plot(parity_3,[0,0],'k--')
    a.set_xticks([])
    a.set_ylabel('Percent Deviation')
    
    subplot_4 = my_figure.add_subplot(2,2,4)
    p = subplot_4.scatter(Z_direct_Mie[Neff_Mie.argsort()],Z_MBAR_Mie[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',label='MBAR')
    subplot_4.plot(parity_4,parity_4,'k',label='Parity')
    subplot_4.set_xlabel('Direct Simulation')
    subplot_4.set_ylabel('Predicted with MBAR')
    subplot_4.set_xlim([-4,10])
    subplot_4.set_ylim([-14,14])
    
    a = inset_axes(subplot_4,width=2.2,height=2.2,loc=4)          
    a.scatter(Z_direct_Mie[Neff_Mie.argsort()],dev4[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',label='MBAR')   
    a.plot(parity_4,[0,0],'k--')
    a.set_xticks([])
    a.set_ylabel('Absolute Deviation')
    
    cb = plt.colorbar(p,ax=subplot_4,pad=0.02)
    cb.set_label('log$_{10}(N_{eff})$')
    
    subplot_1.text(-7900, -800, 'a)')
    subplot_2.text(-5.6, 7.5, 'c)')
    subplot_3.text(-7900, -800, 'b)')
    subplot_4.text(-5.9, 13, 'd)')
        
    my_figure.savefig('MBAR_multi_ref_LJ_Mie.pdf')
            
def main():
    
    fpathroot = 'parameter_space_LJ/'
    
    eps_all, sig_all, eps_matrix, sig_matrix = get_parameter_sets(fpathroot)
    
#    RMS_contours(eps_all,sig_all,fpathroot)
#    RMS_contours_combined(eps_all,sig_all,fpathroot)
#    
#    return
    
    for model_type in [reference,'Direct_simulation', 'MBAR_ref8', 'PCFR_ref0','Constant_']:
        if model_type == 'TraPPE' or model_type == 'Potoff':
            U_ref, dU_ref, P_ref, dP_ref, Z_ref, dZ_ref, Neff_ref = compile_data(model_type,fpathroot)
            # Now I call a function that should calculate the error in the proper manner
            U_error, P_error = PCFR_error(U_ref,P_ref,model_type)
        elif model_type == 'Direct_simulation':
            U_direct, dU_direct, P_direct, dP_direct, Z_direct, dZ_direct, Neff_direct = compile_data(model_type,fpathroot)
            U_direct_Mie, dU_direct_Mie, P_direct_Mie, dP_direct_Mie, Z_direct_Mie, dZ_direct_Mie, Neff_direct_Mie = compile_data(model_type,'parameter_space_Mie16/')
        elif model_type == 'MBAR_ref0' or model_type == 'MBAR_ref1' or model_type == 'MBAR_ref8':
            U_MBAR, dU_MBAR, P_MBAR, dP_MBAR, Z_MBAR, dZ_MBAR, Neff_MBAR = compile_data(model_type,fpathroot)
            U_MBAR_Mie, dU_MBAR_Mie, P_MBAR_Mie, dP_MBAR_Mie, Z_MBAR_Mie, dZ_MBAR_Mie, Neff_MBAR_Mie = compile_data(model_type,'parameter_space_Mie16/')
        elif model_type == 'PCFR_ref0':
            U_PCFR, dU_PCFR, P_PCFR, dP_PCFR, Z_PCFR, dZ_PCFR, Neff_PCFR = compile_data(model_type,fpathroot)
        elif model_type == 'PCFR_mult_ref':
            U_PCFR, dU_PCFR, P_PCFR, dP_PCFR, Z_PCFR, dZ_PCFR, Neff_PCFR = merge_PCFR(sig_all,fpathroot)
        elif model_type == 'Constant_':
            U_W1, dU_W1, P_W1, dP_W1, Z_W1, dZ_W1, Neff_W1 = compile_data(model_type,fpathroot)

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
        #embed_parity_residual_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,prop_hat4,Neff,dprop_direct,dprop_MBAR,dprop)
        #parity_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,prop_hat4,Neff,dprop_direct,dprop_MBAR)
        #residual_plot(prop,prop_direct,prop_hat1,prop_hat2,prop_hat3,prop_hat4,Neff,dprop)
        #uncertainty_check(prop_direct,prop_hat1,dprop_direct,dprop_MBAR,Neff)
        
    #contour_plot('U',eps_all,sig_all,U_direct,U_MBAR,U_PCFR,U_W1,U_rec)
    #contour_plot('P',eps_all,sig_all,P_direct,P_MBAR,P_PCFR,P_W1,P_rec)
    #contour_plot('Z',eps_all,sig_all,Z_direct,Z_MBAR,Z_PCFR,Z_W1,Z_rec)
    #contour_plot('Pdep',eps_all,sig_all,Pdep_direct,Pdep_MBAR,Pdep_PCFR)
    
    #multi_prop_plot(U_direct[mask],U_MBAR[mask],U_PCFR[mask],U_W1[mask],U_rec[mask],Z_direct[mask],Z_MBAR[mask],Z_PCFR[mask],Z_W1[mask],Z_rec[mask],Neff,fpathroot)
    
    #multi_prop_multi_ref_plot(U_direct[mask],U_MBAR[mask],Z_direct[mask],Z_MBAR[mask],Neff,fpathroot)
    
    multi_ref_LJ_Mie_plot(U_direct[mask],U_MBAR[mask],U_direct_Mie[mask],U_MBAR_Mie[mask],Z_direct[mask],Z_MBAR[mask],Z_direct_Mie[mask],Z_MBAR_Mie[mask],Neff_MBAR[mask],Neff_MBAR_Mie[mask])
    
    #box_bar_state_plots(Neff_MBAR,Neff_min,Neff_small,mask_MBAR,mask_poor)
    #contours_Neff(Neff_MBAR,sig_all,eps_all)
    
if __name__ == '__main__':
    
    main()