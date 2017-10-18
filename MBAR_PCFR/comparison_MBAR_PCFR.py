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
import matplotlib.colors as mpl
import matplotlib.mlab as mlab

reference = 'TraPPE'  

nReruns = 441
nStates = 19

T_rho = np.loadtxt('Temp_rho.txt')

sig_PCFs = [0.375, 0.370, 0.3725, 0.365, 0.3675, 0.3775, 0.380, 0.3825, 0.385]
eps_PCFs = [98]*9
           
#sig_PCFs = [0.375, 0.370, 0.3725, 0.3775, 0.380, 0.3825, 0.385, 0.3875, 0.390]
#eps_PCFs = [118]*9
           
font = {'size' : '14'}
plt.rc('font',**font)

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
            if model_type == 'Direct_simulation' or model_type == 'MBAR_ref0' or fpathroot == 'parameter_space_LJ/' or (model_type == 'PCFR_ref0' and reference == 'TraPPE') or model_type == 'Lam15/MBAR_ref0' or model_type == 'Lam17/MBAR_ref0' or model_type == 'Lam14/MBAR_ref0' or model_type == 'Lam18/MBAR_ref0' or model_type == 'Lam13/MBAR_ref0' or model_type == 'Lam12/MBAR_ref0' or model_type == 'Lam12/MBAR_ref8' or model_type == 'Lam12/MBAR_ref11':
                iRerun += 1
            if model_type == 'MBAR_ref1':
                iRerun += 1
            if model_type == 'MBAR_ref8' or model_type == 'Lam12/MBAR_ref8':
                iRerun += 8
            if (model_type == 'MBAR_ref8' and fpathroot == 'parameter_space_Mie16/'):
                iRerun += 1
            if model_type == 'Lam12/MBAR_ref11':
                iRerun += 11
            if (model_type == 'Zeroth_re' and fpathroot == 'parameter_space_LJ/'):
                iRerun -= 1
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
    
    return U_compiled/400., dU_compiled/400., P_compiled, dP_compiled, Z_compiled, dZ_compiled, Neff_compiled

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

def contour_combined_plot(eps_all,sig_all,U_direct,Z_direct,U_MBAR,Z_MBAR,U_PCFR,Z_PCFR):
        
    SSE_1 = np.sum((U_direct-U_MBAR)**2,axis=0).reshape(21,21)
    SSE_2 = np.sum((U_direct-U_PCFR)**2,axis=0).reshape(21,21)
    SSE_3 = np.sum((Z_direct-Z_MBAR)**2,axis=0).reshape(21,21)
    SSE_4 = np.sum((Z_direct-Z_PCFR)**2,axis=0).reshape(21,21)
    
    RMS_1 = np.sqrt(SSE_1/len(U_direct))
    RMS_2 = np.sqrt(SSE_2/len(U_direct))
    RMS_3 = np.sqrt(SSE_3/len(Z_direct))
    RMS_4 = np.sqrt(SSE_4/len(Z_direct))
    
    eps_plot = np.unique(eps_all)
    sig_plot = np.unique(sig_all)
                    
    my_figure = plt.figure(figsize=(8,12))
    subplot_1 = my_figure.add_subplot(2,1,1)
    
    contour_lines = [25,50,75,100,125,150,200,250,300,400,500,600,700,800]
    contour_lines = [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00]
    fmt_prop = '%1.2f'
    
    CS1 = subplot_1.contour(sig_plot,eps_plot,RMS_1,contour_lines,colors='r')
    CS2 = subplot_1.contour(sig_plot,eps_plot,RMS_2,contour_lines,colors='g')
    plt.clabel(CS1, inline=1,fontsize=10,colors='r',fmt=fmt_prop)
    plt.clabel(CS2, inline=1,fontsize=10,colors='g',fmt=fmt_prop)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.xticks([0.365,0.370,0.375,0.380,0.385])
    plt.yticks([88,93,98,103,108])
    plt.plot([],[],'r',label='MBAR')
    plt.plot([],[],'g',label='PCFR')
    plt.plot(0.375,98,'mx',label='Reference')
    plt.title('RMS of $U_{dep}$ (kJ/mol)')
    plt.legend()
    
    subplot_2 = my_figure.add_subplot(2,1,2)
    
    contour_lines = [0.1, 0.5, 1., 1.5, 2., 2.5,3.,3.5]
    fmt_prop = '%1.1f'
    
    CS3 = subplot_2.contour(sig_plot,eps_plot,RMS_3,contour_lines,colors='r')
    CS4 = subplot_2.contour(sig_plot,eps_plot,RMS_4,contour_lines,colors='g')
    plt.clabel(CS3, inline=1,fontsize=10,colors='r',fmt=fmt_prop)
    plt.clabel(CS4, inline=1,fontsize=10,colors='g',fmt=fmt_prop)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.xticks([0.365,0.370,0.375,0.380,0.385])
    plt.yticks([88,93,98,103,108])
    plt.plot([],[],'r',label='MBAR')
    plt.plot([],[],'g',label='PCFR')
    plt.plot(0.375,98,'mx',label='Reference')
    plt.title('RMS of $Z$')
    plt.legend()

    plt.tight_layout(pad=0.2,rect=[0,0,1,1])
    
    subplot_1.text(0.363, 107, 'a)')
    subplot_2.text(0.363, 107, 'b)')  
    
    plt.show()
    
    my_figure.savefig('Contour_comparison_combined.pdf')

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
    
    cov_factor = dev_MBAR / u_total
    sign_cov_factor = (prop_MBAR - prop_direct)/u_total
    
    plt.plot(Neff_MBAR,cov_factor)
    plt.show()
    
    Neff_min = 50.
    data_plot = sign_cov_factor[Neff_MBAR>Neff_min]
    mean = np.mean(data_plot)
    var = np.var(data_plot)
    sigma = np.sqrt(var)
    
    result = plt.hist(data_plot,bins=30,normed=True)
  
    x = np.linspace(min(result[1]),max(result[1]),1000)
    dx = result[1][1] - result[1][0]
    scale = len(result[1])*dx
    
    plt.plot(x,mlab.normpdf(x,mean,sigma)*scale)
        
    plt.xlabel('Coverage Factor')
    plt.ylabel('Density')
    plt.title(r'Minimum N$_{eff}$ = '+str(Neff_min))
    plt.show()
    
    Nmin_array = np.linspace(1,1001.,1001)
    zscores = []
    skews = []
    ptest = []
    
    for iN, Nmin in enumerate(Nmin_array):
    
        sample = sign_cov_factor[Neff_MBAR>=Nmin]
        z, p = stats.skewtest(sample)[0], stats.skewtest(sample)[1]
        sk = stats.skew(sample)
        zscores = np.append(zscores,z)
        skews = np.append(skews,sk)
        ptest = np.append(ptest,p)
        
    plt.plot(Nmin_array,zscores)
    plt.xlabel('Minimum Neff')
    plt.ylabel('Z-score for normal')
    plt.ylim([-1,1])
    plt.show()
    print(zscores)    
    
    plt.plot(Nmin_array,ptest)
    plt.xlabel('Minimum Neff')
    plt.ylabel('P-test for normal')
    plt.plot([0,1000.],[0.05,0.05])
    plt.show()
    print(ptest)
    
    plt.plot(Nmin_array,skews)
    plt.xlabel('Minimum Neff')
    plt.ylabel('Skew')
    plt.show()
    print(skews)

    return
    
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
        
    U_PCFR = UPZ_PCFR[:,0]/400.
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
    contour_lines = [50,100,200,400,800]
 
    RMS_U = np.loadtxt(fpathroot+'Direct_simulation_rr_RMS_U_all').reshape(21,21)
    CS1 = plt.contour(sig_plot,eps_plot,RMS_U,contour_lines,label='Direct Simulation',colors='r')
    plt.clabel(CS1, inline=1,fontsize=10,colors='k',fmt='%1.1f')
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.title(r'RMS of U')
    plt.show()
    
    return
    
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
    
    f, axarr = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
    
    #For presentation only
    
    CS3 = axarr.contour(sig_plot,eps_plot,RMSrhoL_MBAR,[5,10,20,30],label='MBAR with single reference',colors='g')
#    CS4 = axarr.contour(sig_plot,eps_plot,RMSrhoL_PCFR,[5,10],label='PCFR with single reference',colors='c')
    CS1 = axarr.contour(sig_plot,eps_plot,RMSrhoL,contour_lines,label='Direct Simulation',colors='r')
#    CS2 = axarr.contour(sig_plot,eps_plot,RMSrhoL_MBAR_9refs,contour_lines,label='MBAR with multiple references',colors='b')
#    axarr.clabel(CS2, inline=1,fontsize=10,colors='w',fmt='%1.0f')
#    axarr.clabel(CS4, inline=1,fontsize=10,colors='c',fmt='%1.0f',manual=[(0,-10),(5,10)])
    axarr.clabel(CS3, inline=1,fontsize=10,colors='g',fmt='%1.0f')
    axarr.clabel(CS1, inline=1,fontsize=10,colors='k',fmt='%1.0f')
    axarr.set_xlabel(r'$\sigma$ (nm)')
    axarr.set_ylabel(r'$\epsilon$ (K)')
    axarr.set_title(r'RMS of $\rho_l$ (kg/m$^3$)')
    axarr.set_yticks([88,93,98,103,108])
    axarr.set_xticks([0.365,0.370,0.375,0.380,0.385])
    axarr.plot([],[],'r',label='Direct Simulation')
#    axarr.plot([],[],'b',label='MBAR multiple references')
    axarr.plot([],[],'g',label='MBAR single reference')
#    axarr.plot([],[],'c',label='PCFR single reference')
    axarr.plot(sig_PCFs[0],eps_PCFs[0],'mx',label='Reference, TraPPE-UA')
#    axarr.plot(sig_PCFs,eps_PCFs,'mx',label='Multiple references')
    axarr.legend(loc=2)

    f, axarr = plt.subplots(nrows=2,ncols=1,figsize=(8,12))
    
    plt.tight_layout(pad=3,rect=[0,0,1,1])
    
    plt.text(0.3627,131,'a)') 
    plt.text(0.3627,107,'b)')
    
    CS3 = axarr[0].contour(sig_plot,eps_plot,RMSrhoL_MBAR,[5,10],label='MBAR with single reference',colors='g')
#    CS4 = axarr[0].contour(sig_plot,eps_plot,RMSrhoL_PCFR,[5,10],label='PCFR with single reference',colors='c')
    CS1 = axarr[0].contour(sig_plot,eps_plot,RMSrhoL,contour_lines,label='Direct Simulation',colors='r')
    CS2 = axarr[0].contour(sig_plot,eps_plot,RMSrhoL_MBAR_9refs,contour_lines,label='MBAR with multiple references',colors='b')
    axarr[0].clabel(CS2, inline=1,fontsize=10,colors='w',fmt='%1.0f')
#    axarr[0].clabel(CS4, inline=1,fontsize=10,colors='c',fmt='%1.0f',manual=[(0,-10),(5,10)])
    axarr[0].clabel(CS3, inline=1,fontsize=10,colors='g',fmt='%1.0f')
    axarr[0].clabel(CS1, inline=1,fontsize=10,colors='k',fmt='%1.0f')
    axarr[0].set_xlabel(r'$\sigma$ (nm)')
    axarr[0].set_ylabel(r'$\epsilon$ (K)')
    axarr[0].set_title(r'RMS of $\rho_l$ (kg/m$^3$)')
    axarr[0].set_yticks([88,93,98,103,108])
    axarr[0].set_xticks([0.365,0.370,0.375,0.380,0.385])
    axarr[0].plot([],[],'r',label='Direct Simulation')
    axarr[0].plot([],[],'b',label='MBAR multiple references')
    axarr[0].plot([],[],'g',label='MBAR single reference')
#    axarr[0].plot([],[],'c',label='PCFR single reference')
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
    axarr[1].set_yticks([88,93,98,103,108])
    axarr[1].set_xticks([0.365,0.370,0.375,0.380,0.385])
    axarr[1].plot([],[],'r',label='Direct Simulation')
    axarr[1].plot([],[],'b',label='MBAR multiple references')
    axarr[1].plot([],[],'g',label='MBAR single reference')
    axarr[1].plot([],[],'c',label='PCFR single reference')
    axarr[1].plot(sig_PCFs,eps_PCFs,'mx',label='References')
    #axarr[1].legend()
        
    f.savefig('RMS_rhol_Psat_comparison.pdf')
    
    f, axarr = plt.subplots(nrows=2,ncols=1,figsize=(8,12))
    
    plt.tight_layout(pad=3,rect=[0,0,1,1])
    
    plt.text(0.3627,131,'a)') 
    plt.text(0.3627,107,'b)')
    
    RMSZ = np.loadtxt(fpathroot+'Direct_simulation_rr_RMS_Z_all').reshape(21,21)/20.
    RMSZ_MBAR = np.loadtxt(fpathroot+'MBAR_ref0rr_RMS_Z_all').reshape(21,21)/20.
    RMSZ_PCFR = np.loadtxt(fpathroot+'PCFR_ref0rr_RMS_Z_all').reshape(21,21)/20.
    RMSZ_constant = np.loadtxt(fpathroot+'Constant_rr_RMS_Z_all').reshape(21,21)/20.
    RMSZ_MBAR_9refs = np.loadtxt(fpathroot+'MBAR_ref8rr_RMS_Z_all').reshape(21,21)/20.
    RMSZ_zeroth = np.loadtxt(fpathroot+'Zeroth_ref0rr_RMS_Z_all').reshape(21,21)/20.
                          
    contour_lines = [0.01, 0.02, 0.03,0.04,0.05,0.06,0.07]
        
    CS3 = axarr[0].contour(sig_plot,eps_plot,RMSZ_MBAR,contour_lines[0:2],label='MBAR with single reference',colors='g')
    CS4 = axarr[0].contour(sig_plot,eps_plot,RMSZ_PCFR,contour_lines[0:2],label='PCFR with single reference',colors='c')
    CS1 = axarr[0].contour(sig_plot,eps_plot,RMSZ,contour_lines,label='Direct Simulation',colors='r')
    CS2 = axarr[0].contour(sig_plot,eps_plot,RMSZ_MBAR_9refs,contour_lines,label='MBAR with multiple references',colors='b')
    CS5 = axarr[0].contour(sig_plot,eps_plot,RMSZ_zeroth,contour_lines,colors='m')
    #axarr[0].clabel(CS2, inline=1,fontsize=10,colors='w',fmt='%1.1f')
    axarr[0].clabel(CS4, inline=1,fontsize=10,colors='c',fmt='%1.2f')
    axarr[0].clabel(CS3, inline=1,fontsize=10,colors='g',fmt='%1.2f')
    axarr[0].clabel(CS1, inline=0,fontsize=10,colors='k',fmt='%1.2f')
    axarr[0].set_xlabel(r'$\sigma$ (nm)')
    axarr[0].set_ylabel(r'$\epsilon$ (K)')
    axarr[0].set_title(r'RMS of $Z$')
    axarr[0].set_yticks([88,93,98,103,108])
    axarr[0].set_xticks([0.365,0.370,0.375,0.380,0.385])
    axarr[0].plot([],[],'r',label='Direct Simulation')
    axarr[0].plot([],[],'b',label='MBAR multiple references')
    axarr[0].plot([],[],'g',label='MBAR single reference')
    axarr[0].plot([],[],'c',label='PCFR single reference')
    axarr[0].plot(sig_PCFs,eps_PCFs,'mx',label='References')
        
    RMSU = np.loadtxt(fpathroot+'Direct_simulation_rr_RMS_U_all').reshape(21,21)/20.
    RMSU_MBAR = np.loadtxt(fpathroot+'MBAR_ref0rr_RMS_U_all').reshape(21,21)/20.
    RMSU_PCFR = np.loadtxt(fpathroot+'PCFR_ref0rr_RMS_U_all').reshape(21,21)/20.
    RMSU_constant = np.loadtxt(fpathroot+'Constant_rr_RMS_U_all').reshape(21,21)/20.
    RMSU_MBAR_9refs = np.loadtxt(fpathroot+'MBAR_ref8rr_RMS_U_all').reshape(21,21)/20.
    RMSU_zeroth = np.loadtxt(fpathroot+'Zeroth_ref0rr_RMS_U_all').reshape(21,21)/20.                            
                                    
    contour_lines = [8, 16, 24, 32, 40,48]
        
    CS3 = axarr[1].contour(sig_plot,eps_plot,RMSU_MBAR,contour_lines,label='MBAR with single reference',colors='g')
    CS4 = axarr[1].contour(sig_plot,eps_plot,RMSU_PCFR,contour_lines,label='PCFR with single reference',colors='c')
    CS1 = axarr[1].contour(sig_plot,eps_plot,RMSU,contour_lines,label='Direct Simulation',colors='r')
    CS2 = axarr[1].contour(sig_plot,eps_plot,RMSU_MBAR_9refs,contour_lines,label='MBAR with multiple references',colors='b')
    CS5 = axarr[1].contour(sig_plot,eps_plot,RMSU_zeroth,contour_lines,colors='m')
    #axarr[1].clabel(CS2, inline=1,fontsize=10,colors='w',fmt='%1.1f')
    axarr[1].clabel(CS4, inline=1,fontsize=10,colors='c',fmt='%1.0f')
    axarr[1].clabel(CS3, inline=1,fontsize=10,colors='g',fmt='%1.0f')
    axarr[1].clabel(CS1, inline=0,fontsize=10,colors='k',fmt='%1.0f')
    axarr[1].set_xlabel(r'$\sigma$ (nm)')
    axarr[1].set_ylabel(r'$\epsilon$ (K)')
    axarr[1].set_title(r'RMS of $U$ (kJ/mol)')
    axarr[1].set_yticks([88,93,98,103,108])
    axarr[1].set_xticks([0.365,0.370,0.375,0.380,0.385])
    axarr[1].plot([],[],'r',label='Direct Simulation')
    axarr[1].plot([],[],'b',label='MBAR multiple references')
    axarr[1].plot([],[],'g',label='MBAR single reference')
    axarr[1].plot([],[],'c',label='PCFR single reference')
    axarr[1].plot(sig_PCFs,eps_PCFs,'mx',label='References')
    
    f.savefig('RMS_U_Z_comparison.pdf')
    
    RMSZ = np.loadtxt('parameter_space_Mie16/Direct_simulation_rr_RMS_Z_all').reshape(21,21)/20.
    RMSZ_PCFR = np.loadtxt('parameter_space_Mie16/Zeroth_ref0rr_RMS_Z_all').reshape(21,21)/20.
#    RMSZ_zeroth = np.loadtxt('parameter_space_Mie16/Zeroth_ref0rr_RMS_Z_all').reshape(21,21)/20.
    RMSZ_MBAR = np.loadtxt('parameter_space_Mie16/MBAR_ref0rr_RMS_Z_all').reshape(21,21)/20.
                          
    RMSU = np.loadtxt('parameter_space_Mie16/Direct_simulation_rr_RMS_U_all').reshape(21,21)/20.
    RMSU_PCFR = np.loadtxt('parameter_space_Mie16/PCFR_ref0rr_RMS_U_all').reshape(21,21)/20.
#    RMSU_zeroth = np.loadtxt('parameter_space_Mie16/Zeroth_ref0rr_RMS_U_all').reshape(21,21)/20.
    RMSU_MBAR = np.loadtxt('parameter_space_Mie16/MBAR_ref0rr_RMS_U_all').reshape(21,21)/20.
    
    f, axarr = plt.subplots(nrows=2,ncols=1,figsize=(8,12))   
    
    plt.tight_layout(pad=3,rect=[0,0,1,1])
    
    plt.text(0.3627,151,'a)') 
    plt.text(0.3627,127,'b)')     
    
    contour_lines = [0.01, 0.02, 0.03,0.04,0.05,0.06,0.07]

    CS1 = axarr[0].contour(sig_plot,eps_plot+20.,RMSZ,contour_lines,colors='r')
#    CS2 = axarr[0].contour(sig_plot,eps_plot+20.,RMSZ_zeroth,contour_lines,colors='m')
    CS3 = axarr[0].contour(sig_plot,eps_plot+20.,RMSZ_PCFR,contour_lines,colors='c')
    CS4 = axarr[0].contour(sig_plot,eps_plot+20.,RMSZ_MBAR,contour_lines,colors='g')
    axarr[0].clabel(CS1, inline=1,fontsize=10,colors='r',fmt='%1.2f')
#    axarr[0].clabel(CS2, inline=1,fontsize=10,colors='m',fmt='%1.2f')
    axarr[0].clabel(CS3, inline=1,fontsize=10,colors='c',fmt='%1.2f')
    axarr[0].clabel(CS4, inline=1,fontsize=10,colors='g',fmt='%1.2f')
    axarr[0].plot([],[],'r',label='Direct Simulation')
    axarr[0].plot([],[],'c',label=r'PCFR,$\theta_{ref} =$ TraPPE-UA')
#    axarr[0].plot([],[],'c',label='PCFR, PMF')
#    axarr[0].plot([],[],'m',label='PCFR, zeroth')
    axarr[0].plot([],[],'g',label=r'MBAR,$\theta_{ref} =$ TraPPE-UA')
    axarr[0].plot(0.3783,121.25,'kx',markersize=10,label=r'Potoff, $\lambda = 16$')
    axarr[0].set_xlabel(r'$\sigma$ (nm)')
    axarr[0].set_ylabel(r'$\epsilon$ (K)')
    axarr[0].set_title(r'RMS of $Z$')
    axarr[0].set_yticks([108,113,118,123,128])
    axarr[0].set_xticks([0.365,0.370,0.375,0.380,0.385])
    axarr[0].legend(loc=2)         
    
    contour_lines = [150/20.,300/20.,450/20.,600/20.,750/20.,900/20.]
    contour_lines = [150/20.,300/20.,450/20.]
                     
    CS1 = axarr[1].contour(sig_plot,eps_plot+20.,RMSU,contour_lines,colors='r')
#    CS2 = axarr[1].contour(sig_plot,eps_plot+20.,RMSU_zeroth,contour_lines,colors='m')
    CS3 = axarr[1].contour(sig_plot,eps_plot+20.,RMSU_PCFR,contour_lines,colors='c')
    CS4 = axarr[1].contour(sig_plot,eps_plot+20.,RMSU_MBAR,contour_lines,colors='g')
    axarr[1].clabel(CS1, inline=1,fontsize=10,colors='r',fmt='%1.1f')
#    axarr[1].clabel(CS2, inline=1,fontsize=10,colors='m',fmt='%1.1f')
    axarr[1].clabel(CS3, inline=1,fontsize=10,colors='c',fmt='%1.1f')
    axarr[1].clabel(CS4, inline=1,fontsize=10,colors='g',fmt='%1.1f')
    axarr[1].plot([],[],'r',label='Direct Simulation')
    axarr[1].plot([],[],'c',label=r'PCFR,$\theta_{ref} =$ TraPPE-UA')
#    axarr[1].plot([],[],'c',label='PCFR, PMF')
#    axarr[1].plot([],[],'m',label='PCFR, zeroth')
    axarr[1].plot([],[],'g',label=r'MBAR,$\theta_{ref} =$ TraPPE-UA')
    axarr[1].plot(0.3783,121.25,'kx',markersize=10,label=r'Potoff, $\lambda = 16$')
    axarr[1].set_xlabel(r'$\sigma$ (nm)')
    axarr[1].set_ylabel(r'$\epsilon$ (K)')
    axarr[1].set_title(r'RMS of $U_{dep}$ (kJ/mol)')
    axarr[1].set_yticks([108,113,118,123,128])
    axarr[1].set_xticks([0.365,0.370,0.375,0.380,0.385])
    axarr[1].legend(loc=3)
    plt.show()
    
    RMSrhoL = np.loadtxt('parameter_space_Mie16/Direct_simulation_rr_RMS_rhoL_all').reshape(21,21)
    RMSrhoL_MBAR = np.loadtxt('parameter_space_Mie16/MBAR_ref0rr_RMS_rhoL_all').reshape(21,21)
    RMSrhoL_PCFR = np.loadtxt('parameter_space_Mie16/Zeroth_ref0rr_RMS_rhoL_all').reshape(21,21)
    
    f, axarr = plt.subplots(nrows=2,ncols=1,figsize=(8,12))   
    
    plt.tight_layout(pad=3,rect=[0,0,1,1])
    
#    plt.text(0.3627,151,'a)') 
#    plt.text(0.3627,127,'b)')     
    
    contour_lines = [5,10,20,30]

    CS1 = axarr[0].contour(sig_plot,eps_plot+20.,RMSrhoL,contour_lines,colors='r')
#    CS2 = axarr[0].contour(sig_plot,eps_plot+20.,RMSrhoL_zeroth,contour_lines,colors='m')
    CS3 = axarr[0].contour(sig_plot,eps_plot+20.,RMSrhoL_PCFR,contour_lines[0:3],colors='c')
    CS4 = axarr[0].contour(sig_plot,eps_plot+20.,RMSrhoL_MBAR,contour_lines[0:3],colors='g')
    axarr[0].clabel(CS1, inline=1,fontsize=10,colors='r',fmt='%1.2f')
#    axarr[0].clabel(CS2, inline=1,fontsize=10,colors='m',fmt='%1.2f')
    axarr[0].clabel(CS3, inline=1,fontsize=10,colors='c',fmt='%1.2f')
    axarr[0].clabel(CS4, inline=1,fontsize=10,colors='g',fmt='%1.2f')
    axarr[0].plot([],[],'r',label='Direct Simulation')
    #    axarr[0].plot([],[],'c',label='PCFR, PMF')
#    axarr[0].plot([],[],'m',label='PCFR, zeroth')
    axarr[0].plot([],[],'g',label=r'MBAR,$\theta_{ref} =$ TraPPE-UA')
    axarr[0].plot([],[],'c',label=r'PCFR,$\theta_{ref} =$ TraPPE-UA')
    axarr[0].plot(0.3783,121.25,'kx',markersize=10,label=r'Potoff, $\lambda = 16$')
    axarr[0].set_xlabel(r'$\sigma$ (nm)')
    axarr[0].set_ylabel(r'$\epsilon$ (K)')
    axarr[0].set_title(r'RMS of $\rho_l$ (kg/m$^3$)')
    axarr[0].set_yticks([108,113,118,123,128])
    axarr[0].set_xticks([0.365,0.370,0.375,0.380,0.385])
    axarr[0].legend(loc=2)
    
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
        xmin = -7000/400.
        xmax = -500/400.
        ymin = -7000/400.
        ymax = -500/400.
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
    
    plt.tight_layout(pad=2,rect=[0.02,0.01,0.965,0.99])
    
    normalize = mpl.Normalize(vmin=0, vmax=3)
    
    font = {'size' : '14'}
    plt.rc('font',**font)
    
    if fpathroot == 'parameter_space_LJ/':

        plt.text(-19.8,34,r'a)') 
        plt.text(-19.8,11,r'b)')
        plt.text(-3.8,34,r'c)')
        plt.text(-3.8,11,r'd)')
        plt.text(-13,35.5,r'Constant Model: LJ 12-6, $88\leq\epsilon/K\leq108, 0.365\leq\sigma/nm\leq0.385$')
        plt.text(-19.8,35.5,r'$\theta_{ref} = \theta_{TraPPE-UA}$')

    elif fpathroot == 'parameter_space_Mie16/':

        plt.text(-22,59.5,r'a)') 
        plt.text(-22,20,r'b)')
        plt.text(-3.8,59.5,r'c)')
        plt.text(-3.8,20,r'd)') 
        plt.text(-13,62,r'Perturbed Model: Mie 16-6, $108\leq\epsilon/K\leq128, 0.365\leq\sigma/nm\leq0.385$')  
        plt.text(-22,62,r'$\theta_{ref} = \theta_{TraPPE-UA}$')                              
                                             
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

#        parity = np.array([np.min(np.array([np.min(prop_direct),np.min(prop_hat1),np.min(prop_hat2)])),np.max(np.array([np.max(prop_direct),np.max(prop_hat1),np.max(prop_hat2)]))])
        
        parity = np.array([np.min(prop_direct),np.max(prop_direct)])
                                 
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
            xlabel = r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Direct Simulation'
            ylabel = r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Predicted'
            dev1 = rel_dev1
            dev2 = rel_dev2
            dev3 = rel_dev3
            dev4 = rel_dev4
            dev_type = 'Percent'
            xmin = -7000/400.
            xmax = -500/400.
            ymin = -7000/400.
            ymax = -500/400.
            embed_comparison = [0.58,0.16,0.31,0.31]
            embed_MBAR = [0.45,0.13,0.28,0.28]
        elif prop == 'Z':
            units = ''
            title = 'Compressibility Factor'
            xlabel = r'$Z$, Direct Simulation'
            ylabel = r'$Z$, Predicted'
            dev1 = abs_dev1
            dev2 = abs_dev2
            dev3 = abs_dev3
            dev4 = abs_dev4
            dev_type = 'Absolute'
            if fpathroot == 'parameter_space_LJ/':
                
                xmin = -4 
                xmax = 10
                ymin = -8
                ymax = 12
                
            elif fpathroot == 'parameter_space_Mie16/':
                
                xmin = -4 
                xmax = 12
                ymin = -12
                ymax = 22
                
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
        axarr[ifig,jfig].set_xlabel(xlabel,fontdict=font)
        axarr[ifig,jfig].set_ylabel(ylabel,fontdict=font)
        axarr[ifig,jfig].set_xlim([xmin,xmax])
        axarr[ifig,jfig].set_ylim([ymin,ymax])
        
        if jfig == 1 and fpathroot == 'parameter_space_LJ/':
            axarr[ifig,jfig].set_yticks([-8,-4,0,4,8,12])
        #axarr[ifig,jfig].set_title(title)
        #axarr[ifig,jfig].text(2,0.65,panels[ifig,jfig])
        if ifig == 0 and jfig == 0:
            axarr[ifig,jfig].legend(['Constant PCF','MBAR','PCFR','Recommended','Parity'],loc='upper center',frameon=False)
            a = inset_axes(axarr[ifig,jfig],width=2.5,height=2.5,loc=4)
        elif ifig == 0 and jfig == 1:
            a = inset_axes(axarr[ifig,jfig],width=2.3,height=2.3,loc=4)
        a.plot(prop_direct,dev3,'b.',label='Constant PCF',alpha=0.2)   
        a.plot(prop_direct,dev1,'r.',label='MBAR',alpha=0.2)
        a.plot(prop_direct,dev2,'g.',label='PCFR',alpha=0.2)
        a.plot(prop_direct,dev4,'c.',label='Recommended',alpha=0.2)
        a.plot(parity,[0,0],'k--')
        a.xaxis.tick_top()
        #a.set_xticks([])
        #plt.xlabel('Direct Simulation '+units)
        if dev_type == 'Percent':
            a.set_ylabel(dev_type+' Deviation',labelpad=-5)
        else:
            a.set_ylabel('Deviation '+units,labelpad=-5)
#            a.set_xlabel(xlabel,labelpad=-5)
#            a.xaxis.set_label_position('top')
            
        ifig += 1
            
        p = axarr[ifig,jfig].scatter(prop_direct[Neff.argsort()],prop_hat1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR',norm=normalize)    
        axarr[ifig,jfig].plot(parity,parity,'k',label='Parity')
        axarr[ifig,jfig].set_xlabel(xlabel,fontdict=font)
        axarr[ifig,jfig].set_ylabel(ylabel+' with MBAR',fontdict=font)
        axarr[ifig,jfig].set_xlim([xmin,xmax])
        axarr[ifig,jfig].set_ylim([ymin,ymax])
        #axarr[1].set_title(title)
        
        if jfig == 1 and fpathroot == 'parameter_space_LJ/':
            axarr[ifig,jfig].set_yticks([-8,-4,0,4,8,12])
        
        if jfig == 1:
            cbaxes = f.add_axes([0.94,0.065,0.01,0.412])
            cb = plt.colorbar(p,ax=axarr[ifig,jfig],cax=cbaxes,format='%.1f')
#            cb = plt.colorbar(p,ax=axarr[ifig,jfig],pad=0.02)
            cb.set_label('log$_{10}(N_{eff})$')
        #cax = cb.ax
        #cax.set_position([0.5,0.5,0.5,0.5])
        a = inset_axes(axarr[ifig,jfig],width=2.5,height=2.5,loc=4)
        a.scatter(prop_direct[Neff.argsort()],dev1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR',norm=normalize)   
        a.xaxis.tick_top()
#        a.set_xticks([])
        a.plot(parity,[0,0],'k--')
        #plt.xlabel('Direct Simulation '+units)
        #a.set_ylim([np.floor(np.min(dev1)),np.ceil(np.max(dev1))])
        
        if jfig == 1 and fpathroot == 'parameter_space_Mie16/':
            
            a.set_ylim([-3.5,16.5])
            a.set_yticks([-3,0,3,6,9,12,15])
        
        if dev_type == 'Percent':
            a.set_ylabel(dev_type+' Deviation',labelpad=-5)
        else:
            a.set_ylabel('Deviation '+units,labelpad=-5)
            
        jfig += 1
        
    f.savefig('Multi_prop_combined.pdf')
    
def multi_prop_multi_ref_plot(U_direct,U_MBAR,Z_direct,Z_MBAR,Neff,fpathroot):
    
    abs_dev = lambda hat, direct: (hat - direct)
    rel_dev = lambda hat, direct: abs_dev(hat,direct)/direct * 100.
                                         
    ifig = 0                                     
                                                                
    f, axarr = plt.subplots(nrows=1,ncols=2,figsize=(16,6)) 
    
    normalize = mpl.Normalize(vmin=0, vmax=3)

    if fpathroot == 'parameter_space_LJ/':

        plt.text(-23.5,7.5,'a)') 
        plt.text(-5.5,7.5,'b)')

    elif fpathroot == 'parameter_space_Mie16/':

        plt.text(-29,7.5,'a)') 
        plt.text(-5.5,7.5,'b)')                            
                                             
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
            xmin = -7000/400.
            xmax = -500/400.
            ymin = -7000/400.
            ymax = -500/400.
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
                            
        p = axarr[ifig].scatter(prop_direct[Neff.argsort()],prop_hat1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR',norm=normalize)    
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
        a.scatter(prop_direct[Neff.argsort()],dev1[Neff.argsort()],c=np.log10(Neff[Neff.argsort()]),cmap='rainbow',label='MBAR',norm=normalize)   
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
    
    normalize = mpl.Normalize(vmin=0, vmax=np.log10(np.max(Neff_LJ)))
    normalize = mpl.Normalize(vmin=0, vmax=3)
    
    my_figure = plt.figure(figsize=(16,12))
    subplot_1 = my_figure.add_subplot(2,2,1)
    plt.tight_layout(pad=4,rect=[0.02,0.01,0.965,0.99])
    
#    font = {'size' : '14'}
#    plt.rc('font',**font)
    
    subplot_1.scatter(U_direct_LJ[Neff_LJ.argsort()],U_MBAR_LJ[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')
    subplot_1.plot(parity_1,parity_1,'k',label='Parity')
    subplot_1.set_xlabel(r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Direct Simulation')
    subplot_1.set_ylabel(r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Predicted with MBAR')
    subplot_1.set_xlim([-7000/400.,-500/400.])
    subplot_1.set_ylim([-7000/400.,-500/400.])
    
    a = inset_axes(subplot_1,width=2.6,height=2.6,loc=4)          
    a.scatter(U_direct_LJ[Neff_LJ.argsort()],dev1[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')   
    a.plot(parity_1,[0,0],'k--')
    a.set_xticks([])
    a.set_yticks([-1.0,-0.5,0.0,0.5,1.0])
    a.set_ylabel('Percent Deviation')
    
    subplot_2 = my_figure.add_subplot(2,2,2)
    p = subplot_2.scatter(Z_direct_LJ[Neff_LJ.argsort()],Z_MBAR_LJ[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')
    subplot_2.plot(parity_2,parity_2,'k',label='Parity')
    subplot_2.set_xlabel('$Z$, Direct Simulation')
    subplot_2.set_ylabel('$Z$, Predicted with MBAR')
    subplot_2.set_xlim([-4,8])
    subplot_2.set_ylim([-4,8])
    
    a = inset_axes(subplot_2,width=2.5,height=2.5,loc=4)          
    a.scatter(Z_direct_LJ[Neff_LJ.argsort()],dev2[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')   
    a.plot(parity_2,[0,0],'k--')
    a.set_yticks([-0.2,-0.1,0,0.1,0.2])
    a.set_xticks([])
    a.set_ylabel('Absolute Deviation')
    
#    cb = plt.colorbar(p,ax=subplot_2,pad=0.02)
#    cb.set_label('log$_{10}(N_{eff})$')
    
    subplot_3 = my_figure.add_subplot(2,2,3)
    subplot_3.scatter(U_direct_Mie[Neff_Mie.argsort()],U_MBAR_Mie[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')
    subplot_3.plot(parity_3,parity_3,'k',label='Parity')
    subplot_3.set_xlabel(r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Direct Simulation')
    subplot_3.set_ylabel(r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Predicted with MBAR')
    subplot_3.set_xlim([-7000/400.,-500/400.])
    subplot_3.set_ylim([-7000/400.,-500/400.])
    
    a = inset_axes(subplot_3,width=2.65,height=2.65,loc=4)          
    a.scatter(U_direct_Mie[Neff_Mie.argsort()],dev3[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')   
    a.plot(parity_3,[0,0],'k--')
    a.set_xticks([])
    a.set_yticks([-16,-12,-8,-4,-2,0,2])
    a.set_ylabel('Percent Deviation')
    
    subplot_4 = my_figure.add_subplot(2,2,4)
    p = subplot_4.scatter(Z_direct_Mie[Neff_Mie.argsort()],Z_MBAR_Mie[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')
    subplot_4.plot(parity_4,parity_4,'k',label='Parity')
    subplot_4.set_xlabel('$Z$, Direct Simulation')
    subplot_4.set_ylabel('$Z$, Predicted with MBAR')
    subplot_4.set_xlim([-4,10])
    subplot_4.set_ylim([-10,14])
    
    a = inset_axes(subplot_4,width=2.5,height=2.5,loc=4)          
    a.scatter(Z_direct_Mie[Neff_Mie.argsort()],dev4[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')   
    a.plot(parity_4,[0,0],'k--')
    a.set_xticks([])
    a.set_ylabel('Absolute Deviation')
    
#    cb = plt.colorbar(p,ax=subplot_4,pad=0.02)
#    cb.set_label('log$_{10}(N_{eff})$')
    
    subplot_1.text(-6800/400., -800/400., r'a) Multiple $\theta_{ref}$: $\epsilon = 98$ K,  $0.365\leq\sigma/nm\leq0.385$.'+'\n'+r'Constant Model: LJ 12-6,'+'\n'+r'$88\leq\epsilon/K\leq108, 0.365\leq\sigma/nm\leq0.385$')
    subplot_2.text(-3.8, 7.5, r'c) Multiple $\theta_{ref}$: $\epsilon = 98$ K,  $0.365\leq\sigma/nm\leq0.385$.'+'\n'+r'Constant Model: LJ 12-6,'+'\n'+r'$88\leq\epsilon/K\leq108, 0.365\leq\sigma/nm\leq0.385$')
    subplot_3.text(-6800/400., -800/400., r'b) Multiple $\theta_{ref}$: $\epsilon = 98$ K,  $0.365\leq\sigma/nm\leq0.385$.'+'\n'+r'Perturbed Model: Mie 16-6,'+'\n'+r'$108\leq\epsilon/K\leq128, 0.365\leq\sigma/nm\leq0.385$')
    subplot_4.text(-3.8, 13, r'd) Multiple $\theta_{ref}$: $\epsilon = 98$ K,  $0.365\leq\sigma/nm\leq0.385$.'+'\n'+r'Perturbed Model: Mie 16-6,'+'\n'+r'$108\leq\epsilon/K\leq128, 0.365\leq\sigma/nm\leq0.385$')
            
    cbaxes = my_figure.add_axes([0.94,0.055,0.02,0.885])
    cb = plt.colorbar(p,ax=subplot_4,cax=cbaxes,format='%.1f')
    cb.set_label('log$_{10}(N_{eff})$')
        
    my_figure.savefig('MBAR_multi_ref_LJ_Mie.pdf')
    
def multi_ref_LJ_Mie_LJhighEps_plot(U_direct_LJ,U_MBAR_LJ,U_direct_Mie,U_MBAR_Mie,U_MBAR_LJhighEps,Z_direct_LJ,Z_MBAR_LJ,Z_direct_Mie,Z_MBAR_Mie,Z_MBAR_LJhighEps,Neff_LJ,Neff_Mie,Neff_LJhighEps):

    abs_dev = lambda hat, direct: (hat - direct)
    rel_dev = lambda hat, direct: abs_dev(hat,direct)/direct * 100.
    
    dev1 = rel_dev(U_MBAR_LJ,U_direct_LJ)
    dev2 = abs_dev(Z_MBAR_LJ,Z_direct_LJ)
    dev3 = rel_dev(U_MBAR_Mie,U_direct_Mie)
    dev4 = abs_dev(Z_MBAR_Mie,Z_direct_Mie)
    dev5 = rel_dev(U_MBAR_LJhighEps,U_direct_Mie)  
    dev6 = abs_dev(Z_MBAR_LJhighEps,Z_direct_Mie)                                  
                                         
    parity_1 = np.array([np.min(np.array([np.min(U_direct_LJ),np.min(U_MBAR_LJ)])),np.max(np.array([np.max(U_direct_LJ),np.max(U_MBAR_LJ)]))])
    parity_2 = np.array([np.min(np.array([np.min(Z_direct_LJ),np.min(Z_MBAR_LJ)])),np.max(np.array([np.max(Z_direct_LJ),np.max(Z_MBAR_LJ)]))])
    parity_3 = np.array([np.min(np.array([np.min(U_direct_Mie),np.min(U_MBAR_Mie)])),np.max(np.array([np.max(U_direct_Mie),np.max(U_MBAR_Mie)]))])
    parity_4 = np.array([np.min(np.array([np.min(Z_direct_Mie),np.min(Z_MBAR_Mie)])),np.max(np.array([np.max(Z_direct_Mie),np.max(Z_MBAR_Mie)]))])      
    parity_5 = np.array([np.min(np.array([np.min(U_direct_Mie),np.min(U_MBAR_LJhighEps)])),np.max(np.array([np.max(U_direct_Mie),np.max(U_MBAR_LJhighEps)]))])
    parity_6 = np.array([np.min(np.array([np.min(Z_direct_Mie),np.min(Z_MBAR_LJhighEps)])),np.max(np.array([np.max(Z_direct_Mie),np.max(Z_MBAR_LJhighEps)]))])      
    
    parity_4 = np.array([np.min(Z_direct_Mie),np.max(Z_direct_Mie)]) #This is really what I want in my plots
    parity_6 = np.array([np.min(Z_direct_Mie),np.max(Z_direct_Mie)]) #This is really what I want in my plots
    
    normalize = mpl.Normalize(vmin=0, vmax=np.log10(np.max(Neff_LJ)))
    normalize = mpl.Normalize(vmin=0, vmax=3)
    
    my_figure = plt.figure(figsize=(16,18))
    subplot_1 = my_figure.add_subplot(3,2,1)
    plt.tight_layout(pad=2.5,rect=[0.02,0.01,0.965,0.99])
    
#    font = {'size' : '14'}
#    plt.rc('font',**font)
    
    subplot_1.scatter(U_direct_LJ[Neff_LJ.argsort()],U_MBAR_LJ[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')
    subplot_1.plot(parity_1,parity_1,'k',label='Parity')
    subplot_1.set_xlabel(r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Direct Simulation',labelpad=-2)
    subplot_1.set_ylabel(r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Predicted with MBAR',labelpad=-4)
    subplot_1.set_xlim([-7000/400.,-500/400.])
    subplot_1.set_ylim([-7000/400.,-500/400.])
    
    a = inset_axes(subplot_1,width=2.6,height=2.6,loc=4)          
    a.scatter(U_direct_LJ[Neff_LJ.argsort()],dev1[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')   
    a.plot(parity_1,[0,0],'k--')
    a.xaxis.tick_top()
#    a.set_xticks([])
    a.set_yticks([-1,0,1])
    a.set_ylabel('Percent Deviation',labelpad=-4)
    
    subplot_2 = my_figure.add_subplot(3,2,2)
    p = subplot_2.scatter(Z_direct_LJ[Neff_LJ.argsort()],Z_MBAR_LJ[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')
    subplot_2.plot(parity_2,parity_2,'k',label='Parity')
    subplot_2.set_xlabel('$Z$, Direct Simulation',labelpad=-2)
    subplot_2.set_ylabel('$Z$, Predicted with MBAR',labelpad=-7)
    subplot_2.set_xlim([-4,8])
    subplot_2.set_ylim([-4,8])
    
    a = inset_axes(subplot_2,width=2.5,height=2.5,loc=4)          
    a.scatter(Z_direct_LJ[Neff_LJ.argsort()],dev2[Neff_LJ.argsort()],c=np.log10(Neff_LJ[Neff_LJ.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')   
    a.plot(parity_2,[0,0],'k--')
    a.set_yticks([-0.2,-0.1,0,0.1,0.2])
    a.xaxis.tick_top()
#    a.set_xticks([])
    a.set_ylabel('Deviation',labelpad=-4)
    
#    cb = plt.colorbar(p,ax=subplot_2,pad=0.02)
#    cb.set_label('log$_{10}(N_{eff})$')
    
    subplot_3 = my_figure.add_subplot(3,2,3)
    subplot_3.scatter(U_direct_Mie[Neff_Mie.argsort()],U_MBAR_Mie[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')
    subplot_3.plot(parity_3,parity_3,'k',label='Parity')
    subplot_3.set_xlabel(r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Direct Simulation',labelpad=-2)
    subplot_3.set_ylabel(r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Predicted with MBAR',labelpad=-4)
    subplot_3.set_xlim([-7000/400.,-500/400.])
    subplot_3.set_ylim([-7000/400.,-500/400.])
    
    a = inset_axes(subplot_3,width=2.65,height=2.65,loc=4)          
    a.scatter(U_direct_Mie[Neff_Mie.argsort()],dev3[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')   
    a.plot(parity_3,[0,0],'k--')
    a.xaxis.tick_top()
#    a.set_xticks([])
    a.set_yticks([-16,-12,-8,-4,-2,0,2])
    a.set_ylabel('Percent Deviation',labelpad=-4)
    
    subplot_4 = my_figure.add_subplot(3,2,4)
    p = subplot_4.scatter(Z_direct_Mie[Neff_Mie.argsort()],Z_MBAR_Mie[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')
    subplot_4.plot(parity_4,parity_4,'k',label='Parity')
    subplot_4.set_xlabel('$Z$, Direct Simulation',labelpad=-2)
    subplot_4.set_ylabel('$Z$, Predicted with MBAR',labelpad=-15)
    subplot_4.set_xlim([-4,10])
    subplot_4.set_ylim([-10,14])
    
    a = inset_axes(subplot_4,width=2.5,height=2.5,loc=4)          
    a.scatter(Z_direct_Mie[Neff_Mie.argsort()],dev4[Neff_Mie.argsort()],c=np.log10(Neff_Mie[Neff_Mie.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')   
    a.plot(parity_4,[0,0],'k--')
    a.xaxis.tick_top()
#    a.set_xticks([])
    a.set_ylabel('Deviation',labelpad=0)
    
#    cb = plt.colorbar(p,ax=subplot_4,pad=0.02)
#    cb.set_label('log$_{10}(N_{eff})$')

    subplot_5 = my_figure.add_subplot(3,2,5)
    subplot_5.scatter(U_direct_Mie[Neff_LJhighEps.argsort()],U_MBAR_LJhighEps[Neff_LJhighEps.argsort()],c=np.log10(Neff_LJhighEps[Neff_LJhighEps.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')
    subplot_5.plot(parity_5,parity_5,'k',label='Parity')
    subplot_5.set_xlabel(r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Direct Simulation',labelpad=-2)
    subplot_5.set_ylabel(r'$U_{dep} \left(\frac{kJ}{mol}\right)$, Predicted with MBAR',labelpad=-4)
    subplot_5.set_xlim([-7000/400.,-500/400.])
    subplot_5.set_ylim([-7000/400.,-500/400.])
    
    a = inset_axes(subplot_5,width=2.65,height=2.65,loc=4)          
    a.scatter(U_direct_Mie[Neff_LJhighEps.argsort()],dev5[Neff_LJhighEps.argsort()],c=np.log10(Neff_LJhighEps[Neff_LJhighEps.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')   
    a.plot(parity_5,[0,0],'k--')
    a.xaxis.tick_top()
#    a.set_xticks([])
    a.set_yticks([-2,0,2])
    a.set_ylabel('Percent Deviation',labelpad=-4)
    
    subplot_6 = my_figure.add_subplot(3,2,6)
    p = subplot_6.scatter(Z_direct_Mie[Neff_LJhighEps.argsort()],Z_MBAR_LJhighEps[Neff_LJhighEps.argsort()],c=np.log10(Neff_LJhighEps[Neff_LJhighEps.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')
    subplot_6.plot(parity_6,parity_6,'k',label='Parity')
    subplot_6.set_xlabel('$Z$, Direct Simulation',labelpad=-2)
    subplot_6.set_ylabel('$Z$, Predicted with MBAR',labelpad=-5)
    subplot_6.set_xlim([-4,10])
    subplot_6.set_ylim([-6,10])
    
    a = inset_axes(subplot_6,width=2.5,height=2.5,loc=4)          
    a.scatter(Z_direct_Mie[Neff_LJhighEps.argsort()],dev6[Neff_LJhighEps.argsort()],c=np.log10(Neff_LJhighEps[Neff_LJhighEps.argsort()]),cmap='rainbow',norm=normalize,label='MBAR')   
    a.plot(parity_6,[0,0],'k--')
    a.xaxis.tick_top()
#    a.set_xticks([])
    a.set_ylabel('Deviation',labelpad=-2)
    
    subplot_1.text(-17., -4., r'a) Multiple $\theta_{ref}$: $\epsilon = 98$ K,  $0.365\leq\sigma/nm\leq0.385$'+'\n'+r'Constant Model: LJ 12-6,'+'\n'+r'$88\leq\epsilon/K\leq108, 0.365\leq\sigma/nm\leq0.385$')
    subplot_2.text(-3.8, 6, r'd) Multiple $\theta_{ref}$: $\epsilon = 98$ K,  $0.365\leq\sigma/nm\leq0.385$'+'\n'+r'Constant Model: LJ 12-6,'+'\n'+r'$88\leq\epsilon/K\leq108, 0.365\leq\sigma/nm\leq0.385$')
    subplot_3.text(-17., -4., r'b) Multiple $\theta_{ref}$: $\epsilon = 98$ K,  $0.365\leq\sigma/nm\leq0.385$'+'\n'+r'Perturbed Model: Mie 16-6,'+'\n'+r'$108\leq\epsilon/K\leq128, 0.365\leq\sigma/nm\leq0.385$')
    subplot_4.text(-3.8, 10, r'e) Multiple $\theta_{ref}$: $\epsilon = 98$ K,  $0.365\leq\sigma/nm\leq0.385$'+'\n'+r'Perturbed Model: Mie 16-6,'+'\n'+r'$108\leq\epsilon/K\leq128, 0.365\leq\sigma/nm\leq0.385$')
    subplot_5.text(-17., -4., r'c) Multiple $\theta_{ref}$: $\epsilon = 118$ K,  $0.365\leq\sigma/nm\leq0.393$'+'\n'+r'Perturbed Model: Mie 16-6,'+'\n'+r'$108\leq\epsilon/K\leq128, 0.365\leq\sigma/nm\leq0.385$')
    subplot_6.text(-3.8, 7.3, r'f) Multiple $\theta_{ref}$: $\epsilon = 118$ K,  $0.365\leq\sigma/nm\leq0.393$'+'\n'+r'Perturbed Model: Mie 16-6,'+'\n'+r'$108\leq\epsilon/K\leq128, 0.365\leq\sigma/nm\leq0.385$')
        
    cbaxes = my_figure.add_axes([0.945,0.035,0.01,0.925])
    cb = plt.colorbar(p,ax=subplot_6,cax=cbaxes,format='%.1f')
    cb.set_label('log$_{10}(N_{eff})$')
        
    my_figure.savefig('MBAR_multi_ref_LJ_Mie_LJhighEps.pdf')
    
def lambda_comparison(eps_all,sig_all,fpathroot):
    
    for model_type in ['Direct_simulation','Lam12/MBAR_ref0','Lam13/MBAR_ref0','Lam14/MBAR_ref0','Lam15/MBAR_ref0','Lam17/MBAR_ref0','Lam18/MBAR_ref0']:
        if model_type == 'Direct_simulation':
            U_direct, dU_direct, P_direct, dP_direct, Z_direct, dZ_direct, Neff_direct = compile_data(model_type,fpathroot)
        elif model_type == 'Lam12/MBAR_ref0':
            U_12, dU_12, P_12, dP_12, Z_12, dZ_12, Neff_12 = compile_data(model_type,fpathroot)
        elif model_type == 'Lam13/MBAR_ref0':
            U_13, dU_13, P_13, dP_13, Z_13, dZ_13, Neff_13 = compile_data(model_type,fpathroot)
        elif model_type == 'Lam14/MBAR_ref0':
            U_14, dU_14, P_14, dP_14, Z_14, dZ_14, Neff_14 = compile_data(model_type,fpathroot)
        elif model_type == 'Lam15/MBAR_ref0':
            U_15, dU_15, P_15, dP_15, Z_15, dZ_15, Neff_15 = compile_data(model_type,fpathroot)
        elif model_type == 'Lam17/MBAR_ref0':
            U_17, dU_17, P_17, dP_17, Z_17, dZ_17, Neff_17 = compile_data(model_type,fpathroot)
        elif model_type == 'Lam18/MBAR_ref0':
            U_18, dU_18, P_18, dP_18, Z_18, dZ_18, Neff_18 = compile_data(model_type,fpathroot)
    
    SSE_U12 = np.sum((U_direct-U_12)**2,axis=0).reshape(21,21)
    SSE_U13 = np.sum((U_direct-U_13)**2,axis=0).reshape(21,21)        
    SSE_U14 = np.sum((U_direct-U_14)**2,axis=0).reshape(21,21)
    SSE_U15 = np.sum((U_direct-U_15)**2,axis=0).reshape(21,21)
    SSE_U17 = np.sum((U_direct-U_17)**2,axis=0).reshape(21,21)
    SSE_U18 = np.sum((U_direct-U_18)**2,axis=0).reshape(21,21)
    
    RMS_U12 = np.sqrt(SSE_U12/len(U_direct))
    RMS_U13 = np.sqrt(SSE_U13/len(U_direct))
    RMS_U14 = np.sqrt(SSE_U14/len(U_direct))
    RMS_U15 = np.sqrt(SSE_U15/len(U_direct))
    RMS_U17 = np.sqrt(SSE_U17/len(U_direct))
    RMS_U18 = np.sqrt(SSE_U18/len(U_direct))
    
    SSE_Z12 = np.sum((Z_direct-Z_12)**2,axis=0).reshape(21,21)
    SSE_Z13 = np.sum((Z_direct-Z_13)**2,axis=0).reshape(21,21)
    SSE_Z14 = np.sum((Z_direct-Z_14)**2,axis=0).reshape(21,21)
    SSE_Z15 = np.sum((Z_direct-Z_15)**2,axis=0).reshape(21,21)
    SSE_Z17 = np.sum((Z_direct-Z_17)**2,axis=0).reshape(21,21)
    SSE_Z18 = np.sum((Z_direct-Z_18)**2,axis=0).reshape(21,21)
    
    RMS_Z12 = np.sqrt(SSE_Z12/len(Z_direct))
    RMS_Z13 = np.sqrt(SSE_Z13/len(Z_direct))
    RMS_Z14 = np.sqrt(SSE_Z14/len(Z_direct))
    RMS_Z15 = np.sqrt(SSE_Z15/len(Z_direct))
    RMS_Z17 = np.sqrt(SSE_Z17/len(Z_direct))
    RMS_Z18 = np.sqrt(SSE_Z18/len(Z_direct))
    
    eps_plot = np.unique(eps_all)
    sig_plot = np.unique(sig_all)
                    
    my_figure = plt.figure(figsize=(8,12))
    subplot_1 = my_figure.add_subplot(2,1,1)
    
    contour_lines = [0.15,0.30]
    contour_lines = [0.15]
    fmt_prop = '%1.2f'
    
    CS12 = subplot_1.contour(sig_plot,eps_plot,RMS_U12,contour_lines,colors='y')
    CS13 = subplot_1.contour(sig_plot,eps_plot,RMS_U13,contour_lines,colors='m')
    CS14 = subplot_1.contour(sig_plot,eps_plot,RMS_U14,contour_lines,colors='r')
    CS15 = subplot_1.contour(sig_plot,eps_plot,RMS_U15,contour_lines,colors='g')
    CS16 = subplot_1.contour(sig_plot,eps_plot,RMS_U17,contour_lines,colors='b')
    CS17 = subplot_1.contour(sig_plot,eps_plot,RMS_U18,contour_lines,colors='c')
    plt.clabel(CS12, inline=1,fontsize=10,colors='y',fmt=fmt_prop)
    plt.clabel(CS13, inline=1,fontsize=10,colors='m',fmt=fmt_prop)
    plt.clabel(CS14, inline=1,fontsize=10,colors='r',fmt=fmt_prop)
    plt.clabel(CS15, inline=1,fontsize=10,colors='g',fmt=fmt_prop)
    plt.clabel(CS16, inline=1,fontsize=10,colors='b',fmt=fmt_prop)
    plt.clabel(CS17, inline=1,fontsize=10,colors='c',fmt=fmt_prop)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.xticks([0.365,0.370,0.375,0.380,0.385])
    plt.yticks([108,113,118,123,128])
    plt.plot([],[],'y',label=r'$\lambda = 12$')
    plt.plot([],[],'m',label=r'$\lambda = 13$')
    plt.plot([],[],'r',label=r'$\lambda = 14$')
    plt.plot([],[],'g',label=r'$\lambda = 15$')
    plt.plot([],[],'b',label=r'$\lambda = 17$')
    plt.plot([],[],'c',label=r'$\lambda = 18$')
    plt.plot(0.375,118,'kx',label='Reference')
    plt.title('RMS of $U_{dep}$ (kJ/mol)')
    plt.legend()
    
    subplot_2 = my_figure.add_subplot(2,1,2)
    
    contour_lines = [0.5, 1., 2., 4.,8.]
    contour_lines=[1.]
    fmt_prop = '%1.1f'
    
    CS12 = subplot_2.contour(sig_plot,eps_plot,RMS_Z12,contour_lines,colors='y')
    CS13 = subplot_2.contour(sig_plot,eps_plot,RMS_Z13,contour_lines,colors='m')
    CS14 = subplot_2.contour(sig_plot,eps_plot,RMS_Z14,contour_lines,colors='r')
    CS15 = subplot_2.contour(sig_plot,eps_plot,RMS_Z15,contour_lines,colors='g')
    CS16 = subplot_2.contour(sig_plot,eps_plot,RMS_Z17,contour_lines,colors='b')
    CS17 = subplot_2.contour(sig_plot,eps_plot,RMS_Z18,contour_lines,colors='c')
    plt.clabel(CS12, inline=1,fontsize=10,colors='y',fmt=fmt_prop)
    plt.clabel(CS13, inline=1,fontsize=10,colors='m',fmt=fmt_prop)
    plt.clabel(CS14, inline=1,fontsize=10,colors='r',fmt=fmt_prop)
    plt.clabel(CS15, inline=1,fontsize=10,colors='g',fmt=fmt_prop)
    plt.clabel(CS16, inline=1,fontsize=10,colors='b',fmt=fmt_prop)
    plt.clabel(CS17, inline=1,fontsize=10,colors='c',fmt=fmt_prop)
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.xticks([0.365,0.370,0.375,0.380,0.385])
    plt.yticks([108,113,118,123,128])
    plt.plot([],[],'y',label=r'$\lambda = 12$')
    plt.plot([],[],'m',label=r'$\lambda = 13$')
    plt.plot([],[],'r',label=r'$\lambda = 14$')
    plt.plot([],[],'g',label=r'$\lambda = 15$')
    plt.plot([],[],'b',label=r'$\lambda = 17$')
    plt.plot([],[],'c',label=r'$\lambda = 18$')
    plt.plot(0.375,118,'kx',label='Reference')
    plt.title('RMS of $Z$')
    plt.legend()

    plt.tight_layout(pad=0.2,rect=[0,0,1,1])
    
    subplot_1.text(0.363, 127, 'a)')
    subplot_2.text(0.363, 127, 'b)')  
    
    plt.show()
    
    my_figure.savefig('Contour_lambda.pdf')
    
#    print(np.mean(RMS_U13))
#    print(np.mean(RMS_U14))
#    print(np.mean(RMS_U15))
#    print(np.mean(RMS_U17))
#    print(np.mean(RMS_U18))
#    print(np.mean(RMS_Z13))
#    print(np.mean(RMS_Z14))
#    print(np.mean(RMS_Z15))
#    print(np.mean(RMS_Z17))
#    print(np.mean(RMS_Z18))
      
    parity = np.array([np.min(np.array([np.min(U_direct),np.min(U_12),np.min(U_18)])),np.max(np.array([np.max(U_direct),np.max(U_12),np.max(U_18)]))])
    
    plt.plot(U_direct,U_12,'y.',label='$\lambda=12$',alpha=0.2)  
    plt.plot(U_direct,U_13,'b.',label='$\lambda=13$',alpha=0.2)  
    plt.plot(U_direct,U_14,'r.',label='$\lambda=14$',alpha=0.2)
    plt.plot(U_direct,U_15,'g.',label='$\lambda=15$',alpha=0.2)
    plt.plot(U_direct,U_17,'c.',label='$\lambda=17$',alpha=0.2)
    plt.plot(U_direct,U_18,'m.',label='$\lambda=18$',alpha=0.2)
    plt.plot(parity,parity,'k',label='Parity')
    plt.xlabel('U (kJ/mol), Direct Simulation')
    plt.ylabel('U (kJ/mol), Predicted')
#    plt.legend()
    plt.show()
    
    parity = np.array([np.min(np.array([np.min(Z_direct),np.min(Z_12),np.min(Z_18)])),np.max(np.array([np.max(Z_direct),np.max(Z_12),np.max(Z_18)]))])
    
    plt.plot(Z_direct,Z_12,'y.',label='$\lambda=12$',alpha=0.2)  
    plt.plot(Z_direct,Z_13,'b.',label='$\lambda=13$',alpha=0.2)  
    plt.plot(Z_direct,Z_14,'r.',label='$\lambda=14$',alpha=0.2)
    plt.plot(Z_direct,Z_15,'g.',label='$\lambda=15$',alpha=0.2)
    plt.plot(Z_direct,Z_17,'c.',label='$\lambda=17$',alpha=0.2)
    plt.plot(Z_direct,Z_18,'m.',label='$\lambda=18$',alpha=0.2)
    plt.plot(parity,parity,'k',label='Parity')
    plt.xlabel('Z, Direct Simulation')
    plt.ylabel('Z, Predicted')
#    plt.legend()
    plt.show()
    
    lam_array = np.array([12.,13.,14.,15.,17.,18.])
    eps_max = np.zeros([len(Neff_direct),len(lam_array)])
    sig_max = np.zeros([len(Neff_direct),len(lam_array)])
    lam_max = np.zeros([len(Neff_direct),len(lam_array)])
    N_max = np.zeros([len(Neff_direct),len(lam_array)])
    Neff_all = np.array([Neff_12,Neff_13,Neff_14,Neff_15,Neff_17,Neff_18])
    RMS_Uall = np.array([RMS_U12,RMS_U13,RMS_U14,RMS_U15,RMS_U17,RMS_U18])
    RMS_Zall = np.array([RMS_Z12,RMS_Z13,RMS_Z14,RMS_Z15,RMS_Z17,RMS_Z18])
    
    for ilam in range(len(lam_array)):   
        for iState in range(len(Neff_direct)):
            imax = np.argmax(Neff_all[ilam][iState][:])
            #imax = np.argmin(RMS_Uall[ilam][iState][:])
            imax = np.argmin(RMS_Zall[ilam][iState][:])
            N_max[iState,ilam] = Neff_all[ilam][iState][imax]
            eps_max[iState,ilam] = eps_all[imax]
            sig_max[iState,ilam] = sig_all[imax]
            lam_max[iState,ilam] = lam_array[ilam]
            
    plt.plot(sig_max[:,0],eps_max[:,0],'yo',label=r'$\lambda=12$')
    plt.plot(sig_max[:,1],eps_max[:,1],'mo',label=r'$\lambda=13$')
    plt.plot(sig_max[:,2],eps_max[:,2],'ro',label=r'$\lambda=14$')
    plt.plot(sig_max[:,3],eps_max[:,3],'go',label=r'$\lambda=15$')
    plt.plot(sig_max[:,4],eps_max[:,4],'bo',label=r'$\lambda=17$')
    plt.plot(sig_max[:,5],eps_max[:,5],'co',label=r'$\lambda=18$')
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel(r'$\epsilon$ (K)')
    plt.xlim([0.365,0.385])
    plt.ylim([108,128])
    plt.xticks([0.365,0.370,0.375,0.380,0.385])
    plt.yticks([108,113,118,123,128])
    plt.legend()
    plt.show()
    
    plt.plot(lam_max[:,0],sig_max[:,0],'yo',label=r'$\lambda=12$')
    plt.plot(lam_max[:,1],sig_max[:,1],'mo',label=r'$\lambda=13$')
    plt.plot(lam_max[:,2],sig_max[:,2],'ro',label=r'$\lambda=14$')
    plt.plot(lam_max[:,3],sig_max[:,3],'go',label=r'$\lambda=15$')
    plt.plot(lam_max[:,4],sig_max[:,4],'bo',label=r'$\lambda=17$')
    plt.plot(lam_max[:,5],sig_max[:,5],'co',label=r'$\lambda=18$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\sigma$ (nm)')
    plt.xlim([11,19])
    plt.ylim([0.365,0.385])
    plt.xticks([12,13,14,15,17,18])
    plt.yticks([0.365,0.370,0.375,0.380,0.385])
    plt.legend()
    plt.show()
    
    sig_avg = np.array([np.mean(sig_max[:,0]),np.mean(sig_max[:,1]),np.mean(sig_max[:,2]),np.mean(sig_max[:,3]),np.mean(sig_max[:,4]),np.mean(sig_max[:,5])])    
    
    plt.plot(lam_array,sig_avg)
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\sigma$ (nm)')
    plt.xlim([11,19])
    plt.ylim([0.365,0.385])
    plt.xticks([12,13,14,15,16,17,18])
    plt.yticks([0.365,0.370,0.375,0.380,0.385])
    plt.show()
    
    for iplot in range(len(sig_max)):
        plt.plot(lam_array,sig_max[iplot,:])
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\sigma$ (nm)')
    plt.xlim([11,19])
    plt.ylim([0.365,0.385])
    plt.xticks([12,13,14,15,16,17,18])
    plt.yticks([0.365,0.370,0.375,0.380,0.385])
    plt.show()
    
    
    plt.plot(lam_array[0],np.mean(sig_max[:,0]),'yo',label=r'$\lambda=12$')
    plt.plot(lam_array[1],np.mean(sig_max[:,1]),'mo',label=r'$\lambda=13$')
    plt.plot(lam_array[2],np.mean(sig_max[:,2]),'ro',label=r'$\lambda=14$')
    plt.plot(lam_array[3],np.mean(sig_max[:,3]),'go',label=r'$\lambda=15$')
    plt.plot(lam_array[4],np.mean(sig_max[:,4]),'bo',label=r'$\lambda=17$')
    plt.plot(lam_array[5],np.mean(sig_max[:,5]),'co',label=r'$\lambda=18$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\sigma$ (nm)')
    plt.xlim([11,19])
    plt.ylim([0.365,0.385])
    plt.xticks([12,13,14,15,17,18])
    plt.yticks([0.365,0.370,0.375,0.380,0.385])
    plt.legend()
    plt.show()
        
    return
    
    
 
def main():
    
    fpathroot = 'parameter_space_LJ/'
    
    eps_all, sig_all, eps_matrix, sig_matrix = get_parameter_sets(fpathroot)
    
#    RMS_contours(eps_all,sig_all,fpathroot)
    RMS_contours_combined(eps_all,sig_all,fpathroot)
    

#    lambda_comparison(eps_all,sig_all,fpathroot)
#
    return
    
    for model_type in [reference,'Direct_simulation', 'MBAR_ref0', 'Zeroth_re','Constant_']:
        if model_type == 'TraPPE' or model_type == 'Potoff':
            U_ref, dU_ref, P_ref, dP_ref, Z_ref, dZ_ref, Neff_ref = compile_data(model_type,fpathroot)
            # Now I call a function that should calculate the error in the proper manner
            U_error, P_error = PCFR_error(U_ref,P_ref,model_type)
        elif model_type == 'Direct_simulation':
            U_direct, dU_direct, P_direct, dP_direct, Z_direct, dZ_direct, Neff_direct = compile_data(model_type,fpathroot)
            U_direct_Mie, dU_direct_Mie, P_direct_Mie, dP_direct_Mie, Z_direct_Mie, dZ_direct_Mie, Neff_direct_Mie = compile_data(model_type,'parameter_space_Mie16/')
        elif model_type == 'MBAR_ref0' or model_type == 'MBAR_ref1' or model_type == 'MBAR_ref8' or model_type == 'Lam15/MBAR_ref0' or model_type == 'Lam17/MBAR_ref0' or model_type == 'Lam14/MBAR_ref0' or model_type == 'Lam18/MBAR_ref0' or model_type == 'Lam13/MBAR_ref0' or model_type == 'Lam12/MBAR_ref0' or model_type == 'Lam12/MBAR_ref8' or model_type == 'Lam12/MBAR_ref11':
            U_MBAR, dU_MBAR, P_MBAR, dP_MBAR, Z_MBAR, dZ_MBAR, Neff_MBAR = compile_data(model_type,fpathroot)
            U_MBAR_Mie, dU_MBAR_Mie, P_MBAR_Mie, dP_MBAR_Mie, Z_MBAR_Mie, dZ_MBAR_Mie, Neff_MBAR_Mie = compile_data(model_type,'parameter_space_Mie16/')
            U_MBAR_LJhighEps, dU_MBAR_LJhighEps, P_MBAR_LJhighEps, dP_MBAR_LJhighEps, Z_MBAR_LJhighEps, dZ_MBAR_LJhighEps, Neff_MBAR_LJhighEps = compile_data('Lam12/MBAR_ref11','parameter_space_Mie16/')
        elif model_type == 'PCFR_ref0' or model_type == 'Zeroth_re':
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
    
    
    for prop in ['U','Z']:
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
        
#    contour_plot('U',eps_all,sig_all,U_direct,U_MBAR,U_PCFR,U_W1,U_rec)
#    contour_plot('P',eps_all,sig_all,P_direct,P_MBAR,P_PCFR,P_W1,P_rec)
#    contour_plot('Z',eps_all,sig_all,Z_direct,Z_MBAR,Z_PCFR,Z_W1,Z_rec)
    #contour_plot('Pdep',eps_all,sig_all,Pdep_direct,Pdep_MBAR,Pdep_PCFR)
    
    multi_prop_plot(U_direct[mask],U_MBAR[mask],U_PCFR[mask],U_W1[mask],U_rec[mask],Z_direct[mask],Z_MBAR[mask],Z_PCFR[mask],Z_W1[mask],Z_rec[mask],Neff,fpathroot)
    
#    multi_prop_multi_ref_plot(U_direct[mask],U_MBAR[mask],Z_direct[mask],Z_MBAR[mask],Neff,fpathroot)
    
#    multi_ref_LJ_Mie_plot(U_direct[mask],U_MBAR[mask],U_direct_Mie[mask],U_MBAR_Mie[mask],Z_direct[mask],Z_MBAR[mask],Z_direct_Mie[mask],Z_MBAR_Mie[mask],Neff_MBAR[mask],Neff_MBAR_Mie[mask])
    
#    multi_ref_LJ_Mie_LJhighEps_plot(U_direct[mask],U_MBAR[mask],U_direct_Mie[mask],U_MBAR_Mie[mask],U_MBAR_LJhighEps[mask],Z_direct[mask],Z_MBAR[mask],Z_direct_Mie[mask],Z_MBAR_Mie[mask],Z_MBAR_LJhighEps[mask],Neff_MBAR[mask],Neff_MBAR_Mie[mask],Neff_MBAR_LJhighEps[mask])
    
    #box_bar_state_plots(Neff_MBAR,Neff_min,Neff_small,mask_MBAR,mask_poor)
    #contours_Neff(Neff_MBAR,sig_all,eps_all,fpathroot)
    
#    contour_combined_plot(eps_all,sig_all,U_direct,Z_direct,U_MBAR,Z_MBAR,U_PCFR,Z_PCFR)
    
if __name__ == '__main__':
    '''
    Provides various plots to be considered in publication.
    '''
    main()