# -*- coding: utf-8 -*-
"""
Plotting the average Neff for lam=12 and lam=16
"""

import numpy as np
import matplotlib.pyplot as plt

nReruns = 441

Neff_ratio = np.array([0.6,0.65,0.6,0.9,0.95,0.99,0.98,0.85,0.9,0.75,0.65,0.8,0.75,0.7,0.9,0.9,0.99,0.7,0.95])
    
sig_plot = np.linspace(0.365,0.385,21)
eps_plot = np.linspace(88,128,41)
           
font = {'size' : '16'}
plt.rc('font',**font)

fmt_prop = '%1.0f'
        
Neff_avg = np.zeros(41*21)
Neff_min = np.zeros(41*21)

for irr in range(nReruns):
    # The first irr is actually the reference
    MBAR_irr = np.loadtxt('C:/calc1_Backup/Ethane/Gromacs/LJscan/ref0/MBAR/lam12/MBAR_ref0rr'+str(irr+1)+'_lam12')
    Neff_irr = MBAR_irr[:,-1]
    Neff_irr -= 1.
    Neff_irr = np.multiply(Neff_ratio,Neff_irr.T).T
    Neff_irr += 1.
    Neff_avg[irr] = Neff_irr.mean()
    Neff_min[irr] = Neff_irr.min()
    
for irr in range(nReruns-21):
    # The first irr is actually the reference
    try:
        MBAR_irr = np.loadtxt('C:/calc1_Backup/Ethane/Gromacs/LJscan/ref0/MBAR/lam12/MBAR_rr'+str(irr+21)+'_highEps',skiprows=1)
        Neff_irr = MBAR_irr[:,-1]
        Neff_irr -= 1.
        Neff_irr = np.multiply(Neff_ratio,Neff_irr.T).T
        Neff_irr += 1.
        Neff_avg[irr+nReruns] = Neff_irr.mean()
        Neff_min[irr+nReruns] = Neff_irr.min()
    except:
        print('File for '+str(irr+21)+' did not exist, TC too low')

contour_lines = [50,200]
           
f = plt.figure(figsize=(8,8))
    
#CS3= plt.contour(sig_plot,eps_plot,Neff_avg.reshape(41,21),contour_lines,colors='b',linestyle='solid')
#plt.clabel(CS3,inline=1,fontsize=16,colors='b',fmt=fmt_prop)
#CS4= plt.contour(sig_plot,eps_plot,Neff_min.reshape(21,21),contour_lines,colors='c')
#plt.clabel(CS4,inline=1,fontsize=10,colors='c',fmt=fmt_prop)
plt.xlabel(r'$\sigma_{\rm CH_3}$ (nm)',fontsize=26)
plt.ylabel(r'$\epsilon_{\rm CH_3}$ (K)',fontsize=26)
plt.xticks([0.365,0.370,0.375,0.380,0.385])
plt.plot(0.375,98.,'kx',markersize=10,label=r'$\theta_{\rm ref} = \theta_{\rm TraPPE-UA}$')#, $\lambda = 12$')

lam_range = [12,13,14,15,16,17,18]
contour_lines = [[50,100,400],[50],[50],[20],[10],[6],[5],[50,100,400]]
color_scheme = ['b','g','c','m','r','orange','brown','purple']
line_scheme = ['solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid']
line_scheme_alt = ['-','--','-.',':','-','--','-.',':','-']

for ilam, lam_rr in enumerate(lam_range):
    
    Neff_avg = np.zeros(41*21)
    Neff_min = np.zeros(41*21)
    
    if lam_rr == 16:
        
        for irr in range(nReruns):
            # The first irr is actually the reference
            MBAR_irr = np.loadtxt('C:/calc1_Backup/Ethane/Gromacs/LJscan/ref0/MBAR_noP_basis/lam16/MBAR_ref0rr'+str(irr+1)+'_lam16')
            Neff_irr = MBAR_irr[:,-1]
            Neff_irr -= 1.
            Neff_irr = np.multiply(Neff_ratio,Neff_irr.T).T
            Neff_irr += 1.
            Neff_avg[irr] = Neff_irr.mean()
            Neff_min[irr] = Neff_irr.min()
        
        for irr in range(nReruns-21):
            # The first irr is actually the reference
            MBAR_irr = np.loadtxt('C:/calc1_Backup/Ethane/Gromacs/LJscan/ref0/MBAR/lam16/MBAR_ref0rr'+str(irr+22)+'_lam16_highEps')
            Neff_irr = MBAR_irr[:,-1]
            Neff_irr -= 1.
            Neff_irr = np.multiply(Neff_ratio,Neff_irr.T).T
            Neff_irr += 1.
            Neff_avg[irr+nReruns] = Neff_irr.mean()
            Neff_min[irr+nReruns] = Neff_irr.min()
        
    else:
    
        for irr in range(len(Neff_avg)):
            try:
                MBAR_irr = np.loadtxt('H:/Ethane_Neff/TraPPE_ref/lam'+str(lam_rr)+'/MBAR_rr'+str(irr),skiprows=1)
                Neff_irr = MBAR_irr[:,-1]
                Neff_irr -= 1.
                Neff_irr = np.multiply(Neff_ratio,Neff_irr.T).T
                Neff_irr += 1.
                Neff_avg[irr] = Neff_irr.mean()
                Neff_min[irr] = Neff_irr.min()
            except:
                print('File for '+str(irr)+' for lam = '+str(lam_rr)+' did not exist, TC too low')
        
    CS1= plt.contour(sig_plot,eps_plot,Neff_avg.reshape(41,21),contour_lines[ilam],colors=color_scheme[ilam],linestyles=line_scheme[ilam])
    if lam_rr >= 15:
        plt.clabel(CS1,inline=1,fontsize=16,colors=color_scheme[ilam],fmt=fmt_prop,manual=list([(0.368,93)]))
    elif lam_rr == 13:
        plt.clabel(CS1,inline=1,fontsize=16,colors=color_scheme[ilam],fmt=fmt_prop,manual=list([(0.373,110)]))
    elif lam_rr == 14:
        plt.clabel(CS1,inline=1,fontsize=16,colors=color_scheme[ilam],fmt=fmt_prop,manual=list([(0.373,103)]))
    else:
        plt.clabel(CS1,inline=1,fontsize=16,colors=color_scheme[ilam],fmt=fmt_prop)
    plt.plot([],[],color=color_scheme[ilam],linestyle=line_scheme_alt[ilam],label=r'$\lambda =$'+str(lam_rr))
    
    print(Neff_avg.max())
#
#Neff_avg = np.zeros(41*21)
#Neff_min = np.zeros(41*21)
#
#for irr in range(nReruns):
#    # The first irr is actually the reference
#    MBAR_irr = np.loadtxt('C:/calc1_Backup/Ethane/Gromacs/LJscan/ref0/MBAR_noP_basis/lam16/MBAR_ref0rr'+str(irr+1)+'_lam16')
#    Neff_irr = MBAR_irr[:,-1]
#    Neff_irr -= 1.
#    Neff_irr = np.multiply(Neff_ratio,Neff_irr.T).T
#    Neff_irr += 1.
#    Neff_avg[irr] = Neff_irr.mean()
#    Neff_min[irr] = Neff_irr.min()
#
#for irr in range(nReruns-21):
#    # The first irr is actually the reference
#    MBAR_irr = np.loadtxt('C:/calc1_Backup/Ethane/Gromacs/LJscan/ref0/MBAR/lam16/MBAR_ref0rr'+str(irr+22)+'_lam16_highEps')
#    Neff_irr = MBAR_irr[:,-1]
#    Neff_irr -= 1.
#    Neff_irr = np.multiply(Neff_ratio,Neff_irr.T).T
#    Neff_irr += 1.
#    Neff_avg[irr+nReruns] = Neff_irr.mean()
#    Neff_min[irr+nReruns] = Neff_irr.min()
#
#contour_lines = [10]
#    
#CS1= plt.contour(sig_plot,eps_plot,Neff_avg.reshape(41,21),contour_lines,colors='r',linestyles='solid')
#plt.clabel(CS1,inline=1,fontsize=16,colors='r',fmt=fmt_prop)
#CS2= plt.contour(sig_plot,eps_plot,Neff_min.reshape(42,21),contour_lines,colors='g')
#plt.clabel(CS2,inline=1,fontsize=10,colors='g',fmt=fmt_prop)
plt.xlabel(r'$\sigma_{\rm CH_3}$ (nm)',fontsize=27)
plt.ylabel(r'$\epsilon_{\rm CH_3}$ (K)',fontsize=27)
plt.xticks([0.365,0.370,0.375,0.380,0.385])
plt.yticks([88,98,108,118,128])
#plt.plot([],[],'b-',label=r'$\lambda = 12$')
#plt.plot([],[],'g-.',label=r'$\lambda = 13$')
#plt.plot([],[],'r-',label=r'$\lambda = 16$')
plt.title('Average Number of Effective Samples')  
plt.legend()

f.savefig('Average_Neff_lam.pdf')
#f.savefig('Average_Neff_lam12_lam16_IT.pdf')
#f.savefig('Average_Neff_lam12_lam16_IC.pdf')