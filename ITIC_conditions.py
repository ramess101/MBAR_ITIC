# -*- coding: utf-8 -*-
"""
Determines the ITIC conditions at which to perform simulations

"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from scipy.optimize import minimize, minimize_scalar, fsolve

#Before running script run, "pip install pymbar, pip install CoolProp"

compound='Neopentane'
#REFPROP_path='/home/ram9/REFPROP-cmake/build/' #Change this for a different system

#CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH,REFPROP_path)

Mw = CP.PropsSI('M','REFPROP::'+compound) #[kg/mol]
RP_TC = CP.PropsSI('TCRIT','REFPROP::'+compound)
RP_Tmin =  CP.PropsSI('TMIN','REFPROP::'+compound)

Nmol = 800.
N_A = 6.0221409e23 #Avogadro's number [1/mol]
conv_m3tonm3 = 1e-27 #[m3/nm3]

Tr_IT = 1.2

T_IT = np.round(Tr_IT * RP_TC)

print(T_IT*np.ones(9))

T_IC_min = np.round(np.max([0.45*RP_TC,RP_Tmin]))

print(T_IC_min)

rho_IC_min = CP.PropsSI('D','T',T_IC_min,'Q',0,'REFPROP::'+compound) #[kg/m3]  

drho = rho_IC_min/7.

rho_IT = np.ones(9)*drho
         
for i in np.arange(1,5):
    rho_IT[i] = rho_IT[i-1]+drho
          
for i in np.arange(5,9):
    rho_IT[i] = rho_IT[i-1]+drho/2.

print(rho_IT)

T_IC0 = np.zeros(2)
T_IC1 = np.zeros(2)
T_IC2 = np.zeros(2)
T_IC3 = np.zeros(2)
T_IC4 = np.zeros(2)

def calc_Tsat_rhol(rhol):
    Tsat_guess = np.average([RP_Tmin,RP_TC])
    dev = lambda Tsat: (CP.PropsSI('D','T',Tsat,'Q',0,'REFPROP::'+compound) - rhol)**2.
    opt = minimize(dev,Tsat_guess)
    return np.round(opt.x)

rho_IC0 = rho_IT[-1]
rho_IC1 = rho_IT[-2]
rho_IC2 = rho_IT[-3]
rho_IC3 = rho_IT[-4]
rho_IC4 = rho_IT[-5]


T_IC0[0] = calc_Tsat_rhol(rho_IC0)
T_IC0[1] = np.round(1./np.average([1./T_IC0[0],1./T_IT]))

T_IC1[0] = calc_Tsat_rhol(rho_IC1)
T_IC1[1] = np.round(1./np.average([1./T_IC1[0],1./T_IT]))

T_IC2[0] = calc_Tsat_rhol(rho_IC2)
T_IC2[1] = np.round(1./np.average([1./T_IC2[0],1./T_IT]))

T_IC3[0] = calc_Tsat_rhol(rho_IC3)
T_IC3[1] = np.round(1./np.average([1./T_IC3[0],1./T_IT]))

T_IC4[0] = calc_Tsat_rhol(rho_IC4)
T_IC4[1] = np.round(1./np.average([1./T_IC4[0],1./T_IT]))

print(T_IC0)
print(T_IC1)
print(T_IC2)
print(T_IC3)
print(T_IC4)

T_IC = np.array([T_IC0,T_IC1,T_IC2,T_IC3,T_IC4])

def convert_rhol_Lbox(rhol):
    nrho = rhol/Mw*N_A*conv_m3tonm3 #[1/nm3]
    Vbox= Nmol / nrho #[nm3]
    Lbox = Vbox**(1./3.)
    return Lbox

Lbox_IT = convert_rhol_Lbox(rho_IT)
Lbox_IC = Lbox_IT[::-1][:5]

print(Lbox_IT)
print(Lbox_IC)

T_plot = np.linspace(RP_Tmin,RP_TC,1000)
RP_rhol_plot = CP.PropsSI('D','T',T_plot,'Q',0,'REFPROP::'+compound) #[kg/m3]
RP_rhov_plot = CP.PropsSI('D','T',T_plot,'Q',1,'REFPROP::'+compound) #[kg/m3] 

plt.plot(rho_IT,T_IT*np.ones(9),'ro',label='Isotherm')
plt.plot(rho_IC0*np.ones(2),T_IC0,'bs',label='Isochore 0')
plt.plot(rho_IC1*np.ones(2),T_IC1,'bs',label='Isochore 1')
plt.plot(rho_IC2*np.ones(2),T_IC2,'bs',label='Isochore 2')
plt.plot(rho_IC3*np.ones(2),T_IC3,'bs',label='Isochore 3')
plt.plot(rho_IC4*np.ones(2),T_IC4,'bs',label='Isochore 4')
plt.plot(RP_rhol_plot,T_plot,'k',label='REFPROP')
plt.plot(RP_rhov_plot,T_plot,'k',label='REFPROP')
plt.xlabel('Density (kg/m3)')
plt.ylabel('Temperature (K)')
plt.show()

f = open('T_IT','w')
for i in range(9):
    f.write(str(T_IT)+' ')
f.close()

f = open('rho_IT','w')
for rho in rho_IT:
    f.write(str(rho)+' ')
f.close()

f = open('Lbox_IT','w')
for Lbox in Lbox_IT:
    f.write(str(Lbox)+' ')
f.close()

f = open('Lbox_IC','w')
for Lbox in Lbox_IC:
    f.write(str(Lbox)+' ')
f.close()

iIC = 0
f = open('T_IC','w')
for Temp in T_IC:
    f.write('Tic'+str(iIC)+'=('+str(Temp[0])+' '+str(Temp[1])+')\n')
    iIC += 1
f.close()