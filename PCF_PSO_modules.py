from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
import CoolProp
from scipy.optimize import minimize
import sys
import abc

model = 'Mie'

eps_ref = 117.181 #[K]
sig_ref = 0.380513 #[nm]
lam_ref = 15.6796
      
fname = 'H:/PCF-PSO/RDF_Iteration_4_corrected_gromacs_Non_VLE.txt'
RDFs_highP = np.loadtxt(fname,delimiter='\t')
RDFs_highP = RDFs_highP[1:,:] 

T_highP = np.array([135, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])

U_L_highP_ens = np.array([-15.37811761, -15.17739261, -14.56776761, -13.99989261, -13.46889261, -12.96789261, -12.49721761, -12.03651761, -11.60244261, -11.16036761, -10.74169261])

rhoL_highP = np.array([12.61698]*len(T_highP)) #[1/nm**3]

R_g = 8.314472e-3 # [kJ/mol/K]

N = 400 # Number of molecules

## Simulation constants
#

## Gromacs uses a different format
#    
N_sites = 1
N_pair = N_sites**2
N_int = 4.

## RDF bins

r_c = 1.4 # [nm]

r = np.linspace(0.002,1.4,num=700) #[nm]

dr = r[1] - r[0]
#
#r_c_plus_ref = r_c / sig_ref
#
#r_plus_ref = r/sig_ref
#
#dr_plus_ref = dr/sig_ref

class BasePCFR(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, r, RDF, rho, Temp, ref):
        self.r = r
        self.RDF = RDF
        self.rho = rho
        self.Temp = Temp
        self.ref = ref
        
    @abc.abstractmethod
    def get_Unb(self):
        """ Return the nonbonded potential """
        return None
    
    def get_Unb_ref(self):
        """ Return the nonbonded potential for the reference system """
        if not self.ref:
            print('This is the reference system')
            self.Unb_ref = self.get_Unb()
        else:
            print('Different reference system')
            self.Unb_ref = self.ref.get_Unb()
        return self.Unb_ref
    
    def get_deltaUnb(self):
        deltaUnb = self.get_Unb() - self.get_Unb_ref()
        deltaUnb[deltaUnb>1e3] = 1e3
        deltaUnb[deltaUnb<-1e3] = -1e3
        return deltaUnb
    
    @abc.abstractmethod
    def get_Ucorr(self):
        """ Return the nonbonded correction to U """
        return None
    
    def get_RDF(self):
        """ Return the radial distribution function """
        return self.RDF
    
    def get_rho(self):
        """ Return the vector of density """
        return self.rho
    
    def smooth_RDF(self):
        pass
    
    def pred_RDF(self):
        """ Predicts the RDF based on the Unb and Unb_ref"""
        RDF_hat = self.get_RDF()
        for iTemp, Temp_i in enumerate(self.Temp):
            rescale = np.exp(-self.get_deltaUnb()/Temp_i)
            plt.plot(self.r,rescale)
            plt.ylim([0,2])
            plt.show()
            RDF_hat[:,iTemp] *= rescale        
        return RDF_hat
       
    def calc_Uint(self,RDF_pair):
        Uint = (self.get_Unb()*RDF_pair*self.r**2*dr).sum() # [K*nm^3]
        return Uint
    
    def calc_Utotal(self,RDF_pair):
        Uint = self.calc_Uint(RDF_pair)
        Ucorr = self.get_Ucorr()
        Utotal = Uint + Ucorr
        Utotal *= 2*math.pi
        return Utotal
    
    def calc_Uhat(self,RDF_state):
        Uhat = 0
        
        for n in range(0,N_pair):
            
            RDF_pair = RDF_state[:,n]
            
            Uhat += self.calc_Utotal(RDF_pair)
            
        return Uhat
    
    def calc_Ureal(self):
        
        Ureal = []
        #for rho_state, Temp_state in zip(self.rho, self.Temp):
        for istate, rho_state in enumerate(self.rho):
            RDF_state = self.RDF[:,istate*N_pair:istate*N_pair+N_pair]
            Ureal.append(R_g * rho_state * self.calc_Uhat(RDF_state) * N_int)
        return Ureal
    
#def RDF_0(U,T):
#    return np.exp(-U/T)
#
#def RDF_hat_calc(RDF_real, RDF_0_ref,RDF_0_hat):
#    RDF_0_ref_zero = RDF_0_ref[RDF_0_ref<1e-2] # Using exactly 0 leads to some unrealistic ratios at very close distances
#    RDF_0_ref_non_zero = RDF_0_ref[RDF_0_ref>1e-2]
#    RDF_real_non_zero = RDF_real[RDF_0_ref>1e-2]
#    RDF_ratio_non_zero = RDF_real_non_zero / RDF_0_ref_non_zero
#    RDF_ones = np.zeros(len(RDF_0_ref_zero))
#    RDF_ratio = np.append(RDF_ones,RDF_ratio_non_zero)
#    RDF_hat = RDF_ratio * RDF_0_hat
#    #RDF_hat = RDF_real # Override so that not scaling
##    
##    print(len(np.array(RDF_0_ref)))
##    print(len(RDF_0_hat))
##    print(len(RDF_real))
##    print(len(RDF_hat))
#    
##    if print_RDFs == 1:
##        plt.scatter(r,RDF_real,label='Ref')
##        plt.scatter(r,RDF_0_ref,label='Ref_0')
##        plt.scatter(r,RDF_0_hat,label='Hat_0')
##        plt.scatter(r,RDF_hat,label='Hat')
##        plt.legend()
##        plt.show()
#        #plt.scatter(r,RDF_ratio)
#        #plt.show()
#        
#        #plt.scatter(r,-np.log(RDF_ratio))
#        #plt.show()
#    
#    return RDF_hat
    
#    def calc_Udev(self):
#        Udev = self.Uens - self.get_Uref()
#        return Udev
#    
#    def calc_Ufinal(self):
#        Ufinal = c
        
#def deltaU(eps,sig,lam,RDF_all=RDFs_ref,rho_v=rho_v,rho_L=rho_L,Temp=Temp):
#    
#    U_L = np.empty(len(Temp))
#    U_v = U_L.copy()
#    
#    for t in range(0, len(Temp)):
#        rhov_Temp = rho_v[t]
#        rhoL_Temp = rho_L[t]
#        
#        RDF_Temp_L = RDF_all[:,8*t:8*t+N_pair]
#        RDF_Temp_v = RDF_all[:,8*t+N_pair:8*t+2*N_pair]
#        
#        U_L_Temp = U_hat(eps,sig,lam,RDF_Temp_L)
#        U_v_Temp = U_hat(eps,sig,lam,RDF_Temp_v)
#        
#        U_L_Temp *= R_g * rhoL_Temp
#        U_v_Temp *= R_g * rhov_Temp
#        
#        U_L[t] = U_L_Temp
#        U_v[t] = U_v_Temp
#           
#    deltaU = U_v - U_L
#    return deltaU, U_L, U_v

   
#U_L_error = U_L_ens - deltaU(eps_ref,sig_ref,lam_ref)[1]
#U_v_error = U_v_ens - deltaU(eps_ref,sig_ref,lam_ref)[2]
#
#def deltaU_hat(eps,sig,lam):
#        
#    deltaU_hat = deltaU(eps,sig,lam)[0] + deltaU_error
#    return deltaU_hat
#
#def U_L_hat(eps,sig,lam):
#            
#    U_L_hat = deltaU(eps,sig,lam)[1] + U_L_error #Easier to just compare to residual
#    #U_L_hat = deltaU(eps,sig,lam)[1] + U_intra + U_L_error
#    return U_L_hat
        
class LennardJones(BasePCFR):
    def __init__(self, r, RDF, rho, T, epsilon, sigma, n=12., m =6., ref=None, **kwargs):
        BasePCFR.__init__(self, r, RDF, rho, T, ref)
        for key in kwargs:
            if key == 'place holder':
                pass                
        self.epsilon = epsilon
        self.sigma = sigma
        self.n = n
        self.m = m
        self.C = (n/(n-m))*(n/m)**(m/(n-m)) # The normalization constant for the Mie potential
        if n == 12.:
            assert (self.C - 4.) < 1e-3, 'The coefficient for LJ 12-6 should be 4'
        self.r_c_plus = r_c / sigma
        
    def get_Unb(self):
        """
    The LJ potential calculated in [K]. 
    Note that r and sigma must be in the same units. 
    The exponents (n and m) are set to default values of 12 and 6    
    """
        epsilon, sigma, n, m, r, C = self.epsilon, self.sigma, self.n, self.m, self.r, self.C
        U = C*epsilon*((r/sigma)**-n - (r/sigma)**-m)
        U[U>1e8] = np.max(U[U<1e8]) # No need to store extremely large values of U
        return U        
    
    def get_Ucorr(self):
        epsilon, sigma, n, r_c_plus, C = self.epsilon, self.sigma, self.n, self.r_c_plus, self.C
        U = C*epsilon*((1./(n-3.))*r_c_plus**(3.-n) - (1./3.) * r_c_plus **(-3.))*sigma**3. #[K * nm^3]
        return U
        
#class Exp6(BasePCFR):
#    def __init__(self, epsilon, alpha, r_m, RDF, rho, T):
#        BasePCFR.__init__(self, RDF, rho, T)
#        self.epsilon = epsilon
#        self.alpha = alpha
#        self.r_m = r_m
#    
#    def get_ur(self):
#        return [1,2,3]        

LJref = LennardJones(r,RDFs_highP, rhoL_highP, T_highP, eps_ref, sig_ref, lam_ref)
Udev = U_L_highP_ens - LJref.calc_Ureal()
LJPotoff = LennardJones(r,RDFs_highP, rhoL_highP, T_highP, 121.25, 0.3783, 16., ref=LJref)
UPotoff = LJPotoff.calc_Ureal() + Udev
LJTraPPE = LennardJones(r,RDFs_highP,rhoL_highP,T_highP,98.,0.375,12., ref=LJref)
#e6 = Exp6(100, 0.3, 4, [], rho, T)

#plt.plot(r,LJref.get_Unb_ref(),label='Ref Ref')
#plt.plot(r,LJPotoff.get_Unb_ref(),label='Potoff Ref')
#plt.plot(r,LJTraPPE.get_Unb_ref(),label='TraPPE Ref')
#plt.plot(r,LJPotoff.get_Unb(),label='Potoff')
#plt.plot(r,LJTraPPE.get_Unb(),label='TraPPE')
plt.plot(r,LJref.get_deltaUnb(),label='Ref deltaU')
plt.plot(r,LJPotoff.get_deltaUnb(),label='Potoff deltaU')
plt.plot(r,LJTraPPE.get_deltaUnb(),label='TraPPE deltaU')
plt.xlim([0,0.5])
plt.ylim([-2000,2000])
plt.legend()
plt.show()

RDFref = LJref.get_RDF()
RDFPotoff = LJPotoff.pred_RDF()
RDFTraPPE = LJTraPPE.pred_RDF()

plt.plot(r,RDFref[:,0],'r')
plt.plot(r,RDFPotoff[:,0],'b')
plt.plot(r,RDFTraPPE[:,0],'g')
plt.show()

class FFComparator(object):
    def __init__(self, reference,*args,**kwargs):
        self.reference = reference
        self.Udev = U_L_highP_ens - self.reference.calc_Ureal()
        for key in kwargs:
            if key == 'model':
                if kwargs[key] == 'LJ':
                    self.epsilon = kwargs['epsilon']
                    self.sigma = kwargs['sigma']
                    self.lam = kwargs['lam']
                    perturbed = LennardJones(self.reference.r, self.reference.RDF, self.reference.rho, self.reference.Temp, self.epsilon,self.sigma,self.lam) 
#### Maybe I can just modify LennardJones so that if it is a reference it does things slightly differently. Or maybe
# they can stay the same since the deltaU would be zero and so the RDF should be constant and Udev would be as well. 
### Yeah, I can just have ref as a keyword and if it is a reference then it becomes its' own reference, effectively.        
    def process_one(self, epsilon_i_sigma_i):
        epsilon_i, sigma_i = epsilon_i_sigma_i
#    def process_one(self, parameters):
#        self.reference.process(parameters)
        U = 30 # From these parameters
        P = 300
        Uhat = self.reference.calc_Utotal()
        return Uhat-U+epsilon_i, P
    
comp = FFComparator(LJref,epsilon=121.25,sigma=0.3783,lam=16.,model='LJ')
import itertools
inputs = itertools.product([100,200,300], [0.3,0.4,0.5])

#from  multiprocessing import Pool
#p = Pool(5)
outputs = map(comp.process_one, inputs)
print(list(outputs))

#
#
#class ref_ff:
#    def __init__(self, *params):
#        self.params = None
##    def __init__(self, *coeffs):
##        self.coeffs = coeffs
#
## The general concept for this code is:
## 1) Input:
##    a) Reference system
##        i) PCF
##        ii) Force field parameters
##    b) Target system
##        i) Force field parameters
##    c) State points
##        i) Temperature
##        ii) Density
##    d) Compound details
##        i) Molecular weight
##       ii) Number/type of interaction sites
##    e) Specific type of PCFR
## 2) Output:
##    a) U
##    b) P
#     

## RDF bins
#
#r_c = 1.4 #[nm]
#
#r = np.linspace(0.002,1.4,num=700) #[nm]
#
#dr = r[1] - r[0]
#
#r_c_plus_ref = r_c / sig_ref
#
#r_plus_ref = r/sig_ref
#
#dr_plus_ref = dr/sig_ref
#
#def RDF_smooth(RDFs):
#    RDFs_smoothed = np.empty(RDFs.shape)
#    for i in range(RDFs.shape[1]):
#        RDF = RDFs[:,i]
#        RDF_non_zero = RDF[RDF>0]
#        RDF_zero = RDF[RDF==0]
#        if len(RDF_non_zero) > 5:
#            # Smooth the first two and last two points differently
#            
#            RDF_smoothed = RDF_zero
#            
#            RDF_smoothed = np.append(RDF_smoothed,1./70 * (69.*RDF_non_zero[0] + 4.*RDF_non_zero[1] - 6.*RDF_non_zero[2] + 4.*RDF_non_zero[3] - RDF_non_zero[4]))
#            RDF_smoothed = np.append(RDF_smoothed,1./35 * (2.*RDF_non_zero[0] + 27.*RDF_non_zero[1] + 12.*RDF_non_zero[2] - 8.*RDF_non_zero[3] + 2*RDF_non_zero[4]))
#            
#            for j in range(2,len(RDF_non_zero)-2):
#                RDF_smoothed = np.append(RDF_smoothed,1./35 * (-3.*RDF_non_zero[j-2] + 12.*RDF_non_zero[j-1] + 17.*RDF_non_zero[j] +12.*RDF_non_zero[j+1] - 3*RDF_non_zero[j+2]))
#    
#            RDF_smoothed = np.append(RDF_smoothed,1./35 * (2.*RDF_non_zero[-1] + 27.*RDF_non_zero[-2] + 12.*RDF_non_zero[-3] - 8.*RDF_non_zero[-4] + 2*RDF_non_zero[-5]))
#            RDF_smoothed = np.append(RDF_smoothed,1./70 * (69.*RDF_non_zero[-1] + 4.*RDF_non_zero[-2] - 6.*RDF_non_zero[-3] + 4.*RDF_non_zero[-4] - RDF_non_zero[-5]))    
#                        
#            RDF_smoothed[RDF_smoothed<0]=0
#            RDFs_smoothed[:,i] = RDF_smoothed
#        else:
#            RDFs_smoothed[:,i] = RDF
#        if Temp[int(i/2)] == min(Temp) and i%2 == 1: # Treat the vapor phase at lowest Temp differently
#            RDF_smoothed = 1./70 * (69.*RDF[0] + 4.*RDF[1] - 6.*RDF[2] + 4.*RDF[3] - RDF[4])
#            RDF_smoothed = np.append(RDF_smoothed,1./35 * (2.*RDF[0] + 27.*RDF[1] + 12.*RDF[2] - 8.*RDF[3] + 2*RDF[4]))
#            
#            for j in range(2,len(RDF)-2):
#                RDF_smoothed = np.append(RDF_smoothed,1./35 * (-3.*RDF[j-2] + 12.*RDF[j-1] + 17.*RDF[j] +12.*RDF[j+1] - 3*RDF[j+2]))
#    
#            RDF_smoothed = np.append(RDF_smoothed,1./35 * (2.*RDF[-1] + 27.*RDF[-2] + 12.*RDF[-3] - 8.*RDF[-4] + 2*RDF[-5]))
#            RDF_smoothed = np.append(RDF_smoothed,1./70 * (69.*RDF[-1] + 4.*RDF[-2] - 6.*RDF[-3] + 4.*RDF[-4] - RDF[-5]))    
#                        
#            RDF_smoothed[RDF_smoothed<0]=0
#            RDFs_smoothed[:,i] = RDF_smoothed
#            #RDFs_smoothed[:,i] = RDF # Just leave it the same
#    return RDFs_smoothed
#
#def U_Mie(r, e_over_k, sigma, n = 12., m = 6.):
#    """
#    The Mie potential calculated in [K]. 
#    Note that r and sigma must be in the same units. 
#    The exponents (n and m) are set to default values of 12 and 6    
#    """
#    C = (n/(n-m))*(n/m)**(m/(n-m)) # The normalization constant for the Mie potential
#    U = C*e_over_k*((r/sigma)**-n - (r/sigma)**-m)
#    return U
#
#def U_Corr(e_over_k, sigma, r_c_plus, n = 12., m = 6.): #I need to correct this for m != 6
#    C = (n/(n-m))*(n/m)**(m/(n-m)) # The normalization constant for the Mie potential
#    U = C*e_over_k*((1./(n-3))*r_c_plus**(3-n) - (1./3) * r_c_plus **(-3))*sigma**3 #[K * nm^3]
#    return U
#
#def U_total(r, e_over_k, sigma, r_c_plus, RDF, dr,  n = 12., m = 6.):
#    U_int = (U_Mie(r, e_over_k, sigma,n,m)*RDF*r**2*dr).sum() # [K*nm^3]
#    U_total = U_int + U_Corr(e_over_k,sigma,r_c_plus,n,m)
#    U_total *= 2*math.pi
#    return U_total
#
#def r_min_calc(sig, n=12., m=6.):
#    r_min = (n/m*sig**(n-m))**(1./(n-m))
#    return r_min
#
#def U_hat(eps_pred,sig_pred,lam_pred,RDF_Temp,r_ref=r,sig_ref=sig_ref,r_c=r_c,r_plus_ref = r_plus_ref, dr_plus_ref = dr_plus_ref, r_c_plus_ref = r_c_plus_ref):
#    
#    U_hat = 0 
#                      
#    for n in range(0,N_pair):
#        
#        RDF = RDF_Temp[:,n]
#                
#        U_hat += U_total(r_plus_ref*sig_pred,eps_pred,sig_pred,r_c_plus_ref,RDF,dr_plus_ref*sig_pred,lam_pred) # Constant r_plus
#    return U_hat
#    
#    
#def deltaU(eps,sig,lam,RDF_all=RDFs_ref,rho_v=rho_v,rho_L=rho_L,Temp=Temp):
#    
#    U_L = np.empty(len(Temp))
#    U_v = U_L.copy()
#    
#    for t in range(0, len(Temp)):
#        rhov_Temp = rho_v[t]
#        rhoL_Temp = rho_L[t]
#        
#        RDF_Temp_L = RDF_all[:,8*t:8*t+N_pair]
#        RDF_Temp_v = RDF_all[:,8*t+N_pair:8*t+2*N_pair]
#        
#        U_L_Temp = U_hat(eps,sig,lam,RDF_Temp_L)
#        U_v_Temp = U_hat(eps,sig,lam,RDF_Temp_v)
#        
#        U_L_Temp *= R_g * rhoL_Temp
#        U_v_Temp *= R_g * rhov_Temp
#        
#        U_L[t] = U_L_Temp
#        U_v[t] = U_v_Temp
#           
#    deltaU = U_v - U_L
#    return deltaU, U_L, U_v
#
#deltaU_error = deltaU_ens - deltaU(eps_ref,sig_ref,lam_ref)[0]
#U_L_error = U_L_ens - deltaU(eps_ref,sig_ref,lam_ref)[1]
#U_v_error = U_v_ens - deltaU(eps_ref,sig_ref,lam_ref)[2]
#
#def deltaU_hat(eps,sig,lam):
#        
#    deltaU_hat = deltaU(eps,sig,lam)[0] + deltaU_error
#    return deltaU_hat
#
#def U_L_hat(eps,sig,lam):
#            
#    U_L_hat = deltaU(eps,sig,lam)[1] + U_L_error #Easier to just compare to residual
#    #U_L_hat = deltaU(eps,sig,lam)[1] + U_intra + U_L_error
#    return U_L_hat
#
#def U_v_hat(eps,sig,lam):
#            
#    U_v_hat = deltaU(eps,sig,lam)[2] + U_v_error # Easier to just compare to residual
#    #U_v_hat = deltaU(eps,sig,lam)[2] + U_intra + U_v_error
#    return U_v_hat
#
#def objective(params):
#    eps = params[0]
#    sig = params[1]
#    lam = params[2]
#    dev_deltaU = deltaU_hat(eps,sig,lam) - RP_deltaU
#    dev_U_L = U_L_hat(eps,sig,lam) - RP_U_L_res
#    dev_U_v = U_v_hat(eps,sig,lam) - RP_U_v_res
#    dev_dev_U = dev_U_L - dev_U_v
#    #dev_deltaU = dev_deltaU[Temp>110]
#    #dev_deltaU = dev_deltaU[0:-1]
#    SSE_deltaU = np.sum(np.power(dev_deltaU,2))
#    SSE_U_L = np.sum(np.power(dev_U_L,2))
#    SSE_U_v = np.sum(np.power(dev_U_v,2))
#    SSE_dev_dev = np.sum(np.power(dev_dev_U,2))
#    SSE = 0
#    SSE += SSE_deltaU #+ SSE_dev_dev
#    #SSE += SSE_U_L + SSE_U_v
#    #SSE += SSE_U_L
#    print('eps = '+str(eps)+', sig = '+str(sig)+', lam = '+str(lam)+' SSE ='+str(SSE))
#    return SSE
#
#def constraint1(params):
#    sig = params[1]
#    lam = params[2]
#    if lam < lam_ref:
#        return sig_ref - sig
#    elif lam >= lam_ref:
#        return sig - sig_ref
#    
#def constraint2(params):
#    sig = params[1]
#    lam = params[2]
#    if lam < lam_ref:
#        return r_min_calc(sig,lam) - r_min_ref
#    elif lam >= lam_ref:
#        return r_min_ref - r_min_calc(sig,lam)
#
#sig_TraPPE = 0.375 #[nm]
#lam_TraPPE = 12
#r_min_TraPPE = r_min_calc(sig_TraPPE,lam_TraPPE)
#
#def constraint3(params):
#    sig = params[1]
#    lam = params[2]
#    if lam < lam_TraPPE:
#        return sig_TraPPE - sig
#    elif lam >= lam_TraPPE:
#        return sig - sig_TraPPE
#    
#def constraint4(params):
#    sig = params[1]
#    lam = params[2]
#    if lam < lam_TraPPE:
#        return r_min_calc(sig,lam) - r_min_TraPPE
#    elif lam >= lam_TraPPE:
#        return r_min_TraPPE - r_min_calc(sig,lam)
#    
#lam_fixed = 15    
#
#def constraint5(params):
#    lam = params[2]
#    return lam - lam_fixed
#
#def constraint6(params):
#    lam = params[2]
#    return lam % 2
#    
#guess = [eps_ref, sig_ref, 14.]
#con1 = {'type':'ineq','fun':constraint1}
#con2 = {'type':'ineq','fun':constraint2}
#con3 = {'type':'ineq','fun':constraint3}
#con4 = {'type':'ineq','fun':constraint4}
#con5 = {'type':'eq','fun':constraint5}
#con6 = {'type':'eq','fun':constraint6}
#cons = [con3,con4]
##cons = []
#bnds = ((50,200),(0.3,0.45),(10,20))
#
#sol = minimize(objective,guess,method='SLSQP',bounds=bnds,constraints=cons)#,options={'eps':1e-6})
#params_opt = sol.x
#eps_opt = sol.x[0]
#sig_opt = sol.x[1]
#lam_opt = sol.x[2]
#r_min_opt = r_min_calc(sig_opt,lam_opt)
#
#print(params_opt)
#
##Taken from optimization_Mie_ITIC
#guess_scaled = [1.,1.,1.]
#objective_scaled = lambda params_scaled: objective(params_scaled*guess)
#bnds_scaled = ((50/eps_ref,200/eps_ref),(0.3/sig_ref,0.45/sig_ref),(10./14.,20./14.)) 
#sol = minimize(objective_scaled,guess_scaled,method='SLSQP',bounds=bnds_scaled,options={'eps':1e-5,'maxiter':50}) #'eps' accounts for the algorithm wanting to take too small of a step change for the Jacobian that Gromacs does not distinguish between the different force fields
#params_scaled_opt = sol.x
#eps_opt = sol.x[0]*eps_ref
#sig_opt = sol.x[1]*sig_ref
#lam_opt = sol.x[2]*14.
#
#eps_It_1 = 122.94636444
#sig_It_1 = 0.37620538
#lam_It_1 = 14.59923016
#
##print(deltaU_hat(eps_ref,sig_ref,lam_ref))
##print(deltaU(eps_ref,sig_ref,lam_ref))
#
#r_plot = np.linspace(0.8*sig_ref,2*sig_ref,num=1000)
#
#Mie_ref = U_Mie(r_plot,eps_ref,sig_ref,lam_ref)
#Mie_opt = U_Mie(r_plot,eps_opt,sig_opt,lam_opt)
#Mie_Potoff = U_Mie(r_plot,121.25,0.3783,16)
#
#plt.plot(r_plot,Mie_ref,label='Initial')
#plt.plot(r_plot,Mie_opt,label='Optimal')
#plt.plot(r_plot,Mie_Potoff,label='Potoff')
#plt.xlabel('r (nm)')
#plt.ylabel('U (K)')
#plt.ylim([-1.1*np.max(np.array([eps_ref,eps_opt])),1.1*np.max(np.array([eps_ref,eps_opt]))])
#plt.xlim([np.min(r_plot),np.max(r_plot)])
#plt.legend()
#plt.show()
#
#T_plot = np.linspace(Temp.min(), Temp.max())
#RP_HVP_plot = (CP.PropsSI('HMOLAR','T',T_plot,'Q',1,'REFPROP::'+compound) - CP.PropsSI('HMOLAR','T',T_plot,'Q',0,'REFPROP::'+compound)) / 1000 #[kJ/mol]
#RP_deltaU_plot = (CP.PropsSI('UMOLAR','T',T_plot,'Q',1,'REFPROP::'+compound) - CP.PropsSI('UMOLAR','T',T_plot,'Q',0,'REFPROP::'+compound)) / 1000 #[kJ/mol]
#RP_U_L_plot = CP.PropsSI('UMOLAR','T',T_plot,'Q',0,'REFPROP::'+compound) / 1000 #[kJ/mol]
#RP_U_v_plot = CP.PropsSI('UMOLAR','T',T_plot,'Q',1,'REFPROP::'+compound) / 1000 #[kJ/mol]
#RP_U_ig_plot = CP.PropsSI('UMOLAR','T',T_plot,'D',0,'REFPROP::'+compound) / 1000 #[kJ/mol]
#RP_U_L_res_plot = RP_U_L_plot - RP_U_ig_plot
#RP_U_v_res_plot = RP_U_v_plot - RP_U_ig_plot
#
#deltaU_opt = deltaU_hat(eps_opt,sig_opt,lam_opt)
#deltaU_It_1 = deltaU_hat(eps_It_1,sig_It_1,lam_It_1)
#deltaU_ref = deltaU_hat(eps_ref,sig_ref,lam_ref)
#U_L_opt = U_L_hat(eps_opt,sig_opt,lam_opt)
#U_L_ref = U_L_hat(eps_ref,sig_ref,lam_ref)
#U_v_opt = U_v_hat(eps_opt,sig_opt,lam_opt)
#U_v_ref = U_v_hat(eps_ref,sig_ref,lam_ref)
#
#plt.scatter(Temp,deltaU_ref,label='Initial')
#plt.scatter(Temp,deltaU_opt,label='Optimal')
#plt.plot(T_plot,RP_deltaU_plot,label='RefProp')
##plt.plot(Temp,YE_deltaU_fit,label='YE')
#plt.xlabel('T (K)')
#plt.ylabel(r'$\Delta U_v \left(\frac{kJ}{mol}\right)$')
#plt.legend()
#plt.show()
#
#plt.scatter(Temp,U_L_ref,label='Initial')
#plt.scatter(Temp,U_L_opt,label='Optimal')
#plt.plot(T_plot,RP_U_L_res_plot,label='RefProp')
##plt.plot(Temp,YE_U_L_fit,label='YE')
#plt.xlabel('T (K)')
#plt.ylabel(r'$U_l \left(\frac{kJ}{mol}\right)$')
#plt.legend()
#plt.show()
#
#plt.scatter(Temp,U_v_ref,label='Initial')
#plt.scatter(Temp,U_v_opt,label='Optimal')
#plt.plot(T_plot,RP_U_v_res_plot,label='RefProp')
##plt.plot(Temp,YE_U_v_fit,label='YE')
#plt.xlabel('T (K)')
#plt.ylabel(r'$U_v \left(\frac{kJ}{mol}\right)$')
#plt.legend()
#plt.show()
#
#def main():
#    
#    parser = argparse.ArgumentParser()
#    parser.add_argument("-opt","--optimizer",type=str,choices=['fsolve','steep','LBFGSB','leapfrog','scan','points','SLSQP'],help="choose which type of optimizer to use")
#    parser.add_argument("-prop","--properties",type=str,nargs='+',choices=['rhoL','Psat','rhov','P','U','Z'],help="choose one or more properties to use in optimization" )
#    args = parser.parse_args()
#    if args.optimizer:
#        eps_opt, sig_opt, lam_opt = call_optimizers(args.optimizer,args.properties)
#    else:
#        print('Please specify an optimizer type')
#        eps_opt = 0.
#        sig_opt = 0.
#        lam_opt = 0.
#
#if __name__ == '__main__':
#    
#    main()