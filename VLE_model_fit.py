
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize  

class ITIC_VLE(object):
    
    def __init__(self, Tsat, rhol, rhov, Psat):
        
        # Ensures that the lowest density IC corresponds to maximum temperature 
        rhol = rhol[np.argmax(Tsat):]
        rhov = rhov[np.argmax(Tsat):]
        Psat = Psat[np.argmax(Tsat):]
        Tsat = Tsat[np.argmax(Tsat):]  
    
        # Ensures that no Psats of 0
        Tsat = Tsat[Psat>0]
        rhol = rhol[Psat>0]
        rhov = rhov[Psat>0]
        Psat = Psat[Psat>0]
        
        self.Tsat = Tsat
        self.rhol = rhol
        self.rhov = rhov
        self.Psat = Psat
        self.logPsat = np.log10(Psat)
        self.invTsat = 1000./Tsat
        self.beta = 0.326
        self.fitAntoine()
        self.fitRectScale()
        self.fitrhol()
        self.fitrhov()
        self.Pc = self.PsatHat(self.Tc)
        
    def SSE(self,data,model):
        SE = (data - model)**2
        SSE = np.sum(SE)
        return SSE 

    def logPAntoine(self,b,T):
        logP = b[0] - b[1]/(b[2] + T)
        return logP

    def guessAntoine(self):
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.invTsat,self.logPsat)
        guess = np.array([intercept,-slope*1000.,0])
        return guess
    
    def fitAntoine(self):
        Tfit = self.Tsat
        logPfit = self.logPsat
        SSElog = lambda b: self.SSE(logPfit,self.logPAntoine(b,Tfit))
        guess = self.guessAntoine()
        if len(Tfit) >= 3:
            bopt = minimize(SSElog,guess).x
        else:
            bopt = guess
        self.boptPsat = bopt
        return bopt
    
    def logPsatHat(self,T):
        bopt = self.boptPsat
        logPsatHat = self.logPAntoine(bopt,T)
        return logPsatHat

    def PsatHat(self,T):
        PsatHat = 10.**(self.logPsatHat(T))
        return PsatHat
    
    def rholRectScale(self,b,T):
        beta = self.beta
        rhol = b[0] + b[1]*(b[2] - T) + b[3]*(b[2] - T)**beta
        return rhol
    
    def rhovRectScale(self,b,T):
        beta = self.beta
        rhov = b[0] + b[1]*(b[2] - T) - b[3]*(b[2] - T)**beta
        return rhov
    
    def rhorRect(self,b,T):
        rhor = b[0] + b[1]*(b[2]-T)
        return rhor
    
    def rhosScale(self,b,T,beta = 0):
        if beta == 0: beta = self.beta
        rhos = b[3]*(b[2]-T)**beta
        return rhos
    
    def guessRectScale(self):
        self.rhor = (self.rhol + self.rhov)/2.
        rhocGuess = np.mean(self.rhor)
        TcGuess = np.max(self.Tsat)/0.85
        guess = np.array([rhocGuess,2,TcGuess,50])
        ylin = self.rhol - rhocGuess #Modify to get decent guesses
        xlin = lambda b: self.rholRectScale(np.array([rhocGuess,b[0],TcGuess,b[1]]),self.Tsat) - rhocGuess
        SSEguess = lambda b: self.SSE(ylin,xlin(b))
        guess = np.array([2,50])
        bnd = ((0,None),(0,None))
        bopt = minimize(SSEguess,guess,bounds=bnd).x
        guess = np.array([rhocGuess,bopt[0],TcGuess,bopt[1]])
        return guess
    
    def fitRectScale(self):
        Tfit = self.Tsat
        rhorfit = (self.rhol + self.rhov)/2.
        rhosfit = (self.rhol - self.rhov)/2.
        SSErhor = lambda b: self.SSE(rhorfit,self.rhorRect(b,Tfit))
        SSErhos = lambda b: self.SSE(rhosfit,self.rhosScale(b,Tfit))
        SSERectScale = lambda b: SSErhor(b) + SSErhos(b)
        guess = self.guessRectScale()
        if len(Tfit) >= 2:
            bnd = ((0,np.min(self.rhol)),(0,None),(np.max(Tfit),None),(0,None))
            bopt = minimize(SSERectScale,guess,bounds=bnd).x
        else:
            bopt = guess
        self.rhoc = bopt[0]
        self.Tc = bopt[2]
        self.boptRectScale = bopt
        return bopt
    
    def fitrhol(self):
        Tfit, rholfit = self.Tsat, self.rhol
        SSErhol = lambda b: self.SSE(rholfit,self.rholRectScale(b,Tfit))
        guess = self.boptRectScale
        #print(SSErhol(guess))
        if len(Tfit) >= 4: #rhol can get a better fit, although it extrapolates poorly
#            bnd = ((0,np.min(rholfit)),(0,None),(np.max(Tfit),None),(0,None))
            bnd = ((0,np.min(rholfit)),(0,None),(self.Tc,self.Tc),(0,None)) #Sets Tc to what RectScale got
            bopt = minimize(SSErhol,guess,bounds=bnd).x
        else:
            bopt = guess
        self.boptrhol = bopt
        #bopt = guess #If the optimization is not converging, this is a better option
        return bopt
    
    def rholHat(self,T):
        bopt = self.boptrhol
        rholHat = self.rholRectScale(bopt,T)
        return rholHat
    
    def fitrhov(self):
        Tfit, rhovfit = self.Tsat, self.rhov
        SSErhov = lambda b: self.SSE(rhovfit,self.rhovRectScale(b,Tfit))
        guess = self.boptRectScale
        #print(guess)
        #print(SSErhol(guess))
        if len(Tfit) >= 6: #rhov has problems fitting the data, better to just use from Rect Scale
            bnd = ((0,np.min(rhovfit)),(0,None),(np.max(Tfit),None),(0,None))
            bopt = minimize(SSErhov,guess,bounds=bnd).x
        else:
            bopt = guess
        self.boptrhov = bopt
        #bopt = guess #If the optimization is not converging, this is a better option
        return bopt
    
    def rhovHat(self,T):
        bopt = self.boptrhov
        #print(bopt)
        rhovHat = self.rhovRectScale(bopt,T)
        return rhovHat
    
    def rhorHat(self,T):
        bopt = self.boptRectScale
        rhorHat = self.rhorRect(bopt,T)
        return rhorHat       
        
    def bootstrapCriticals(self,plothist=False):
        
        nBoots = 100
        
        rhocBoots = np.zeros(nBoots)
        TcBoots = np.zeros(nBoots)
        PcBoots = np.zeros(nBoots)
        
        for iBoot in range(nBoots):
        
            ### First fit Tc and rhoc
            
            randbeta = np.random.uniform(0.3,0.35)
            
            randint = np.random.randint(0, len(self.Tsat),len(self.Tsat))
            Tfit = self.Tsat[randint]
            rholfit = self.rhol[randint]
            rhovfit = self.rhov[randint]
            rhorfit = (rholfit + rhovfit)/2.
            rhosfit = (rholfit - rhovfit)/2.
            SSErhor = lambda b: self.SSE(rhorfit,self.rhorRect(b,Tfit))
            SSErhos = lambda b: self.SSE(rhosfit,self.rhosScale(b,Tfit,beta=randbeta))
            SSERectScale = lambda b: SSErhor(b) + SSErhos(b)
            guess = self.guessRectScale()
            if len(Tfit) >= 2:
                bnd = ((0,np.min(self.rhol)),(0,None),(np.max(Tfit),None),(0,None))
                bopt = minimize(SSERectScale,guess,bounds=bnd).x
            else:
                bopt = guess
            rhocBoots[iBoot] = bopt[0]
            TcBoots[iBoot] = bopt[2]
            
            ### Then fit Pc
            
            logPfit = self.logPsat[randint]
            SSElog = lambda b: self.SSE(logPfit,self.logPAntoine(b,Tfit))
            guess = self.guessAntoine()
            if len(Tfit) >= 3:
                bopt = minimize(SSElog,guess).x
            else:
                bopt = guess
                
            logPcBoot = self.logPAntoine(bopt,TcBoots[iBoot])
            PcBoot = 10.**(logPcBoot)
            
            PcBoots[iBoot] = PcBoot
        
        if plothist:
            plt.hist(rhocBoots,bins=50,color='k')
            plt.xlabel(r'$\rho_{\rm c}$ (kg/m$^3$)')
            plt.show()
    
            plt.hist(TcBoots,bins=50,color='k')
            plt.xlabel(r'$T_{\rm c}$ (K)')
            plt.show()
            
            plt.hist(PcBoots,bins=50,color='k')
            plt.xlabel(r'$P_{\rm c}$ (bar)')
            plt.show()
        
        ### Assumes normal distribution of errors
        urhoc = 1.96 * np.std(rhocBoots)
        uTc = 1.96 * np.std(TcBoots)
        uPc = 1.96 * np.std(PcBoots)
        
        return urhoc, uTc, uPc

def main():
    
    # Values for Potoff model predicted from TraPPE samples
    # Taken from PCFR-ITIC
#    Tsat = np.array([276.778, 251.3329, 221.4212, 187.5534, 149.2056])
#    rhol = np.array([428.5761,471.4212,514.2969,557.1258,600.0072])
#    rhov = np.array([43.59846,21.13519,7.967988,1.86442,0.143268])
#    Psat = np.array([23.62316,12.02041,4.435127,0.938118,0.058897])
#    
    # Taken from MBAR-ITIC
#    Tsat = np.array([211.3574, 197.6162, 178.8908, 123.1425, 102.5166])
#    rhol = np.array([428.5761,471.4212,514.2969,557.1258,600.0072])
#    rhov = np.array([10.07994, 7.19709, 3.250055, 0.042836, 0.002265])
#    Psat = np.array([30.20705, 19.92031, 7.845697, 0.12644, 0.007994])
##    

    #ref0rr21 for scan at highEps with lam16. Perhaps IC is above critical point
#    Tsat = np.array([38.4551842,	152.0941454,	102.5972211,	68.34082747,	64.65333992])
#    rhol = np.array([428.5760696,	471.4212141,	514.2968664,	557.1257821,	600.0072033])
#    rhov = np.array([0,	0.514123698,	0.100480245,	4.62E-05,	1.33E-05])
#    Psat = np.array([0,	1.265481573,	0.376355903,	0.000244417,	7.41E-05])
    
    #ref0rr123 problematic for fitting because of bounds on guess
#    Tsat = np.array([95.42869923,	165.489684,	136.1076539,	81.68891776,	75.7025512])
#    rhol = np.array([428.5760696,	471.4212141,	514.2968664,	557.1257821,	600.0072033])
#    rhov = np.array([0.012913259,	2.580919739,	1.082752784,	3.55E-04,	7.02E-05])
#    Psat = np.array([0.04927028,	6.977463115,	3.689933732,	0.001572231,	3.35E-04])
    
    #ref0rr168 problematic for fitting, overflow in Psat
#    Tsat = np.array([38.68872081,	152.57235,	104.491397,	68.76362042,	64.86466136])
#    rhol = np.array([428.5760696,	471.4212141,	514.2968664,	557.1257821,	600.0072033])
#    rhov = np.array([0,	0.742605243,	0.089549641,	2.29E-05,	5.69E-06])
#    Psat = np.array([0,	1.857142769,	0.321819648,	0.000120528,	3.17E-05])

    # My Helium results:
    # Towhee
#    Tsat = np.array([6,7,8,9,10,11]) #[K]
#    rhol = np.array([0.31871202,0.30470999,0.289710787,0.273406827,0.254900377,0.231594877])*1000. #[kg/m3]
#    rhov = np.array([0.000179161,0.000713421,0.002167491,0.004683633,0.010768837,0.019240848])*1000. #[kg/m3]
#    Psat = np.array([2.221184667,10.17843633,34.23847833,80.274382,188.3929867,337.2498733])/100. #[bar]

    #Cassandra
    path = 'H:/Helium_ab_initio/Cassandra/Results/1400_10_md1/'
    VLCC = np.loadtxt(path+'VLCC.txt',skiprows=1)
    Psat = np.loadtxt(path+'Psat.txt',skiprows=1)[:,1]
    Tsat = VLCC[:,0]
    rhol = VLCC[:,1]
    rhov = VLCC[:,2]
    
    #Mostafa's data
#    path = 'H:/Publications/ITIC-paper_figure_scripts/Generated_ITIC_data/'
#    VLE_load = np.loadtxt(path+'TraPPE_C1.txt',skiprows=1)
#    Tsat = VLE_load[:,0] #[K]
#    Psat = VLE_load[:,1] #[MPa]
#    rhol = VLE_load[:,2] #[kg/m3]
#    rhov = VLE_load[:,3] #[kg/m3]
    
#    Psat*=0.8314472
#    
###    
##    #### Limit the range of data included in the fit
#    Tsat_low = 7.
#    Tsat_high = 11.
#    
#    rhol = rhol[Tsat>=Tsat_low]
#    rhov = rhov[Tsat>=Tsat_low]
#    Psat = Psat[Tsat>=Tsat_low]
#    Tsat = Tsat[Tsat>=Tsat_low]
#    
#    rhol = rhol[Tsat<=Tsat_high]
#    rhov = rhov[Tsat<=Tsat_high]
#    Psat = Psat[Tsat<=Tsat_high]
#    Tsat = Tsat[Tsat<=Tsat_high]

   ## TraPPE-Siepmann Validation values
#    Tsat = np.array([178,197,217,256,275,279,283,288])
#    rhol = np.array([551.2,526.2,498.4,434.2,393.7,383.5,372.6,358.9])
#    rhov = np.array([2.3,5.3,11.1,35.0,59.8,64.8,73.9,90.])
#    Psat = np.array([1.11,2.72,5.98,18.8,29.75,32,35.02,39.4])
#    


    if Tsat[0] < Tsat[-1]:
        ###Have to make sure that Tsat[0] is the highest value since this code was written for ITIC
        Tsat = Tsat[::-1]
        rhol = rhol[::-1]
        rhov = rhov[::-1]
        Psat = Psat[::-1]
        
    #######

    ITIC_fit = ITIC_VLE(Tsat,rhol,rhov,Psat)
    
    invTsat = ITIC_fit.invTsat
    logPsat = ITIC_fit.logPsat
    Tsat = ITIC_fit.Tsat
    rhol = ITIC_fit.rhol
    rhov = ITIC_fit.rhov
    rhor = ITIC_fit.rhor
    Psat = ITIC_fit.Psat
#    ITIC_fit.fitRectScale()
#    #Tc = ITIC_fit.boptRectScale[2] #Very poor Tc since only using rhol
    Tc = ITIC_fit.Tc
    Pc = ITIC_fit.Pc
    rhoc = ITIC_fit.rhoc
    
#    
##    invTsat = 1000./Tsat
##    logPsat = np.log10(Psat)
#
    Tplot = np.linspace(min(Tsat),Tc,1000)
    invTplot = 1000./Tplot
    logPsatplot = ITIC_fit.logPsatHat(Tplot)
    rholplot = ITIC_fit.rholHat(Tplot)
    rholRSplot = ITIC_fit.rholRectScale(ITIC_fit.boptRectScale,Tplot)
    Psatplot = ITIC_fit.PsatHat(Tplot)
    rhovplot = ITIC_fit.rhovHat(Tplot)
    rhovRSplot = ITIC_fit.rhovRectScale(ITIC_fit.boptRectScale,Tplot)
    rhorplot = ITIC_fit.rhorHat(Tplot)
    Psatsmoothed = ITIC_fit.PsatHat(Tsat)
    rholsmoothed = ITIC_fit.rholHat(Tsat)
#    print(Psatplot)
#    print(Psatsmoothed)

    urhoc, uTc, uPc = ITIC_fit.bootstrapCriticals()
    ulogPc = (np.log10(Pc+uPc) - np.log10(Pc-uPc))/2.
    uinvTc = (1000./(Tc-uTc)-1000./(Tc+uTc))/2.
    
    Mw_He = 4.0026 #[gm/mol]
             
    Pc_Kofke = 0.95*10. #[bar]
    rhoc_Kofke = 27.5 * Mw_He #[kg/m3]
    Tc_Kofke = 13.05 #[K]
    
    uPc_Kofke = 0.2*10. #[bar]
    urhoc_Kofke = 2.5 * Mw_He #[kg/m3]
    uTc_Kofke = 0.05 #[K]    

    ulogPc_Kofke = (np.log10(Pc_Kofke+uPc_Kofke) - np.log10(Pc_Kofke-uPc_Kofke))/2.
    uinvTc_Kofke = (1000./(Tc_Kofke-uTc_Kofke)-1000./(Tc_Kofke+uTc_Kofke))/2.     
    
    print('Critical temperature = '+str(np.round(Tc,2))+r'$ \pm$ '+str(np.round(uTc,2))+' K, Critical Pressure = '+str(np.round(Pc,3))+r'$ \pm$ '+str(np.round(uPc,3))+' bar, Critical Density = '+str(np.round(rhoc,1))+' $\pm$ '+str(np.round(urhoc,2))+' (kg/m$^3$).')
    
    plt.figure(figsize=[8,8])
    
    plt.plot(invTsat,logPsat,'ro',label='GEMC')
    plt.plot(invTplot,logPsatplot,'k',label='Fit, GEMC')
    plt.errorbar(1000./Tc,np.log10(Pc),xerr=uinvTc,yerr=ulogPc,fmt='b*',label='Critical, GEMC')
    plt.errorbar(1000./Tc_Kofke,np.log10(Pc_Kofke),xerr=uinvTc_Kofke,yerr=ulogPc_Kofke,mfc='None',fmt='gs',label='Critical, Kofke')
    plt.xlabel('1000/T (K)')
    plt.ylabel('log(Psat/bar)')
    plt.legend()
    plt.show()
    
    plt.plot(Tsat,logPsat,'ro')
    plt.plot(Tplot,logPsatplot,'k')
    plt.errorbar(Tc,np.log10(Pc),xerr=uTc,yerr=ulogPc,fmt='b*')
    plt.xlabel('Temperature (K)')
    plt.ylabel('log(Psat/bar)')
    plt.show()
    
    plt.plot(Tsat,Psat,'ro')
#    plt.plot(Tsat,Psatsmoothed,'gx')
    plt.plot(Tplot,Psatplot,'k')
    plt.errorbar(Tc,Pc,xerr=uTc,yerr=uPc,fmt='b*')
    plt.errorbar(Tc_Kofke,Pc_Kofke,xerr=uTc_Kofke,yerr=uPc_Kofke,mfc='None',fmt='gs')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Psat (bar)')
    plt.show()

    plt.figure(figsize=[8,8])

    plt.plot(rhol,Tsat,'ro',label='GEMC')
    plt.plot(rhov,Tsat,'ro')
    plt.plot(rhor,Tsat,'ro')
#    plt.plot(rholsmoothed,Tsat,'gx')
#    plt.plot(rholplot,Tplot,'g')
#    plt.plot(rhovplot,Tplot,'g')
    plt.plot(rhorplot,Tplot,'k',label='Fit, GEMC')
    plt.plot(rholRSplot,Tplot,'k')
    plt.plot(rhovRSplot,Tplot,'k')
    plt.errorbar(rhoc,Tc,xerr=urhoc,yerr=uTc,fmt='b*',label='Critical, GEMC')
    plt.errorbar(rhoc_Kofke,Tc_Kofke,xerr=urhoc_Kofke,yerr=uTc_Kofke,fmt='gs',mfc='None',label='Critical, Kofke')
    plt.xlabel('Density (kg/m3)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.show()
    
    Rg = 8.3144598e-5 #[m3 bar / K / mol]
    Vv = Mw_He/rhov/1000. #[m3/mol]
    Zv = Psat*Vv/Rg/Tsat
    
    Zvplot = Psatplot * Mw_He / rhovRSplot / Rg / Tplot / 1000.
    
    Vc = Mw_He/rhoc/1000.
    Zc = Pc * Vc / Rg / Tc
    
    uZc = Zc * np.sqrt((uPc/Pc)**2. + (urhoc/rhoc)**2. + (uTc/Tc)**2.)
    
    Vc_Kofke = Mw_He/rhoc_Kofke/1000.
    Zc_Kofke = Pc_Kofke * Vc_Kofke / Rg / Tc_Kofke
    
    uZc_Kofke = Zc_Kofke * np.sqrt((uPc_Kofke/Pc_Kofke)**2. + (urhoc_Kofke/rhoc_Kofke)**2. + (uTc_Kofke/Tc_Kofke)**2.)
    
    print(Zv,Zc,uZc)
    
    Tplot_Zv = Tplot[Zvplot>0]
    Zvplot = Zvplot[Zvplot>0]
    
    Tplot_Zv = Tplot_Zv[Zvplot<1]
    Zvplot = Zvplot[Zvplot<1]
    
    plt.figure(figsize=[8,8])

    plt.plot(Tsat,Zv,'ro',label='GEMC')
    plt.plot(Tplot_Zv,Zvplot,'k',label='Fit, GEMC')
    plt.errorbar(Tc,Zc,xerr=uTc,yerr=uZc,fmt='b*',label='Critical, GEMC')
    plt.errorbar(Tc_Kofke,Zc_Kofke,xerr=uTc_Kofke,yerr=uZc_Kofke,fmt='gs',mfc='None',label='Critical, Kofke')
    plt.ylabel('Compressibility Factor')
    plt.xlabel('Temperature (K)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    main()