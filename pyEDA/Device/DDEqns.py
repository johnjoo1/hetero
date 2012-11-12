__all__=['InsulatorMaterial', 'SemiconductorMaterial', 'SiMaterial',  'Cu2OMaterial', 'ZnOMaterial',
         'SemiconductorRegionEqn', 'InsulatorRegionEqn', 'SemiconductorRegionEquEqn', 
         'OhmicBoundaryEqn', 'GateBoundaryEqn', 'OhmicBoundaryEquEqn', 'SimpleIFEqn']

from pyEDA.PDE.AutoDeriv import *
from pyEDA.PDE.NLEqns import *
from pyEDA.Mesh.Mesh1D import *
from pyEDA.FVMEqn.FVMEqn import *
import pyEDA.Device.PhysUnit as Unit

import numpy as np
import scipy
import math

class InsulatorMaterial(object):
    def __init__(self):
        self.affinity = 0.97*Unit.V
        self.eps = 3.9*Unit.eps0
        self.Eg = 9.0*Unit.V
        self.me_tnl = 0.5*Unit.me
        self.mh_tnl = 0.5*Unit.me

class SiMaterial(object):
    def __init__(self):
        self.eps = 11.7*Unit.eps0
        self.affinity = 4.17*Unit.V
        self.Eg = 1.12*Unit.V
        self.mun = 1000*Unit.cm*Unit.cm/Unit.V/Unit.s
        self.mup = 400*Unit.cm*Unit.cm/Unit.V/Unit.s
        self.tau = 1e-7*Unit.s
        self.ni = 1.45e10*pow(Unit.cm, -3.0)

        kb = 1.3806503e-23
        T = 300.
        hbar = 1.054571596e-34
        m_e=9.10938188*10**-31 #kg
        self.mh=((.50*m_e)**1.5 + (.15*m_e)**1.5)**(2./3.)
        self.me=.98*m_e 
        self.Nv=2.*(self.mh*kb*T/(2.*np.pi*hbar**2))**(3./2. )#m**-3
        self.Nc=2.*(self.me*kb*T/(2.*np.pi*hbar**2))**(3./2.) #m**-3
        
        self.ni_T = sqrt(self.Nc*self.Nv*exp(-self.Eg/(kb*T/1.602e-19))) * 1./100.**3.        
 

class SemiconductorMaterial(object):
    def __init__(self):
        self.eps = 11.7*Unit.eps0
        self.affinity = 4.17*Unit.V
        self.Eg = 1.12*Unit.V
        self.mun = 1000*Unit.cm*Unit.cm/Unit.V/Unit.s
        self.mup = 400*Unit.cm*Unit.cm/Unit.V/Unit.s
        self.tau = 1e-7*Unit.s
        self.ni = 1.45e10*pow(Unit.cm, -3.0)

        kb = 1.3806503e-23
        T = 300.
        hbar = 1.054571596e-34
        m_e=9.10938188*10**-31 #kg
        self.mh=((.50*m_e)**1.5 + (.15*m_e)**1.5)**(2./3.)
        self.me=.98*m_e 
        self.Nv=2.*(self.mh*kb*T/(2.*np.pi*hbar**2))**(3./2. )#m**-3
        self.Nc=2.*(self.me*kb*T/(2.*np.pi*hbar**2))**(3./2.) #m**-3
        
        self.ni_T = sqrt(self.Nc*self.Nv*exp(-self.Eg/(kb*T/1.602e-19))) * 1./100.**3.     

class Cu2OMaterial(object):
    def __init__(self):
        self.eps = 6.3*Unit.eps0
        self.affinity = 3.2*Unit.V
        self.Eg = 2.0*Unit.V
        
        kb = 1.3806503e-23
        T = 298.18
        hbar = 1.054571596e-34
        m_e=9.10938188*10**-31 #kg
        mh=.58*m_e #"Thin film deposition of Cu2O and application for solar cells"
        me=.99*m_e #Biccari thesis
        Nv=2.*(mh*kb*T/(2.*np.pi*hbar**2))**(3./2. )#m**-3
        Nc=2.*(me*kb*T/(2.*np.pi*hbar**2))**(3./2.) #m**-3

        self.ni = sqrt(Nc*Nv*exp(-self.Eg/(kb*T/1.602e-19))) * 1./100.**3.*pow(Unit.cm, -3.0)
        print 'ni = %f\n' %self.ni
        
        self.mun = 100*Unit.cm*Unit.cm/Unit.V/Unit.s
        self.mup = 90*Unit.cm*Unit.cm/Unit.V/Unit.s
        self.tau = 1e-7*Unit.s

class ZnOMaterial(object):
    def __init__(self):
        self.eps = 9.0*Unit.eps0 #Sze
        self.affinity = 4.2*Unit.V
        self.Eg = 3.3*Unit.V
        
        kb = 1.3806503e-23
        T = 298.18
        hbar = 1.054571596e-34
        m_e=9.10938188*10**-31 #kg
        mh=.59*m_e #"Zinc Oxide Nanostructures: Synthesis and Properties" Zhiyong Fan and Jia G. Lu
        me=.27*m_e #Sze
        Nv=2.*(mh*kb*T/(2.*np.pi*hbar**2))**(3./2. )#m**-3
        Nc=2.*(me*kb*T/(2.*np.pi*hbar**2))**(3./2.) #m**-3

        self.ni = sqrt(Nc*Nv*exp(-self.Eg/(kb*T/1.602e-19))) * 1./100.**3.*pow(Unit.cm, -3.0)
        print 'ni = %f\n' %self.ni
        
        self.mun = 200*Unit.cm*Unit.cm/Unit.V/Unit.s #"Zinc Oxide Nanostructures: Synthesis and Properties" Zhiyong Fan and Jia G. Lu
        self.mup = 30*Unit.cm*Unit.cm/Unit.V/Unit.s #"Zinc Oxide Nanostructures: Synthesis and Properties" Zhiyong Fan and Jia G. Lu
        self.tau = 1e-7*Unit.s
        
class SemiconductorRegionEqn(RegionEqn): 
    #RegionEqn defined in FVMEqn. Initializes eqnPerCell(self), cellEqn(self, state, cell), elemEqn(self, state, elem), initGuess(self, state, cell), and damp(self, state, cell, dx).
    def __init__(self):
        super(SemiconductorRegionEqn,self).__init__()
        
    def eqnPerCell(self): #defines 3 equations per cell
        return 3
    
    def cellEqn(self, state, cell):
        mtl = self.region.material
        ni,mun,mup,tau,eps = (mtl.ni, mtl.mun, mtl.mup, mtl.tau, mtl.eps)
        vol = cell.volume()
        vars = cell.vars #diode.regions[0].cells[n].vars
            
        dummy, n, p = state.getVars(vars)
        rho = Unit.e*(p-n+cell.fields['C'])
        Rsrh = (n*p-ni*ni)/(tau*(n+ni)+tau*(p+ni))
            
        state.setFunJac(vars[0], -rho*vol) 
        state.setFunJac(vars[1], -Rsrh*vol) 
        state.setFunJac(vars[2], -Rsrh*vol)
    
    def elemEqn(self, state, elem):
        mtl = self.region.material
        ni,mun,mup,tau,eps = (mtl.ni, mtl.mun, mtl.mup, mtl.tau, mtl.eps)
        VT = 0.0258*Unit.V
        
        varsA = elem.cells[0].vars
        varsB = elem.cells[1].vars
        
        h = elem.len
        Va, na, pa = state.getVars(varsA)
        Vb, nb, pb = state.getVars(varsB)
        
        nmid = na*aux2((Va-Vb)/2.0/VT) + nb*aux2((Vb-Va)/2.0/VT)
        dndx = aux1((Va-Vb)/2.0/VT) * (nb-na)/h
        pmid = pa*aux2((Vb-Va)/2.0/VT) + pb*aux2((Va-Vb)/2.0/VT)
        dpdx = aux1((Va-Vb)/2.0/VT) * (pb-pa)/h
    
        Fn =  mun * (-nmid* (Va-Vb)/h - VT*dndx)
        Fp =  mup * ( pmid* (Va-Vb)/h - VT*dpdx)
    
        state.setFunJac(varsA[0], (Va-Vb)/h*eps)
        state.setFunJac(varsA[1], -Fn)
        state.setFunJac(varsA[2], -Fp)
    
        state.setFunJac(varsB[0], (Vb-Va)/h*eps)
        state.setFunJac(varsB[1], Fn)
        state.setFunJac(varsB[2], Fp)

    def damp(self, state, cell, dx):
        vars = cell.vars
        V, n, p = state.getVars(vars)
        dV, dn, dp = dx[vars[0]], dx[vars[1]], dx[vars[2]]
        
        if dV>0.1:
            dV = 0.1*(1+math.log(dV/0.1))
        elif dV<-0.1:
            dV = -0.1*(1+math.log(-dV/0.1))
        if n+dn<0:
            dn = -0.99*n.getVal()
        if p+dp<0:
            dp = -0.99*p.getVal()
        dx[vars[0]], dx[vars[1]], dx[vars[2]] = (dV, dn, dp)


    def initGuess(self, state, cell):
        mtl = self.region.material
        ni = self.region.material.ni
        VT = 0.0258*Unit.V

        C = cell.fields['C']
        if 4*ni*ni < C*C/1e20:
            n = 0.5*(C + math.sqrt(C*C) + 4*ni*ni/(2*abs(C)) - (4*ni**2)**2/(8*abs(C)**3) +  (4*ni**2)**3/(16*abs(C)**5) +  5*(4*ni**2)**4/(128*abs(C)**7) )
        else:
            n=0.5*(C+math.sqrt(C*C+4*ni*ni))
        p=0.5*(-C+math.sqrt(C*C+4*ni*ni))
        phi=VT*math.log(n/ni) - mtl.Eg/2.0 - mtl.affinity

        state.setVar(cell.vars[0],phi)
        state.setVar(cell.vars[1], n)
        state.setVar(cell.vars[2], p)        


class InsulatorRegionEqn(RegionEqn):
    def __init__(self):
        super(InsulatorRegionEqn, self).__init__()
    
    def eqnPerCell(self):
        return 1

    def elemEqn(self, state, elem):
        material = self.region.material
        varA = elem.cells[0].vars[0]
        varB = elem.cells[1].vars[0]
        h = elem.len
        
        Va = state.getVar(varA)
        Vb = state.getVar(varB)
        state.setFunJac(varA, (Va-Vb)/h*material.eps)
        state.setFunJac(varB, (Vb-Va)/h*material.eps)
        
    def damp(self, state, cell, dx):
        vars = cell.vars
        V = state.getVars(vars)
        dV = dx[vars[0]]
        if dV>0.1:
            dV = 0.1*(1+math.log(dV/0.1))
        elif dV<-0.1:
            dV = -0.1*(1+math.log(-dV/0.1))
        dx[vars[0]] = dV

    def initGuess(self, state, cell):
        state.setVar(cell.vars[0], -5.0*Unit.V)


class OhmicBoundaryEqn(BoundaryEqn):
    def __init__(self):
        super(OhmicBoundaryEqn,self).__init__()
        self.voltage = 0.0
        
    def setVoltage(self, V):
        self.voltage = V

    def cellEqn(self, state, cell, dt=0):
        VT = 0.0258*Unit.V
        mtl = self.region.material
        C = cell.fields['C']
        V, n, p = state.getVars(cell.vars)
        
        ni = mtl.ni
#        nb=0.5*(C+math.sqrt(C*C+4*ni*ni))
#        pb=0.5*(-C+math.sqrt(C*C+4*ni*ni))
        
        if 4*ni*ni < C*C/1e20:
            nb = 0.5*(C + math.sqrt(C*C) + 4*ni*ni/(2*abs(C)) - (4*ni**2)**2/(8*abs(C)**3) +  (4*ni**2)**3/(16*abs(C)**5) +  5*(4*ni**2)**4/(128*abs(C)**7) )
        else:
            nb=0.5*(C+math.sqrt(C*C+4*ni*ni))
        pb=0.5*(-C+math.sqrt(C*C+4*ni*ni))
        Vb=self.voltage + VT*math.log(nb/ni) - mtl.Eg/2.0 - mtl.affinity

        state.resetEqn(cell.vars[0])
        state.resetEqn(cell.vars[1])
        state.resetEqn(cell.vars[2])
        state.setFunJac(cell.vars[0], V-Vb)
        state.setFunJac(cell.vars[1], n-nb)
        state.setFunJac(cell.vars[2], p-pb)

class SimpleIFEqn(InterfaceEqn):
    def __init__(self):
        super(SimpleIFEqn, self).__init__()
    
    def cellPairEqn(self, state, cell1, cell2):
        state.connectVar(cell1.vars[0], cell2.vars[0])

class GateBoundaryEqn(BoundaryEqn):
    def __init__(self):
        super(GateBoundaryEqn, self).__init__()
        self.V = 0
        self.workfunc = 4.17*Unit.V
    
    def setVoltage(self, V):
        self.V = V
        
    def voltage(self):
        return self.V

    def cellEqn(self, state, cell):
        phi = state.getVar(cell.vars[0])

        state.resetEqn(cell.vars[0])
        state.setFunJac(cell.vars[0], phi-(self.V-self.workfunc))


###################

class SemiconductorRegionEquEqn(RegionEqn):
    def __init__(self):
        super(SemiconductorRegionEquEqn,self).__init__()
        self.Vqfn = 0.0
        self.Vqfp = 0.0

    def eqnPerCell(self):
        return 1

    def setVqf(self, vqfn, vqfp):
        self.Vqfn = vqfn
        self.Vqfp = vqfp

    def cellEqn(self, state, cell):
        VT = 0.0258*Unit.V
        mtl = self.region.material
        var = cell.vars[0]
        
        Vi = state.getVar(var)+mtl.affinity+mtl.Eg/2.0 # mid-gap potential

        rho = Unit.e* ( cell.fields['C'] + mtl.ni*(exp((self.Vqfp-Vi)/VT)-exp((Vi-self.Vqfn)/VT)) )
        state.setFunJac(var, - rho*cell.volume())

    def elemEqn(self, state, elem):
        eps = self.region.material.eps

        varA = elem.cells[0].vars[0]
        varB = elem.cells[1].vars[0]
        h = elem.len

        Va = state.getVar(varA)
        Vb = state.getVar(varB)
        
        state.setFunJac(varA, (Va-Vb)/h*eps)
        state.setFunJac(varB, (Vb-Va)/h*eps)
 
    def damp(self, state, cell, dx):
        vars = cell.vars
        V = state.getVars(vars)
        dV = dx[vars[0]]
        if dV>0.1:
            dV = 0.1*(1+math.log(dV/0.1))
        elif dV<-0.1:
            dV = -0.1*(1+math.log(-dV/0.1))
        dx[vars[0]] = dV

    def initGuess(self, state, cell):
        mtl = self.region.material
        ni = self.region.material.ni
        VT = 0.0258*Unit.V

        C = cell.fields['C']
        n=0.5*(C+math.sqrt(C*C+4*ni*ni))
        p=0.5*(-C+math.sqrt(C*C+4*ni*ni))
        phi=VT*math.log(n/ni) - mtl.Eg/2.0 - mtl.affinity

        state.setVar(cell.vars[0], phi)


class OhmicBoundaryEquEqn(BoundaryEqn):
    def __init__(self):
        super(OhmicBoundaryEquEqn, self).__init__()
        self.voltage = 0.0
    
    def setVoltage(self, V):
        self.voltage = V
        
    def Voltage(self):
        return self.voltage

    def cellEqn(self, state, cell):
        VT = 0.0258*Unit.V
        mtl = self.region.material
        C = cell.fields['C']
        phi = state.getVar(cell.vars[0])

        ni = self.region.material.ni
        nb=0.5*(C+math.sqrt(C*C+4*ni*ni))
        pb=0.5*(-C+math.sqrt(C*C+4*ni*ni))
        Vb=self.voltage + VT*math.log(nb/ni) - mtl.Eg/2.0 - mtl.affinity

        state.resetEqn(cell.vars[0])
        state.setFunJac(cell.vars[0], phi-Vb)


