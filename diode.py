'''
Created on Jun 18, 2009

@author: hash
'''

from pyEDA.PDE.AutoDeriv import *
from pyEDA.PDE.NLEqns import *
from pyEDA.Mesh.Mesh1D import *
from pyEDA.FVMEqn.FVMEqn import *
import pyEDA.Device.PhysUnit as Unit
from pyEDA.Device.DDEqns import *

import numpy as np
import scipy
import math

class Silicon(SemiconductorMaterial): #SemiconductorMaterial defined in Device/DDEqns
    #defines eps, affinity, Eg, ni, mup, mun, tau
    #to access, var = Silicon
    #   var.Eg
    def __init__(self):
        super(Silicon, self).__init__()

class Diode(Mesh1D): #Mesh1D defined in Mesh/Mesh1D
    #var=Diode()
    #var.regions lists:
        #cell at node(x)
        #Elem1D between cell(x) and cell(x+1)
    def __init__(self):
        NN = 500

        xx = np.linspace(-1e-4*Unit.cm, 1e-4*Unit.cm, NN+1)
        Mesh1D.__init__(self, xx, 
                        rgns=[(0, NN, 'silicon')],
                        bnds=[(0,'anode'),(NN,'cathode')])

        self.setRegionMaterial('silicon', Silicon())

        def doping(x):
            if x<0:
                return -1e18*pow(Unit.cm,-3)
            elif x>0:
                return 1e16*pow(Unit.cm,-3)
            else:
                return 0.0
            
        self.setFieldByFunc(0, 'C', doping)

class DiodeEqns(FVMEqns): #FVMEqns init calls 'diode' (device) which calls Mesh1D to set up mesh, material
    def __init__(self, device):
        super(DiodeEqns, self).__init__(device)

        self.eqnSi = SemiconductorRegionEqn() #SemiconductorRegionEqn defined in Device/DDEqns. does not actually define any equations yet.  just initializes eqnPerCell(self), cellEqn(self, state, cell), elemEqn(self, state, elem), initGuess(self, state, cell), and damp(self, state, cell, dx). You still have to call the functions in the class.  The functions are where the equations are defined.
        self.setRegionEqn('silicon', self.eqnSi) #function in FVMEqn/FVMEqn/class FVMEqns.  sets eqns in eqnSi to region named 'silicon'.

        self.bcAnode = OhmicBoundaryEqn()
        self.bcCathode = OhmicBoundaryEqn()
        self.setBoundaryEqn('anode', self.bcAnode)
        self.setBoundaryEqn('cathode', self.bcCathode)
        self.setupEqns()

if __name__ == '__main__':
    diode = Diode() #sets up mesh, material, and doping
    diodeEqn = DiodeEqns(diode) #sets up equations to be solved
    diodeEqn.initGuess()
    diodeEqn.bcAnode.setVoltage(0.0)
    diodeEqn.solve()

    diodeEqn.bcAnode.setVoltage(0.0)
    diodeEqn.solve()
    
    vi = diode.getVarIdx('silicon', 0) # potential in Substrate
    print '-----------------------'
    print 'Potential in silicon:'
    print diodeEqn.state.x[vi]

    cmc=pow(Unit.cm,-3)
    ni = diode.getVarIdx('silicon', 1) # electron conc in Substrate
    print '-----------------------'
    print 'Elec. conc. in silicon:'
    print diodeEqn.state.x[ni]/cmc

    pi = diode.getVarIdx('silicon', 2) # hole conc in Substrate
    print '-----------------------'
    print 'Hole conc. in silicon:'
    print diodeEqn.state.x[pi]/cmc
    
    #plot potential
    import pylab
    v = diodeEqn.state.x[vi]
    NN=500
    xx = np.linspace(-1, 1, NN+1)
    pylab.figure(1)
    pylab.plot(xx,v)
    pylab.title('Potential')
    print 'Vbi = %f' %(v[0]-v[-1])
    
    #plot electron and hole concentrations
    nn = diodeEqn.state.x[ni]/cmc
    pp = diodeEqn.state.x[pi]/cmc
    pylab.figure(2)
    pylab.plot(xx,nn)
    pylab.plot(xx,pp)
    pylab.title('Electron and hole concentration')
    
    #plot charge density profile
    rho = []
    pylab.figure(3)
    for i,x in enumerate(xx):
        doping = diode.regions[0].cells[i].fields['C']/pow(Unit.cm,-3)
        n_e = nn[i]
        p_e = pp[i]
        charge = Unit.e*(p_e-n_e+doping)
        rho.append(charge)
    pylab.plot(xx,rho, 'o')
   

