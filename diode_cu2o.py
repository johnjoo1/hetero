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

class Silicon(SiMaterial): #SemiconductorMaterial defined in Device/DDEqns
    #defines eps, affinity, Eg, ni, mup, mun, tau
    #to access, var = Silicon
    #   var.Eg
    def __init__(self):
        super(Silicon, self).__init__()

class Cu2O(Cu2OMaterial): #SemiconductorMaterial defined in Device/DDEqns
    #defines eps, affinity, Eg, ni, mup, mun, tau
    #to access, var = Silicon
    #   var.Eg
    def __init__(self):
        super(Cu2O, self).__init__()
        
class ZnO(ZnOMaterial): #SemiconductorMaterial defined in Device/DDEqns
    #defines eps, affinity, Eg, ni, mup, mun, tau
    #to access, var = Silicon
    #   var.Eg
    def __init__(self):
        super(ZnO, self).__init__()

class Diode(Mesh1D): #Mesh1D defined in Mesh/Mesh1D
    #var=Diode()
    #var.regions lists:
        #cell at node(x)
        #Elem1D between cell(x) and cell(x+1)
    def __init__(self):
        NN = 500

        xx = np.linspace(-10e-4*Unit.cm, 10e-4*Unit.cm, NN+1)
        Mesh1D.__init__(self, xx, 
                        rgns=[(0, NN/2, 'cu2o'), (NN/2, NN, 'zno')],
                        bnds=[(0,'anode'),(NN,'cathode')])

        self.setRegionMaterial('cu2o', Cu2O())
        self.setRegionMaterial('zno', ZnO())

        def doping(x):
            if x<0:
                return -1e14*pow(Unit.cm,-3)
            elif x>0:
                return 1e18*pow(Unit.cm,-3)
            else:
                return 0.0
            
        self.setFieldByFunc(0, 'C', doping) #region 0 is cu2o
        self.setFieldByFunc(1, 'C', doping) #region 1 is zno

class DiodeEqns(FVMEqns): #FVMEqns init calls 'diode' (device) which calls Mesh1D to set up mesh, material
    def __init__(self, device):
        super(DiodeEqns, self).__init__(device)

        self.eqnCu2O = SemiconductorRegionEqn() #SemiconductorRegionEqn defined in Device/DDEqns. does not actually define any equations yet.  just initializes eqnPerCell(self), cellEqn(self, state, cell), elemEqn(self, state, elem), initGuess(self, state, cell), and damp(self, state, cell, dx). You still have to call the functions in the class.  The functions are where the equations are defined.
        self.setRegionEqn('cu2o', self.eqnCu2O) #function in FVMEqn/FVMEqn/class FVMEqns.  sets eqns in eqnSi to region named 'silicon'.
        
        self.eqnZnO = SemiconductorRegionEqn()
        self.setRegionEqn('zno', self.eqnZnO)

        self.bcAnode = OhmicBoundaryEqn()
        self.bcCathode = OhmicBoundaryEqn()
        self.setBoundaryEqn('anode', self.bcAnode)
        self.setBoundaryEqn('cathode', self.bcCathode)
        self.setInterfaceEqn('cu2o', 'zno', SimpleIFEqn())
        self.setupEqns()

if __name__ == '__main__':
    diode = Diode() #sets up mesh, material, and doping
    diodeEqn = DiodeEqns(diode) #sets up equations to be solved
    diodeEqn.initGuess()
    diodeEqn.bcAnode.setVoltage(0.0)
    diodeEqn.solve()

    diodeEqn.bcAnode.setVoltage(0.0)
    diodeEqn.solve()
    
    ##################################################
    ## For Cu2O
    v0_i = diode.getVarIdx('cu2o', 0) # potential in Substrate
    print '-----------------------'
    print 'Potential in cu2o:'
    print diodeEqn.state.x[v0_i]

    cmc=pow(Unit.cm,-3)
    n0_i = diode.getVarIdx('cu2o', 1) # electron conc in Substrate
    print '-----------------------'
    print 'Elec. conc. in cu2o:'
    print diodeEqn.state.x[n0_i]/cmc

    p0_i = diode.getVarIdx('cu2o', 2) # hole conc in Substrate
    print '-----------------------'
    print 'Hole conc. in cu2o:'
    print diodeEqn.state.x[p0_i]/cmc
    #################################################
    ## For ZnO
    v1_i = diode.getVarIdx('zno', 0) # potential in Substrate
    print '-----------------------'
    print 'Potential in zno:'
    print diodeEqn.state.x[v1_i]

    cmc=pow(Unit.cm,-3)
    n1_i = diode.getVarIdx('zno', 1) # electron conc in Substrate
    print '-----------------------'
    print 'Elec. conc. in zno:'
    print diodeEqn.state.x[n1_i]/cmc

    p1_i = diode.getVarIdx('zno', 2) # hole conc in Substrate
    print '-----------------------'
    print 'Hole conc. in zno:'
    print diodeEqn.state.x[p1_i]/cmc
    
    
    vi= list(v0_i)+list(v1_i[1:])
    ni = list(n0_i)+list(n1_i[1:])
    pi = list(p0_i)+list(p1_i[1:])
    #plot potential
    import pylab
    v = diodeEqn.state.x[vi]
    NN=500
    xx = np.linspace(-1, 1, NN+1)
    xx = np.linspace(-10, 10, NN+1)
    pylab.figure()
    pylab.plot(xx,v)
    pylab.title('Potential')
    print 'Vbi = %f' %(v[0]-v[-1])
    
    #plot electron and hole concentrations
    nn = diodeEqn.state.x[ni]/cmc
    pp = diodeEqn.state.x[pi]/cmc
    pylab.figure()
    pylab.plot(xx,nn)
    pylab.plot(xx,pp)
    pylab.title('Electron and hole concentration')
    
    #plot charge density profile
    rho = []
    pylab.figure()
    i_endRegion0 = 0
    for i,x in enumerate(xx):
        if x<=0:
            doping = diode.regions[0].cells[i].fields['C']/pow(Unit.cm,-3)
            n_e = nn[i]
            p_e = pp[i]
            charge = Unit.e*(p_e-n_e+doping)
            rho.append(charge)
            i_endRegion0 = i
        if x>0:
            doping = diode.regions[1].cells[i-i_endRegion0].fields['C']/pow(Unit.cm,-3)
            n_e = nn[i]
            p_e = pp[i]
            charge = Unit.e*(p_e-n_e+doping)
            rho.append(charge)
    pylab.plot(xx,rho, 'o')
    pylab.title('Charge density profile')
    pylab.show()   

