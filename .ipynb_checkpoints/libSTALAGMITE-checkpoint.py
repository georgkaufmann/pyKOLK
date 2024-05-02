"""
STALAGMITE
library for stalagmite modelling in 2D
2024-04-04
Georg Kaufmann
"""

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

#================================#
def readParameter2D(infile='STALAGMITE_parameter.in',path='work/',control=False):
    """
    ! read STALAGMITE parameter file
    ! input:
    !  (from file infile)
    ! output:
    !  xmin,xmax,nx         - min/max for x coordinate [m], discretisation
    !  whichtime            - flag for time units used
    !  time_start,time_end  - start/end point for time scale [s]
    !  time_step            - time step [s]
    !  time_scale           - scaling coefficient for user time scale
    ! use:
    !  sidex,nx,init_height,timeStep,timeWrite,TSoilmin,TSoilmax,PSoilmin,PSoilmax,
        PAtmmin,PAtmmax,TCavemin,TCavemax,PCavemin,PCavemax,dropCavemin,dropCavemax = libSTALAGMITE.readParameter2D()
    ! note:
    !  file structure given!
    !  uses readline(),variables come in as string,
    !  must be separated and converted ...
    """
    # read in data from file
    f = open(path+infile,'r')
    # first set of comment lines
    line = f.readline();line = f.readline();line = f.readline()
    line = f.readline()
    sidex,nx = float(line.split()[0]),int(line.split()[1])
    line = f.readline()
    init_height = float(line.split()[0])
    # second set of  comment lines
    line = f.readline();line = f.readline();line = f.readline()
    line = f.readline()
    timeStep = float(line.split()[0])
    line = f.readline()
    timeWrite = float(line.split()[0])
    # third set of  comment lines
    line = f.readline();line = f.readline();line = f.readline()
    line = f.readline()
    TSoilmin,TSoilmax = float(line.split()[0]),float(line.split()[1])
    line = f.readline()
    PSoilmin,PSoilmax = float(line.split()[0]),float(line.split()[1])
    line = f.readline()
    PAtmmin,PAtmmax = float(line.split()[0]),float(line.split()[1])
    line = f.readline()
    TCavemin,TCavemax = float(line.split()[0]),float(line.split()[1])
    line = f.readline()
    PCavemin,PCavemax  = float(line.split()[0]),float(line.split()[1])
    line = f.readline()
    dropCavemin,dropCavemax  = float(line.split()[0]),float(line.split()[1])
    # control output to screen
    if (control):
        print('== STALAGMITE ==')
        print('%30s %20s' % ('path:',path))
        print('%30s %10.2f %10i' % ('sidex [m],nx:',sidex,nx))
        print('%30s %10.2f' % ('init_height [m]:',init_height))
        print('%30s %10.2f %12.2f' % ('timestep,timewrite [a]:',timeStep,timeWrite))
        print('%30s %10.2f %12.2f' % ('TSoilmin,TSoilmax [C]:',TSoilmin,TSoilmax))
        print('%30s %10.2f %12.2f' % ('PSoilmin,PSoilmax [ppm]:',PSoilmin,PSoilmax))
        print('%30s %10.2f %12.2f' % ('PAtmmin,PAtmmax [ppm]:',PAtmmin,PAtmmax))
        print('%30s %10.2f %12.2f' % ('TCavemin,TCavemax [C]:',TCavemin,TCavemax))
        print('%30s %10.2f %12.2f' % ('PCavemin,PCavemax [ppm]:',PCavemin,PCavemax))
        print('%30s %10.2f %12.2f' % ('dropCavemin,dropCavemax [s]:',dropCavemin,dropCavemax))
    return [sidex,nx,init_height,timeStep,timeWrite,TSoilmin,TSoilmax,PSoilmin,PSoilmax,
        PAtmmin,PAtmmax,TCavemin,TCavemax,PCavemin,PCavemax,dropCavemin,dropCavemax]


#================================#
def readTimeline2D(infile='STALAGMITE_timeline.in',path='work/',control=False):
    """
    ! read STALAGMITE timeline file
    ! input:
    !  (from file infile) 
    ! output:
    !  timeStart,timeEnd   - start/end point for time scale [s]    
    ! use:
    !  timeStart,timeEnd,rawTimeline = libSTALAGMITE.readTimeline2D()
    ! note:
    !  file structure given!
    !  uses loadtxt(), with 2 lines skipped!
    """
    rawTimeline = np.loadtxt(path+infile,skiprows=2,dtype='float')
    timeStart = rawTimeline[0,0]
    timeEnd   = rawTimeline[-1,0]
    if (control):
        print('%30s %10.2f %12.2f' % ('timeStart,timeEnd [a]:',timeStart,timeEnd))
    return timeStart,timeEnd,rawTimeline


#================================#
def createGrid2D(sidex,nx,init_height=0,plot=False):
    """
    ! define initial geometry, shape as linear ramp
    ! input:
    !  sidex [m]  : length of model domain
    !  nx         : steps
    !  init_height: initial height at center (default: 0.)
    !  plot       : plot flag (default: False)      
    ! output:
    !  x,y [m]    : x- and y-coordinates
    !  dx [m]     : discretisation
    ! use:
    !  x,y,dx = libSTALAGMITE.createGrid2D(sidex,nx,init_height)
    """
    xmin   = 0.
    xmax   = sidex
    x,dx   = np.linspace(xmin,xmax,nx,retstep=True)
    y      = np.zeros(len(x))
    for i in range(len(x)):
        y = init_height*(1.-x/sidex)**1
    # plot
    if (plot):
        plt.figure(figsize=(10,3))
        plt.xlim([0,sidex])
        plt.ylim([0,1.5*y.max()])
        plt.xlabel('Radius [m]')
        plt.ylabel('Height [m]')
        plt.plot(x,y)
    return x,y,dx


#================================#
def refineGrid2D(x,y):
    """
    ! function used to insert a grid point
    ! next to the y-xis, when the grid is stretched too much
    ! input:
    !  x,y [m]     : x- and y-coordinates
    ! output:
    !  x,y [m]     : new x- and y-coordinates
    ! use:
    !  x,y = libSTALAGMITED.refineGrid2D(x,y)
    """
    xnew = 0.5*(x[0]+x[1])
    ynew = 0.5*(y[0]+y[1])
    for i in range(len(x)-1,1,-1):
        x[i] = x[i-1]
        y[i] = y[i-1]
    x[1] = xnew
    y[1] = ynew
    return x,y


#================================#
def ALPHA(TC=10.,film=0.0001):
    """
    !Precipitation rate coefficient for flux rate
    ! input: 
    !  TC    - temperature [C]
    !  film  - film thickness [m]
    ! output:
    !  alpha - rate coefficient [m/s]
    ! from:
    !  Romanov et al. (2009)
    ! use:
    !  ALPHA = libSTALAGMITE.ALPHA(TC,film)
    """
    if (film >= 1.0e-4):
        ALPHA = 1.e-7*(0.51549e0+0.04015e0*TC+0.00418e0*TC*TC)
    elif (film >= 7.50e-5):
        ALPHA = 1.e-7*(0.4615e0+0.03192e0*TC+0.00408e0*TC*TC)
    elif (film >= 5.00e-5):
        ALPHA = 1.e-7*(0.43182e0+0.02103e0*TC+0.00381e0*TC*TC)
    else:
        ALPHA = 0.
        print ('ALPHA: film thickness film too small')
    return ALPHA


#================================#
def FCaCO3(c,ceq,T=10.,d=0.001):
    """
    ! Flux-rate law for limestone precipitation**
    ! from:
    !  Buhmann & Dreybrodt (1985)
    """
    if (c >= ceq):
        FCaCO3 = ALPHA(T,d)*(c-ceq)
    else:
        FCaCO3 = 0.
    return FCaCO3


#================================#
def growthRate2D(Cin,CEQcave,Dcave,Tsoil,film=0.01e-2,mCaCO3=0.1001 ,rhoCaCO3=2700.):
    """
    ! function calculates the growth rates for stalagmites
    ! input:
    !  Cin         - input calcium concentration [mol/m3]
    !  CEQcave     - calcium equilibrium concentration in cave [mol/m3]
    !  Dcave       - drop interval [s]
    !  Tsoil       - temperature [C]
    !  film        - film thickness [m]
    !  mCaCO3      - atomic mass calcite [kg/mol]
    !  rhoCaCO3    - density calcite [kg/m3]
    !  ALPHA       - rate constant [m/s]
    ! output:
    !  W0          - growth rate [m/s]
    """
    import libSTALAGMITE
    W0 = 0.
    if (Tsoil >= 0.):
        if ((Cin-CEQcave) > 0):
            W0 = mCaCO3/rhoCaCO3 * (Cin-CEQcave) * film / Dcave \
               * (1-np.exp(-libSTALAGMITE.ALPHA(Tsoil)*Dcave/film))
    return W0

#================================#
def equiRadius2D(Cin,CEQcave,Dcave,Tsoil,film=0.01e-2,Vdrop=0.1e-6):
    """
    !function calculates the equilibrium radius for stalagmites
    !input:    
    !  Cin         - input calcium concentration [mol/m3]
    !  CEQcave     - calcium equilibrium concentration in cave [mol/m3]
    !  Dcave       - drop interval [s]
    !  Tsoil       - temperature [C]
    !  Vdrop       - drop volume [m^3]
    !  film        - film thickness [m]
    !  ALPHA       - rate constant [m/s]
    ! output:
    !  R0          - equilibrium radius [m]
    """
    import libSTALAGMITE
    if ((Cin-CEQcave) < 0):
        R0 = 0.
        return R0
    if (Tsoil < 0):
        R0 = np.sqrt(Vdrop / (np.pi*film))
        return R0
    R0 = np.sqrt(Vdrop / (np.pi*film * (1.-np.exp(-libSTALAGMITE.ALPHA(Tsoil)*Dcave/film))))
    return R0


#================================#
def plotStalagmite2D(stal,iSaved,tSaved,sidex,title='Stalagmite shape'):
    plt.figure(figsize=(4,6))
    plt.title(title)
    plt.xlim([-sidex/2,sidex/2])
    plt.ylim([0,2.5])
    plt.xlabel('Radius [m]')
    plt.ylabel('Height [m]')
    plt.fill_between(np.r_[-stal[:,0,iSaved][::-1],stal[:,0,iSaved]],np.r_[stal[:,1,iSaved][::-1],stal[:,1,iSaved]],0.,color='gray',alpha=0.5)
    for i in range(iSaved+1):
        plt.plot(np.r_[-stal[:,0,i][::-1],stal[:,0,i]],np.r_[stal[:,1,i][::-1],stal[:,1,i]],label=str(tSaved[i]/1000)+' ka')
    plt.legend(bbox_to_anchor=(1.4,1.0))
    plt.grid()
    return


#================================#
def plotStalagmiteAges2D(stal,iSaved,tSaved,sidex,title='Stalagmite (flow)'): 
    import matplotlib
    colors = [
    ( 0 , 0 , 1 ),
    ( 0 , 0.196078 , 1 ),
    ( 0 , 0.392157 , 1 ),
    ( 0 , 0.588235 , 1 ),
    ( 0 , 0.784314 , 0.784314 ),
    ( 0 , 1 , 0 ),
    ( 0 , 0.862745 , 0 ),
    ( 0 , 0.784314 , 0 ),
    ( 0 , 0.588235 , 0 ),
    ( 0 , 0.392157 , 0 ),
    ( 1 , 0.784314 , 0 ),
    ( 1 , 0.588235 , 0 ),
    ( 1 , 0.392157 , 0 ),
    ( 1 , 0.196078 , 0 ),
    ( 1 , 0 , 0 )]
    cmap = matplotlib.colors.ListedColormap(colors[::-1])
    ncolors = cmap.N
    #cmap = plt.get_cmap('tab20', ncolors)
    bounds = np.linspace(0,10, ncolors)
    norm = mpl.colors.BoundaryNorm(bounds,ncolors)

    plt.figure(figsize=(4,4))
    plt.title(title)
    plt.xlim([-sidex/4,sidex/4])
    plt.ylim([0.0,2.5])
    plt.xlabel('Radius [m]')
    plt.ylabel('Height [m]')
    for i in range(iSaved,0,-1):
        t = int(-tSaved[i]/1000)
        x  = np.r_[-stal[:,0,i][::-1],stal[:,0,i]]
        y1 = np.r_[stal[:,1,i][::-1],stal[:,1,i]]
        y2 = np.r_[stal[:,1,i-1][::-1],stal[:,1,i-1]]
        cs1=plt.fill_between(x,y1,0,cmap=cmap,color=cmap.colors[t],norm=norm,alpha=1)
        plt.plot(x,y1,lw=1,color='black',alpha=0.5)
    plt.colorbar(cs1,shrink=0.8,format='%4.1f',label='Age [ka BP]')
    plt.grid()
    return


#================================#