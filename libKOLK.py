"""
pyKOLK
library for solution pocket modelling in 2D
2024-04-04
Georg Kaufmann
"""

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import sys
import libCHEM,libKOLK


#================================#
def readParameter2D(infile='KOLK_parameter.in',path='work/',control=False):
    """
    pyKOLK
    ! read KOLK parameter file
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
        PAtmmin,PAtmmax,TCavemin,TCavemax,PCavemin,PCavemax,dropCavemin,dropCavemax = libKOLK.readParameter2D()
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
        print('== KOLK ==')
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
def readTimeline2D(infile='KOLK_timeline.in',path='work/',control=False):
    """
    pyKOLK
    ! read KOLK timeline file
    ! input:
    !  (from file infile) 
    ! output:
    !  timeStart,timeEnd   - start/end point for time scale [s]    
    ! use:
    !  timeStart,timeEnd,rawTimeline = libKOLK.readTimeline2D()
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
    pyKOLK
    define initial geometry, shape as linear ramp
    input:
    sidex [m]  : length of model domain
    nx         : steps
    init_height: initial height at center (default: 0.)
    plot       : plot flag (default: False)      
    returns:
    x,y [m]    : x- and y-coordinates
    dx [m]     : discretisation
    shape[]    : array holds x- and y-coordinates for storage
    """
    xmin   = 0.
    xmax   = sidex
    x,dx   = np.linspace(xmin,xmax,nx,retstep=True)
    y      = np.zeros(len(x))
    for i in range(len(x)):
        y = init_height*(1.-x/sidex)**1
    # plot
    if (plot):
        plt.figure(figsize=(10,4))
        plt.xlim([0,sidex])
        plt.ylim([0,1.5*y.max()])
        plt.xlabel('Radius [m]')
        plt.ylabel('Height [m]')
        plt.plot(x,y)
    return x,y,dx


#================================#
def refineGrid2D(x,y):
    """
    pyKOLK
    function used to insert a grid point
    next to the y-xis, when the grid is stretched too much
    input:
    x,y [m]     : x- and y-coordinates
    returns:
    x,y [m]     : new x- and y-coordinates
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
def createClimate2D(timemin,timemax,tempSoilmin,tempSoilmax,pco2Soilmin,pco2Soilmax,
            tempCavemin,tempCavemax,pco2Cavemin,pco2Cavemax,dropCavemin,dropCavemax):
    """
    pyKOLK
    function uses the climate variables (temperature, CO2-pressure, ...) and the current time
    and linearly interpolates the climate values
    input:
      time,timemin,timemax    - time array, min/max values
      tempSoilmin,tempSoilmax - min/max soil temperature
      pco2Soilmin,pco2Soilmax - min/max soil CO2
      tempCavemin,tempCavemax - min/max cave temperature
      pco2Cavemin,pco2Cavemax - min/max cave CO2
      dropCavemin,dropCavemax - min/max cave drop rate
    output:
      tempSoil,tempCave,pco2Soil,pco2Cave,dropCave - interpolated values ...
    """
    # create interpolation functions
    #tempsoil = 0.5 * (1. - np.cos(2*np.pi*time))
    #tempsoil = tempmin + (tempmax-tempmin)*tempsoil
    #tempSoil = scipy.interpolate.interp1d(time,tempSoilmin + (tempSoilmax-tempSoilmin)*0.5 * (1. - np.cos(2*np.pi*time)) )
    
    tempSoil = scipy.interpolate.interp1d([timemin,timemax],[tempSoilmin,tempSoilmax])
    tempCave = scipy.interpolate.interp1d([timemin,timemax],[tempCavemin,tempCavemax])
    pco2Soil = scipy.interpolate.interp1d([timemin,timemax],[pco2Soilmin,pco2Soilmax])
    pco2Cave = scipy.interpolate.interp1d([timemin,timemax],[pco2Cavemin,pco2Cavemax])
    dropCave = scipy.interpolate.interp1d([timemin,timemax],[dropCavemin,dropCavemax])
    return tempSoil,tempCave,pco2Soil,pco2Cave,dropCave


#================================#
def plotKolk(kolkGeom,kolkSave,sidex,iSaved):
    """
    pyKOLK
    plot shapes of solution pockets for specified times
    input:
      kolkGeom - x- und y-coordinates of saved solution pockets shape
      kolkSave - saved times
      sidex    - length of model domain
      iSaved   - number of saved time steps
    output:
      (to file)
    """
    plt.figure(figsize=(6,6))
    plt.title('Solution pocket')
    plt.xlim([-sidex/2,sidex/2])
    plt.ylim([0,1])
    plt.xlabel('Radius [m]')
    plt.ylabel('Height [m]')
    plt.fill_between(np.r_[-kolkGeom[:,0,iSaved][::-1],kolkGeom[:,0,iSaved]],np.r_[kolkGeom[:,1,iSaved][::-1],kolkGeom[:,1,iSaved]],1,color='gray',alpha=0.5)
    for i in range(iSaved+1):
        plt.plot(np.r_[-kolkGeom[:,0,i][::-1],kolkGeom[:,0,i]],np.r_[kolkGeom[:,1,i][::-1],kolkGeom[:,1,i]],label=str(round(kolkSave[i],0))+' a')
    #plt.plot(x,y)
    plt.legend()
    plt.grid()
    return


#================================#
def runKolk_flow(infile1='KOLK_parameter.in',infile2='KOLK_timeline.in',path='work/',mix=1.0,climate='simple',plot=False):
    """
    pyKOLK
    function creates solution pocket morphologies in 2D,
    using the flow version for shape
    and time as master variable (simple as argument)
    or temperature as master variable (temp as argument)
    adopted from fortran90 code ...
    """
    # fixed parameter values
    dropvolume  = 1e-7    # drop volume [m^3]
    Vdrop       = 0.1e-6  # drop volume [m3]
    rhoWater    = 1000.   # density water [kg/m3]
    g           = 9.81    # grav acceleration [m/s2]
    etaWater    = 1.14e-3 # viscosity water [Pas]
    mCaCO3      = 0.1001  # molar mass calcite [kg/mol]
    rhoCaCO3    = 2700.   # density calcite [kg/m3]
    year2sec    = 365.25*24*60*60
    # read input data
    sidex,nx,init_height,timeStep,timeWrite,TSoilmin,TSoilmax,PSoilmin,PSoilmax, \
                PAtmmin,PAtmmax,TCavemin,TCavemax,PCavemin,PCavemax,dropCavemin,dropCavemax = \
                libKOLK.readParameter2D(infile=infile1,path=path,control=True)
    timeStart,timeEnd,rawTimeline = libKOLK.readTimeline2D(infile=infile2,path=path,control=True)
    # create grid
    x,y,dx = libKOLK.createGrid2D(sidex,nx,init_height,plot=True)
    # create interpolation objects for climate
    tempSoil,tempCave,pco2Soil,pco2Cave,dropCave = libKOLK.createClimate2D(timeStart,timeEnd,TSoilmin,TSoilmax,PSoilmin,PSoilmax,
                                                           TCavemin,TCavemax,PCavemin,PCavemax,dropCavemin,dropCavemax)
    # define stack for geometry, fill with initial shape
    nstack          = 50
    kolkGeom        = np.zeros(nx*2*nstack).reshape(nx,2,nstack)
    kolkGeom[:,0,0] = x
    kolkGeom[:,1,0] = y
    kolkSave        = np.zeros(nstack)
    kolkSave[0]     = timeStart
    # initialize fields along surface
    growth = np.zeros(len(x))
    angle  = np.zeros(len(x))
    width  = np.zeros(len(x))
    flux   = np.zeros(len(x))
    film   = np.zeros(len(x))
    vel    = np.zeros(len(x))
    c      = np.zeros(len(x))
    t      = np.zeros(len(x))
    #---------------------
    # time loop
    #---------------------
    iSaved   = 0
    itime    = 0
    time     = timeStart + timeStep
    timeSave = timeStart + timeWrite
    print('Start time loop ...')
    while (time <= timeEnd):
        # interpolate climate parameter
        if (climate == 'simple'):
            Tsoil = np.interp(time,[timeStart,timeEnd],[TSoilmin,TSoilmax])
            Tcave = np.interp(time,[timeStart,timeEnd],[TCavemin,TCavemax])
            Psoil = np.interp(time,[timeStart,timeEnd],[PSoilmin,PSoilmax])
            Pcave = np.interp(time,[timeStart,timeEnd],[PCavemin,PCavemax])
            Dcave = np.interp(time,[timeStart,timeEnd],[dropCavemin,dropCavemax])
        elif (climate == 'season'):
            Tsoil = TSoilmin + (TSoilmax-TSoilmin)*0.5 * (1. - np.cos(2*np.pi*time))
            Tcave = np.interp(time,[timeStart,timeEnd],[TCavemin,TCavemax])
            Psoil = np.interp(time,[timeStart,timeEnd],[PSoilmin,PSoilmax])
            Pcave = np.interp(time,[timeStart,timeEnd],[PCavemin,PCavemax])
            Dcave = np.interp(time,[timeStart,timeEnd],[dropCavemin,dropCavemax])
        elif (climate == 'paleo'):
            Tsoil = np.interp(time,rawTimeline[:,0],rawTimeline[:,1])
            Tcave = TCavemin+ (TCavemax-TCavemin) * (TSoil-TSoilmin)/(TSoilmax-TSoilmin)
            Psoil = PSoilmin+ (PSoilmax-PSoilmin) * (TSoil-TSoilmin)/(TSoilmax-TSoilmin)
            Pcave = PCavemin+ (PCavemax-PCavemin) * (TSoil-TSoilmin)/(TSoilmax-TSoilmin)
            Dcave = DCavemin+ (DCavemax-DCavemin) * (TSoil-TSoilmin)/(TSoilmax-TSoilmin)
        Q = Vdrop / Dcave
        # drop chemistry
        CEQopen   = libCHEM.CEQ_limestone_open(Tsoil,Psoil/1e6)
        CEQclosed = libCHEM.CEQ_limestone_closed(Tsoil,Psoil/1e6)
        Cin       = mix*CEQopen + (1-mix)*CEQclosed
        CEQcave   = libCHEM.CEQ_limestone_open(Tcave,Pcave/1e6)
        # calcium concentration, flux rate, and growth rate
        c[0]      = Cin
        flux[0]   = libCHEM.FCaCO3(Cin,CEQcave)
        if (flux[0]<0): 
            flux[0]=0.
        growth[0] = mCaCO3 / rhoCaCO3 * flux[0] * timeStep * year2sec
        # refine grid    
        if (x[1] > 1.5*dx):
            x,y = libKOLK.refineGrid2D(x,y)
        # loop along surface
        angle[0]  = 0.
        for i in range(1,len(x)):
            dl       = np.sqrt( (x[i-1]-x[i])**2 + (y[i-1]-y[i])**2)
            width[i] = 2.*np.pi*x[i]
            angle[i] = np.arcsin((y[i-1]-y[i])/dl)
            if (angle[i] > 0.):
                film[i]  = np.cbrt(3*etaWater/(rhoWater*g) * Q / (width[i]*np.sin(angle[i])))
                vel[i]   = np.cbrt(rhoWater*g/(3*etaWater) * Q**2*np.sin(angle[i])/(width[i]**2))
                t[i]     = t[i-1] + dl/vel[i]
                c[i]     = c[i-1] + flux[i-1]/vel[i]/film[i]*dl
            else:
                t[i]     = t[i-1]
                c[i]     = c[i-1]
            flux[i]   = libCHEM.FCaCO3(c[i],CEQcave)
            if (flux[i]<0): 
                flux[i]=0.
            growth[i] = mCaCO3 / rhoCaCO3 * flux[i] * timeStep * year2sec
        # update shape
        x[0]      = x[0]
        y[0]      = y[0] + growth[0]
        for i in range(1,len(x)):
            x[i] = x[i] + growth[i]*np.sin(angle[i])
            y[i] = y[i] + growth[i]*np.cos(angle[i])
        vel[0]  = vel[1]
        film[0] = film[1]
        # save stalagmite shape for defined time step
        if (time >= timeSave):
            iSaved = iSaved + 1
            if (iSaved > kolkGeom.shape[2]-1):
                print('iSaved too large')
                sys.exit()
            print('%30s %10i %10.2f %10.2f' % ('Time step, time [a], TSoil [C]:',iSaved,time,Tsoil))
            kolkGeom[:,0,iSaved] = x
            kolkGeom[:,1,iSaved] = y
            kolkSave[iSaved]     = time
            timeSave            += timeWrite
        # update time
        time += timeStep
        itime += 1
    print('End time loop ...')
    # plot solution pocket
    if (plot):
        libKOLK.plotKolk(kolkGeom,kolkSave,sidex,iSaved)
    return kolkGeom,kolkSave,iSaved,sidex


#================================#
#================================#
#================================#
#================================#
