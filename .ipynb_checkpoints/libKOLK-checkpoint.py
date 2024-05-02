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
import libCHEM

#================================#
def createGrid(sidex,nx,init_height=0,plot=False):
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
    # define stack for geometry
    nstack = 50
    shape = np.zeros(nx*2*nstack).reshape(nx,2,nstack)
    shape[:,0,0] = x
    shape[:,1,0] = y
    # plot
    if (plot):
        plt.figure(figsize=(10,4))
        plt.xlim([0,sidex])
        plt.ylim([0,10*shape[:,1,0].max()])
        plt.xlabel('Radius [m]')
        plt.ylabel('Height [m]')
        plt.fill_between(shape[:,0,0],shape[:,1,0],0.,label='t=0')
        plt.legend()
    return x,y,dx,shape


#================================#
def refineGrid(x,y):
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
def createTime(timemin,timemax,timestep):
    """
    pyKOLK
    function defines time array from min/max values
    input:
    timemin,timemax [s]: min/max time
    timestep [s]       : time step
    returns:
    time [s]           : 1D array of time steps
    """
    nstep = int((timemax-timemin)/timestep)+1
    time = np.linspace(timemin,timemax,nstep,retstep=False)
    return time


#================================#
def createClimate(time,timemin,timemax,tempSoilmin,tempSoilmax,pco2Soilmin,pco2Soilmax,
            tempCavemin,tempCavemax,pco2Cavemin,pco2Cavemax,dropCavemin,dropCavemax,plot=False):
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
    tempSoil = scipy.interpolate.interp1d(time,tempSoilmin + (tempSoilmax-tempSoilmin)*0.5 * (1. - np.cos(2*np.pi*time)) )
    
    #tempSoil = scipy.interpolate.interp1d([timemin,timemax],[tempSoilmin,tempSoilmax])
    tempCave = scipy.interpolate.interp1d([timemin,timemax],[tempCavemin,tempCavemax])
    pco2Soil = scipy.interpolate.interp1d([timemin,timemax],[pco2Soilmin,pco2Soilmax])
    pco2Cave = scipy.interpolate.interp1d([timemin,timemax],[pco2Cavemin,pco2Cavemax])
    dropCave = scipy.interpolate.interp1d([timemin,timemax],[dropCavemin,dropCavemax])

# control plot for climate parameter
    if (plot):
        plt.figure(figsize=(10,6))
        plt.xlabel('Time [a]')
        plt.xlim([timemin,timemax])
        plt.yscale('log')
        plt.plot(time,tempCave(time),lw=4,label='tempcave [${}^{\circ}$C]')
        plt.plot(time,tempSoil(time),lw=1,label='tempsoil [${}^{\circ}$C]')
        plt.plot(time,pco2Soil(time),lw=4,label='pCO2soil [ppm]')
        plt.plot(time,pco2Cave(time),lw=2,label='pCO2cave [ppm]')
        plt.plot(time,dropCave(time),lw=2,label='dropcave [s]')
        plt.legend()
        plt.grid()
    return tempSoil,tempCave,pco2Soil,pco2Cave,dropCave


#================================#
def createKolk_flow(x,y,dx,shape,time,timestep,timewrite,tempSoil,pco2Soil,tempCave,pco2Cave,dropCave,mix=1.,plot=False):
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
    # initialize fields along surface
    growth = np.zeros(len(x))
    angle  = np.zeros(len(x))
    width  = np.zeros(len(x))
    flux   = np.zeros(len(x))
    film   = np.zeros(len(x))
    vel    = np.zeros(len(x))
    c      = np.zeros(len(x))
    t      = np.zeros(len(x))
    # loop over time
    Tsave  = timewrite
    iSave  = 0
    for itime in range(len(time)):
        # set current time
        Time = time[itime]
        # interpolate climate parameter
        Tsoil = tempSoil(Time)
        Tcave = tempCave(Time)
        Psoil = pco2Soil(Time)
        Pcave = pco2Cave(Time)
        Dcave = dropCave(Time)
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
        growth[0] = mCaCO3 / rhoCaCO3 * flux[0] * timestep * year2sec
        # refine grid    
        if (x[1] > 1.5*dx):
            x,y = refineGrid(x,y)
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
            growth[i] = mCaCO3 / rhoCaCO3 * flux[i] * timestep * year2sec
        # update shape
        x[0]      = x[0]
        y[0]      = y[0] + growth[0]
        for i in range(1,len(x)):
            x[i] = x[i] + growth[i]*np.sin(angle[i])
            y[i] = y[i] + growth[i]*np.cos(angle[i])
        vel[0]  = vel[1]
        film[0] = film[1]
        # save stalagmite shape for defined time step
        if (Time == Tsave):
            iSave = iSave + 1
            if (iSave > shape.shape[2]-1):
                print('iSave too large')
                sys.exit()
            print(iSave,Time,shape.shape[2])
            shape[:,0,iSave] = x
            shape[:,1,iSave] = y
            Tsave = Tsave + timewrite
            if (plot):
                length = np.zeros(len(x))
                for i in range(1,len(x)):
                    length[i] = length[i-1] + np.sqrt( (x[i-1]-x[i])**2 + (y[i-1]-y[i])**2)

                fig, ax1 = plt.subplots(1,2,figsize=(15,6))
                #ax1.figure(figsize=[15,10])
                ax1[0].set_title('Solution pocket (t='+str(Time)+' a')
                ax1[0].set_xlabel('Distance [m]')
                ax1[0].set_ylabel('Height [m]')
                ax1[0].set_xlim([0,sidex])
                ax1[0].set_ylim([0,1.1*np.max(y)])
                #ax1[0].text(0.5*length,1.00*CEQsoil,str(Time[-1])+' yrs',horizontalalignment='center')
                ax1[0].plot(x,y,linewidth='4',color='grey',label='surface [m]')
                ax1[0].plot(shape[:,0,0],shape[:,1,0],linewidth='2',linestyle='--',color='black')
                ax1[0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.1),shadow=True, ncol=2)

                ax1[1].set_xlabel('Length along surface [m]')
                ax1[1].plot(length,film*1.e4,label='$\delta$ [$\mu$m]')
                ax1[1].plot(length,vel*1.e2,label='$v$ [cm/s]')
                ax1[1].plot(length,t/3600,label='$t$ [h]')
                ax1[1].plot(length,c,color='red',marker='o',markersize='3',label='$c$ [mol/m$^3$]')
                ax1[1].plot(length,np.repeat(CEQcave,len(x)),color='red',linestyle='--')
                ax1[1].legend(loc='upper left', bbox_to_anchor=(0.0, -0.1),shadow=True, ncol=2)
    return shape,iSave,CEQcave


#================================#
def plotKolk(shape,sidex,iSave,timewrite):
    """
    pyKOLK
    plot shapes of solution pockets for specified times
    input:
      shape     - x- und y-coordinates of saved solution pockets shape
      sidex     - length of model domain
      iSave     - time step
      timewrite - write tinterval
    output:
      (to file)
    """
    plt.figure(figsize=(6,6))
    plt.title('Solution pocket')
    plt.xlim([-sidex/2,sidex/2])
    plt.ylim([0,1])
    plt.xlabel('Radius [m]')
    plt.ylabel('Height [m]')
    plt.fill_between(np.r_[-shape[:,0,iSave][::-1],shape[:,0,iSave]],np.r_[shape[:,1,iSave][::-1],shape[:,1,iSave]],1,color='gray',alpha=0.5)
    for i in range(iSave+1):
        plt.plot(np.r_[-shape[:,0,i][::-1],shape[:,0,i]],np.r_[shape[:,1,i][::-1],shape[:,1,i]],label=str(i*timewrite)+' a')
    #plt.plot(x,y)
    plt.legend()
    plt.grid()
    return


#================================#
