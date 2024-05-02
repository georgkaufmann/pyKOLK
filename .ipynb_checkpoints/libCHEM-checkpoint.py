"""
SOCKS3D
library for water chemistry
2023-03-04
Georg Kaufmann
"""

import numpy as np
import libCHEM
import sys
# general constants
R = 83.1451 # gas constant [ml/bar/K/mol]

def KW(TC,S=0.,D=0.):
    """
    -----------------------------------------------------------------------
    KW - equilibrium constant dissociation of water
    KW: H2O <-> H+ + OH- 
    from:
    Millero, Geochemica et Cosmochemica Acta 43:1651-1661, 1979
    refit data of Harned and Owen, The Physical Chemistry of
    Electrolyte Solutions, 1958
    this is on the SWS pH scale in (mol/kg-SW)^2
    input: 
    TC [C]:         temperature
    output:
    KW [mol^2/l^2]: H2O <-> H+ + OH- 
    -----------------------------------------------------------------------
    """
    TK = 273.16 + TC
    KW = 148.9802 - 13847.26/TK - 23.6521*np.log(TK)
    if (S > 0.):
        KW += (-79.2447 + 3298.72/TK + 12.0408*np.log(TK))*np.sqrt(S) - 0.019813*S
    KW = np.exp(KW)
    return KW


def KH(TC,S=0.):
    """
    -----------------------------------------------------------------------
    KH - Henry constant (solubility of CO2 in water)
    KH: CO2gas <-> CO2water
    from: 
    Weiss, R. F., Marine Chemistry 2:203-215, 1974.
    input:
    TC [C]         - temperature
    output:
    KH [mol/l/atm] - CO2gas <-> CO2water
    -----------------------------------------------------------------------
    """
    TK = 273.16 + TC
    KH =  -60.2409 + (93.4517 / (TK/100)) + (23.3585 * np.log((TK/100)))
    if (S > 0.):
          KH += S *(0.023517 - 0.023656 * (TK/100) + 0.0047036 * (TK/100)**2)
    KH = np.exp(KH)
    return KH


def K1K2old(TC):
    """
    -----------------------------------------------------------------------
    K_1 and K_2 equilibrium constants dissociation of CO2 in water
    K1: CO2 + H2O <-> H+ + HCO3- 
    K2: HCO3-     <-> H+ + CO3--
    
    from:
    Millero et al (2006):
    Dissociation constants of carbonic acid in seawater as a function of 
    salinity and temperature, Marine Chemistry 100(1-2):80-94.
    input:
    TC [C]         - temperature
    S [permil]     - salinity
    output:
    K1 [mol/l]     - K1 = (H+)(HCO3-) / (CO2)
    K2 [mol/l]     - K2 = (H+)(CO3--) / (HCO3-)
    -----------------------------------------------------------------------
    """
    TK  = 273.16 + TC
    pK1 = -126.34048 + 6320.813/TK + 19.568224*np.log(TK)
    K1  = 10**(-pK1)
    pK2 = -90.18333 + 5143.692/TK + 14.613358*np.log(TK)
    K2  = 10**(-pK2)
    return K1,K2


def K1K2(TC,S=0.,D=0.):
    """
    -----------------------------------------------------------------------
    K_1 and K_2 equilibrium constants dissociation of CO2 in water
    K1: CO2 + H2O <-> H+ + HCO3- 
    K2: HCO3-     <-> H+ + CO3--
    
    from:
    ! PURE WATER
    ! Millero, F. J., Geochemica et Cosmochemica Acta 43:1651-1661, 1979:
    ! K1 from refit data from Harned and Davis,
    ! J American Chemical Society, 65:2030-2037, 1943.
    ! K2 from refit data from Harned and Scholes,
    ! J American Chemical Society, 43:1706-1709, 1941.
    ! This is only to be used for Sal=0 water (note the absence of S in the below formulations)
    ! These are the thermodynamic Constants:
    ! this is on the SWS pH scale in mol/kg-SW
    ! SALT WATER
    ! GEOSECS and Peng et al use K1, K2 from Mehrbach et al,
    ! Limnology and Oceanography, 18(6):897-907, 1973.
    ! I.e., these are the original Mehrbach dissociation constants.
    ! The 2s precision in pK1 is .005, or 1.2% in K1.
    ! The 2s precision in pK2 is .008, or 2% in K2.
    """
    TK  = 273.16 + TC
    if (S==0.):
        K1 = 290.9097 - 14554.21/TK - 45.0575*np.log(TK)
        K1 = np.exp(K1)
        K2 = 207.6548 - 11843.79/TK - 33.6485*np.log(TK)
        K2 = np.exp(K2)
    elif (S>0.):
        K1 = - 13.7201 + 0.031334*TK + 3235.76/TK + 1.3e-5*S*TK - 0.1032*S**(0.5)
        K1 = 10**(-K1)      # this is on the NBS scale
        K2 = 5371.9645 + 1.671221*TK + 0.22913*S + 18.3802*np.log10(S) \
        - 128375.28/TK - 2194.3055*np.log10(TK) - 8.0944e-4*S*TK \
        - 5617.11*np.log10(S)/TK + 2.136*S/TK
        K2 = 10**(-K2)      # this is on the NBS scale
    return K1,K2


def K5(TC):
    """
    -----------------------------------------------------------------------
    K_5 - equilibrium constant dissociation of carbonic acid
    K5: H2CO3 <-> H+ + HCO3-
    -----------------------------------------------------------------------
    input:
    TC [C]         - temperature
    output:
    K5 [mol/l]:    - H2CO3 <-> H+ + HCO3-
    """
    TK  = 273.16 + TC
    K5  = 1.707e-4 *TK/TK
    return K5


def KC(TC,S=0.,D=0.):
    """
    -----------------------------------------------------------------------
    calcite solubility
    from:
    Mucci, Alphonso, Amer. J. of Science 283:781-799, 1983.
      sd fit = .01 (for Sal part, not part independent of Sal)
      this is in (mol/kg-SW)^2
    -----------------------------------------------------------------------
    """
    TK  = 273.16 + TC
    KC = -171.9065 - 0.077993*TK + 2839.319/TK
    KC = KC + 71.595*np.log(TK)/np.log(10.0)
    if (S > 0.):
        KC = KC + (-0.77712 + 0.0028426*TK + 178.34/TK)*np.sqrt(S)
        KC = KC - 0.07711*S + 0.0041249*np.sqrt(S)*S
    KC = 10.0**KC
    return KC


def KA(TC,S=0.,D=0.):
    """
    -----------------------------------------------------------------------
    aragonite solubility
    from:
    Mucci, Alphonso, Amer. J. of Science 283:781-799, 1983.
      sd fit = .009 (for Sal part, not part independent of Sal)
      this is in (mol/kg-SW)^2
    -----------------------------------------------------------------------
    """
    TK  = 273.16 + TC
    KA = -171.945 - 0.077993*TK + 2903.293/TK
    KA = KA + 71.595*np.log(TK)/np.log(10.)
    KA = KA + (-0.068393 + 0.0017276*TK + 88.135/TK)*np.sqrt(S)
    KA = KA - 0.10018*S + 0.0059415*np.sqrt(S)*S
    KA    = 10.**(KA)
    return KA


def ion_debyehueckel(IS,TC):
    '''
    !-----------------------------------------------------------------------
    ! function calculates activity coefficients for different ions
    ! following the Debye-Hueckel model
    ! NOTE for Ca2+, Mg2+, Na+, Cl- the bdot extended model is used
    !      bdot value from phreeqc
    ! input:
    !  TC             - temperature [C]
    !  IS             - ionic strength [mol / l]
    ! output
    !  ion_ca2p       - activity [-]
    !  ion_hco3m      - "
    !  ion_mg2p       - "
    !  ion_hp         - "
    !  ion_co32m      - "
    !  ion_ohm        - "
    !  version using density and dielectric constant
    !  written by Georg Kaufmann 03/01/2008
    !-----------------------------------------------------------------------
    '''
    TK          = 273.160 + TC
    rho         = 1.0
    dielectric  = 87.720 - 0.397020 * TC + 0.000817840 * TC**2
    aa          = 1.82483e6 * np.sqrt(rho / (dielectric*TK)**3.)
    bb          = 50.29120 * np.sqrt(rho / (dielectric*TK))
    ion_hp      = 10.0**(-aa*1.0*np.sqrt(IS)/(1.0+bb*9.00*np.sqrt(IS)))
    ion_ca2p    = 10.0**(-aa*4.0*np.sqrt(IS)/(1.0+bb*5.00*np.sqrt(IS)) + 0.1650*IS)
    ion_mg2p    = 10.0**(-aa*4.0*np.sqrt(IS)/(1.0+bb*5.50*np.sqrt(IS)) + 0.2000*IS)
    ion_ohm     = 10.0**(-aa*1.0*np.sqrt(IS)/(1.0+bb*3.50*np.sqrt(IS)))
    ion_hco3m   = 10.0**(-aa*1.0*np.sqrt(IS)/(1.0+bb*5.40*np.sqrt(IS)))
    ion_co32m   = 10.0**(-aa*4.0*np.sqrt(IS)/(1.0+bb*5.40*np.sqrt(IS)))
    ion_so42m   = 10.0**(-aa*4.0*np.sqrt(IS)/(1.0+bb*5.00*np.sqrt(IS)))
    ion_nap     = 10.0**(-aa*1.0*np.sqrt(IS)/(1.0+bb*4.00*np.sqrt(IS)) + 0.040*IS)
    ion_clm     = 10.0**(-aa*1.0*np.sqrt(IS)/(1.0+bb*3.00*np.sqrt(IS)) + 0.040*IS)
    return ion_hp,ion_ca2p,ion_mg2p,ion_ohm,ion_hco3m,ion_co32m,ion_so42m,ion_nap,ion_clm


def CEQ_limestone_open (TC,pco2,S=0.,D=0.):
    '''
    !-----------------------------------------------------------------------
    !  function calculates calcium equilibrium concentration for limestone
    !  and the open system case
    ! input:
    !  TC                  - temperature [C]
    !  pco2                - CO2 pressure [atm]
    !  S                   - salinity [permil]
    !  D                   - depth [m]
    ! output:
    !  chem_ceq_limestone_open - mol / l => mol / m^3
    ! used:
    !  K1                  - mol / l
    !  KC                  - mol^2 / l^2
    !  KH                  - mol / l atm
    !  K2                  - mol / l
    !  written by Georg Kaufmann 03/01/2008
    !-----------------------------------------------------------------------
    '''
    # check for freezing conditions
    if (TC < 0.):
        # sys.exit('chem_ceq_limestone_open: T<0')
        ceq=0.
        return ceq
    # calculate mass balance coeeficients
    K1 = libCHEM.K1K2(TC,S,D)[0]
    K2 = libCHEM.K1K2(TC,S,D)[1]
    KH = libCHEM.KH(TC,S)
    KC = libCHEM.KC(TC,S,D)
    # loop over ionis strength
    strength=1.e-4
    for i in range(1,7):
        [ion_hp,ion_ca2p,ion_mg2p,ion_ohm,ion_hco3m,ion_co32m,ion_so42m,ion_nap,ion_clm]=libCHEM.ion_debyehueckel(strength,TC)
        kk = K1*KC*KH / (4.0*K2*ion_ca2p*ion_hco3m**2)
        ceq = (pco2*kk)**(1.0/3.0)
        strength = 3.0*ceq
       #print (k0,k1,k2,k5,kc,ka,kh,kw)
       #print (ion_hp,ion_ca2p,ion_mg2p,ion_ohm,ion_hco3m,ion_co32m,ion_so42m,ion_nap,ion_clm)
    # rescale to mol/m^3
    ceq = 1000.*ceq
    return ceq


def CEQ_limestone_closed (TC,pco2,S=0.,D=0.):
    '''
    !-----------------------------------------------------------------------
    !  function calculates calcium equilibrium concentration for limestone
    !  and the closed system case
    !  TC                  - temperature [C]
    !  pco2                - CO2 pressure [atm]
    !  S                   - salinity [permil]
    !  D                   - depth [m]
    ! output:
    !  chem_ceq_limestone_closed - mol / l => mol / m^3
    ! used:
    !  K1                  - mol / l
    !  KC                  - mol^2 / l^2
    !  KH                  - mol / l atm
    !  K2                  - mol / l
    !  written by Georg Kaufmann 03/01/2008
    !-----------------------------------------------------------------------
    '''
    # check for freezing conditions
    if (TC < 0.):
        # sys.exit('chem_ceq_limestone_open: T<0')
        ceq=0.
        return ceq
    # calculate mass balance coeeficients
    K1 = libCHEM.K1K2(TC,S,D)[0]
    K2 = libCHEM.K1K2(TC,S,D)[1]
    KH = libCHEM.KH(TC,S)
    KC = libCHEM.KC(TC,S,D)
    # loop over ionis strength
    strength=1.e-4
    for i in range(1,7):
        [ion_hp,ion_ca2p,ion_mg2p,ion_ohm,ion_hco3m,ion_co32m,ion_so42m,ion_nap,ion_clm]=libCHEM.ion_debyehueckel(strength,TC)
        kk = K1*KC*KH / (4.0*K2*ion_ca2p*ion_hco3m**2)
        a1 = 1.0
        a2 = 0.0
        a3 = kk / KH
        a4 = -kk * pco2
        ceq = libCHEM.cubic_root (a1,a2,a3,a4)
        strength = 3.0*ceq
       #print (k0,k1,k2,k5,kc,ka,kh,kw)
       #print (ion_hp,ion_ca2p,ion_mg2p,ion_ohm,ion_hco3m,ion_co32m,ion_so42m,ion_nap,ion_clm)
    # rescale to mol/m^3
    ceq = 1000.*ceq
    return ceq


def cubic_root (a1,a2,a3,a4):
    '''
    !-----------------------------------------------------------------------
    ! find roots of a cubic polynomial
    ! a1*x**3 + a2*x**2 + a3*x + a4 = 0
    ! procedure follows Bronnstein, p. 131ff
    !-----------------------------------------------------------------------
    '''
    import numpy as np
    a  = a2 / a1
    b  = a3 / a1
    c  = a4 / a1
    q = (a**2 - 3.0*b) / 9.0
    r = (2.0*a**3 - 9.0*a*b + 27.0*c) / 54.0
    if (r**2 < q**3):
        phi = np.arccos(r / np.sqrt(q**3))
        x1  = -2.0*np.sqrt(q) * np.cos(phi/3.0) - a/3.0
        x2  = -2.0*np.sqrt(q) * np.cos((phi+2.0*np.pi)/3.0) - a/3.0
        x3  = -2.0*np.sqrt(q) * np.cos((phi-2.0*np.pi)/3.0) - a/3.0
        ceq = x1
    else:
        aa = -(r + np.sqrt(r**2 - q**3))**(1.0/3.0)
        if (aa == 0.):
            bb = 0.0
        elif (aa != 0.):
            bb = q / aa
        ceq = (aa + bb) - a/3.0
    return ceq


def PWP(TC,pco2=0.00042):
    """
    -----------------------------------------------------------------------
    function calculates coefficients of PWP equation
    -----------------------------------------------------------------------
    version tabulated in Buhmann & Dreybrodt (1985)
    input:
    TC               - C
    pco2             - atm
    output:
    kappa1           - m/s
    kappa2           - m/s
    kappa3           - mol/m2/s
    kappa4           - m4/mol/s
    """
    TK = 273.16 + TC
    #-----------------------------------------------------------------------
    # CO_2 equilibrium coefficients
    #-----------------------------------------------------------------------
    kappa1 = 10.**(0.198 - 444./TK)
    kappa2 = 10.**(2.840 - 2177./TK)
    kappa3 = np.where(TC <= 25,
        10.**(-5.860 - 317./TK),
        10.**(-1.100 - 1737./TK))
    kappa4 = np.where(pco2 <= 0.05,
        10.**(-2.375 + 0.025*TC + 0.56*(-np.log10(pco2)-1.3)),
        10.**(-2.375 + 0.025*TC))
    #-----------------------------------------------------------------------
    # convert from original units
    #-----------------------------------------------------------------------
    kappa1 = 1.e-2 * kappa1          # cm/s -> m/s
    kappa2 = 1.e-2 * kappa2          # cm/s -> m/s
    kappa3 = 1.e-3 / 1.e-4 * kappa3  # mmol/cm2/s -> mol/m2/s
    kappa4 = 1.e-8 / 1.e-3 * kappa4  # cm4/mmol/s -> m4/mol/s
    return kappa1,kappa2,kappa3,kappa4


def FCaCO3 (ca2p,ca2peq,delta=0.01e-2):
    """
    !-----------------------------------------------------------------------
    ! Dreybrodt's (1996) flux relations for calcite 
    ! from: Buhmann,Dreybrodt (1985), Chem. Geol., 53, 109-124
    ! from:
    ! input:
    !   ca2p   - current Ca^2+ concentration [mol/m^3] 
    !   ca2peq - saturation Ca^2+ concentration [mol/m^3]
    !   delta  - radius of conduit [m]
    ! fixed:
    !   k1s    - linear rate constant [mol/m2/s]
    !   n1     - linear power-law exponent [-]
    !   n2     - non-linear power-law exponent [-]
    !   diff   - Diffusion constant [m^2/s]
    !   sw12    - switch for rate law
    ! output:
    !   flux   - Ca^2+ flux rate [mol/m^2/s]  
    !            which depends on saturation, for
    !            [Ca^2+] <= sw12 [Ca^2+]_eq the flux rate is linear
    !            [Ca^2+] >  sw12 [Ca^2+]_eq the flux rate is fourth-order
    !   iflux  - flag for flux law 
    !       1  - linear 
    !       2  - fourth-order
    !       3  - over- saturated
    !       4  - (undefined)
    !-----------------------------------------------------------------------
    """
    # fixed parameter values
    n1     = 1
    n2     = 4
    k1s    = 4.e-7
    diff   = 1.e-9
    sw12   = 0.9
    iflux  = -1
    # correction for diffusion for larger film thicknesses
    k1     = k1s / (1.+k1s*2.*delta/6./diff/ca2peq)
    # low-order linear flux rate
    if ca2p <= sw12*ca2peq:
        iflux = 1
        FCaCO3 = k1 * (1. - ca2p/ca2peq)**n1
    # higher-order non-linear flux rate
    elif ca2p >= sw12*ca2peq and ca2p <= ca2peq:
        iflux = 2
        k2    = k1 * (1. - sw12)**(n1-n2)
        FCaCO3 = k2 * (1. - ca2p/ca2peq)**n2
    # precipitation
    elif ca2p >= ca2peq:
        iflux = 3
        FCaCO3 = -k1 * (ca2p/ca2peq-1.)**n1
    # undefined
    else:
        iflux = 4
        FCaCO3 = 0.
    return FCaCO3
