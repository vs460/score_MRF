import numpy as np
import scipy as sp
from scipy.integrate import odeint
import numpy
import matplotlib.pyplot as plt
import sys

# class _kineticModel():
def assign_concentration_compartements(C, types, frames):
    # =========================================================================
    # 	DESCRIPTION:
    #
    # 	INPUT:
    #        par:        parameter struct
    #
    #    required parameters:
    #
    #        C:              signal concentrations:
    #                        [P_blood_LV, P_blood_RV, ...
    #                         P_myo_healthy, L_myo_healthy, B_myo_healthy, ...
    #                         P_myo_infarct, L_myo_infarct, B_myo_infarct]
    #
    #       types:          mask types containing
    #
    #       frames:         number of dynamics
    #
    # 	OUTPUT:
    #        concentrations: time-dependent metabolite concentrations
    #                        {[P],[L],[B]}
    #                         - [P]: time-dependent pyruvate concentrations
    #                                for each mask compartment
    #                         - [L]: time-dependent lactate concentrations
    #                                for each mask compartment
    #                         - [B]: time-dependent bicarbonate concentrations
    #                                for each mask compartment
    #
    # 	VERSION HISTORY:
    #        180117JT Initial version
    # 	    JULIA TRAECHTLER (TRAECHTLER@BIOMED.EE.ETHZ.CH)
    # =========================================================================

    # concentration according to compartement (from mask)
    concentrations = [np.zeros((3, 100)), np.zeros((3, 100)), np.zeros((3, 100)), np.zeros((3, 100)),np.zeros((3, 100))]
    for i in range(0,len(types)):
        if types[i] == 'blood_RV':
            for j in range(0,np.size(C,2)):
                concentrations[j,:,i] = C[:,j,2]
        elif types[i] == 'blood_LV':
            for j in range(0,np.size(C,2)):
                concentrations[j,:,i] = C[:,j,1]
        elif types[i] == 'myo_healthy':
            for j in range(0,np.size(C,2)):
                concentrations[j,:,i] = C[:,j,3]
        elif types[i] == 'myo_infarct':
            for j in range(0,np.size(C,2)):
                concentrations[j,:,i] = C[:,j,4]
        elif types[i] == 'myo_risk':
            for j in range(0,np.size(C,2)):
                concentrations[j,:,i] = C[:,j,5]
    return concentrations

def compress_stretch(td, c, factor):
    # =========================================================================
    #
    # TITLE:
    # compress_stretch.m
    #
    # DESCRIPTION:
    # Compresses(factor > 1) / stretches(factor < 1) the time - discrete curve c by the factor 'factor'.
    #
    # INPUT:
    # td: time vector
    # c: time - discrete curve(same vector size as td)
    # factor: compress / stretch factor
    #  - factor > 1: compress
    #  - factor < 1: stretch
    #
    # OUTPUT:
    # c_mod: compressed / stretched curve
    #
    # VERSION HISTORY:
    # 180110JT Initial version
    #
    # JULIA TRAECHTLER(TRAECHTLER @ BIOMED.EE.ETHZ.CH)
    #
    # =========================================================================
    # compress / stretch
    tPeak = td[c.argmax()]
    c_mod = sp.interpolate.interp1d(td,c)((td - tPeak) / factor + tPeak)
    return c_mod

def convolve(t, c, h, q):
    dt = (t[-1]-t[0]) / (len(t)-1)
    c_ = np.zeros((len(c),len(c)))
    for i in range(0,len(c)):
        for j in range(0,len(c)):
            if(i+1-j>0):
                c_[i,j] = c[i-j]
    if np.size(h,0) == 1:
        h=np.transpose(h)
    c = q*c_*h*dt
    return c



def downsample(t_up, c_up, t_down):
    c_down = sp.interpolate.interp1d(t_up, c_up, kind='linear',bounds_error=False)(t_down)
    return c_down

def gammaVariateFunction(t, alpha=1,beta=1,A=1,t0=3):
    # =========================================================================
    #
    # 	TITLE:
    #        gammaVariateFunction.m
    #
    # 	DESCRIPTION:
    #        Computes the Gamma-variate function:
    #        f(t) = A*((t-t0)^alpha)*exp(-(t-t0)/beta)*(t>t0)
    #
    # 	INPUT:
    #        t:                             time point [s]
    #        params = [alpha, beta, A, t0]: parameters
    #
    # 	OUTPUT:
    #        f:                             Gamma-variate function
    #
    # 	VERSION HISTORY:
    #        171024JT Initial version
    #
    # 	    JULIA TRAECHTLER (TRAECHTLER@BIOMED.EE.ETHZ.CH)
    #
    # =========================================================================

    # Gamma-variate function
    f = A*((t-t0)^alpha)*np.exp(-(t-t0)/beta)*(t>t0)
    return f

def generate_concentration_dynamics(par, flag, path):
    # =========================================================================
    #
    # 	TITLE:
    #        generate_concentration_dynamics.m
    #
    # 	DESCRIPTION:
    #        Generates concentration dynamics of metabolites:
    #        here considered: pyruvate (P), lactate (L), bicarbonate (B)
    #
    # 	INPUT:
    #        par:        parameter struct
    #        flag:       flag struct
    #        path:       path struct
    #
    #    required parameters:
    #
    #        tDyn:           time vector [s]
    #
    #        k:              rate constants [1/s]
    #                        - format: [kpl,klp,kpb]
    #                           - kpl: rate constant of pyruvate -> lactate metabolism
    #                           - klp: rate constant of lactate -> pyruvate metabolism
    #                           - kpb: rate constant of pyruvate -> bicarbonate metabolism
    #
    #        C0:             initial signal concentrations
    #                        - format: [P0, L0, B0]
    #                           - P0: initial pyruvate concentration
    #                           - L0: initial lactate concentration
    #                           - B0: initial bicarbonate concentration
    #
    #        amplitude:      signal amplitudes
    #                        - format: [P_LV, P_RV, P_myo, L, B]
    #                           - P_LV:  amplitude of pyruvate in left ventricle
    #                           - P_RV:  amplitude of pyruvate in right ventricle
    #                           - P_myo: amplitude of pyruvate in myocardium
    #                           - L:     amplitude of lactate in myocardium
    #                           - B:     amplitude of bicarboante in myocardium
    #
    #        types:          mask types containing
    #
    # 	OUTPUT:
    #        concentrations: time-dependent metabolite concentrations
    #                        {[P],[L],[B]}
    #                         - [P]: time-dependent pyruvate concentrations
    #                                for each mask compartment
    #                         - [L]: time-dependent lactate concentrations
    #                                for each mask compartment
    #                         - [B]: time-dependent bicarbonate concentrations
    #                                for each mask compartment
    #
    # 	VERSION HISTORY:
    #        171024JT Initial version
    #        171127JT Cleanup
    #        180117JT Update: infarct concentration
    #
    # 	    JULIA TRAECHTLER (TRAECHTLER@BIOMED.EE.ETHZ.CH)
    #
    # =========================================================================

    print( '\n---------- Concentration dynamics ----------\n\n')
    print('  Number of dynamics:            %d \n'   , par["dynamics"])
    print('  Number of metabolites:         %d \n' , par["M"])

    # parameters
    TR            = par["TR"]
    td            = par["tDyn"]
    Nd            = par["dynamics"]

    k             = par["k"]
    C0            = par["C0"]
    Nm            = par["M"]
    ampl          = par["amplitude"]

    types         = par.maskParams.maskTypes
    T1            = par["T1"]
    MBF           = par["MBF"]

    shift_LV      = par["shift_LV"]
    scale_LV_FWHM = par["FWHM"]

    delay_RV_LV   = par["delay_RV_LV"]
    scale_RV_ampl = par["scale_RV"]
    compress_RV   = par["compress_RV"]

    fermi         = [par.fermi.alpha,par.fermi.beta]
    delay_LV_myo  = par["delay_LV_myo"]

    # pyruvate concentrations (normalized to P_LV)
    P_LV  = get_LV_concentration(td, Nd, TR, shift_LV, scale_LV_FWHM, flag, path, T1[0])
    P_RV  = get_RV_concentration(P_LV, td, TR, scale_RV_ampl, compress_RV, delay_RV_LV)
    P_LV  = P_LV[0:Nd]
    P_RV  = P_RV[0:Nd]

    # system of differential equations: dC(t)/dt = M * C(t) + C_in(t)
    N_type = np.size(k,3)
    C_myo = []
    for t in range(0,N_type):  # iterate over types (healthy, infarct, AAR):
        P_myo = get_myo_concentration(P_LV, td, TR, fermi, delay_LV_myo, MBF(t))
        P_myo = P_myo[1:Nd]
        MIFparams = {"P_myo":P_myo,"td":td, "TR":TR}

        M = np.zeros((Nm,Nm))
        M[0,0] = -sum(k[:,0,t-1])
        for i in range(0,np.size(k,1)):
            M[i+1,1] = k[i,1,t]
            M[i+1,i+1] = -k[i,2,t]
            M[1,i+1] = k[i,2,t]
        # # solution to differential equations system
        # [~,C] = ode15s(@diffEqu_system,td,C0,[],{M, MIF, MIFparams})
        # C_myo = cat(3,C_myo,C)

    # signal vector [C_LV, C_RV, C_Myo]
    C_LV = np.zeros((Nd,Nm))
    C_LV[:,1] = P_LV
    C_RV = np.zeros((Nd,Nm))
    C_RV[:,1] = P_RV
    if Nm-1>np.size(k,1):   # pyruvate hydrate
        C_LV[:,-1] = P_LV/10
        C_RV[:,-1] = P_RV/60
    C = np.concatenate((C_LV,C_RV,C_myo),axis=2)

    # signal amplitude
    C = C*ampl

    # concentration according to compartement (from mask)
    concentrations = assign_concentration_compartements(C, types, Nd)
    return concentrations

def get_AIF(frames):
    # =========================================================================
    #
    # 	TITLE:
    #        get_AIF.m
    #
    # 	DESCRIPTION:
    #        Defines AIF in LV at half dose.
    #
    # 	INPUT:
    #        frames:         number of dynamics
    #
    # 	OUTPUT:
    #   AIF:            arterial input function from @MRXCAT_CMR_PERF
    #
    # 	VERSION HISTORY:
    #        180110JT Initial version, see @MRXCAT_CMR_PERF
    #
    # 	    JULIA TRAECHTLER (TRAECHTLER@BIOMED.EE.ETHZ.CH)
    #
    # =========================================================================

    # ---------------------------------------------------------------------------------------------------------------------
    # Define AIF at half dose (pop avg of 6 healthy volunteers, 0.05 mmol/kg b.w. dose, first-pass by gamma-variate fit)
    # ---------------------------------------------------------------------------------------------------------------------
    aif005 = [0,0,0,0,0.00088825,0.017793,0.13429,0.5466,1.453,2.8276,4.3365,5.5112,6.0165,5.7934,5.0205,3.9772,2.9161,
    1.9988,1.2913,0.79165,0.46315,0.25983,0.14035,0.073254,0.037056,0.018216,0.0087227,0.0040768,0.0018632,
    0.00083405,0.00036621,0.00015793,6.6972e-05,2.7956e-05,1.1499e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------

    dose = 0.075  #[mmol/kg b.w.]

    # crop population average AIF
    if len(aif005) >= frames:
        aif005 = aif005[0:frames]
    else:
        aif005 = [aif005, np.zeros((1,frames-len(aif005)))]
    # --------------------------------------------------------------------
    #   Convert to absolute contrast agent concentration, using assumptions:
    #   - pure Gadovist: c_Gd = 1.0 mmol/ml = 1000 mmol/l
    #   - c = c_Gd*(dilution factor d)
    #   - estimation of dilution factor d:
    #       - 80-100 ml stroke volume
    #       - 80% ejection fraction (lost 20% not considered twice)
    #           - 64-80 ml eff. ejection volume
    #       - looking at normalized AIF shape (aif/trapz(aif))
    #       => peak value d ~ 0.12 = 12% (assume heart rate = 60 bpm)
    #   - absolute dose calc
    #       - e.g. patient weight 75 kg = 75*dose ml Gadovist injected
    #         (e.g. d=0.1 => 7.5 ml Gd)
    #       - e.g. 75 ml eff. ejection volume
    #       - max. 12% of 75*dose ml Gd per heart beat
    #         => e.g. 0.9 ml Gd / 75 ml blood => d = 12/1000
    #   - max c = c_Gd*d = 1000*12/1000 = 12 mmol/l
    # --------------------------------------------------------------------
    aif01 = aif005*12/max(aif005)
    AIF = aif01*dose/0.1    # scale to desired dose

def get_LV_concentration(td, Nd, TR, shift, scale_FWHM, flag, path, T1):
    # =========================================================================
    #
    # 	TITLE:
    #        get_LV_concentration.m
    #
    # 	DESCRIPTION:
    #        Calculates the pyruvate concentration in the left ventricle (LV)
    #        from a measured LV input function.
    #
    # 	INPUT:
    #        td:                     time vector [s]
    #        Nd:                     number of dynamics []
    #        TR:                     repetition time [s]
    #        shift:                  desired temporal shift of LV signal [s]
    #        scale_FWHM:             desired scaling factor of full width at
    #                                half maximum (FWHM) of LV signal []
    #        flag:                   flag struct
    #        path:                   path struct
    #        T1:                     T1 time constant [s]
    #
    # 	OUTPUT:
    #        concentration_LV:       concentration dynamic in LV
    #
    # 	VERSION HISTORY:
    #        171026JT Initial version
    #
    # 	    JULIA TRAECHTLER (TRAECHTLER@BIOMED.EE.ETHZ.CH)
    #
    # =========================================================================

    # flag initialization
    flag_bloodpool = flag["load_bloodpool"]

    # path initialization
    loadpath_bloodpool = path["loadpath_bloodpool"]

    # LV input function
    if flag_bloodpool:
        # measured LV input function
        # load(loadpath_bloodpool,'P_LV')
        # concentration_LV = P_LV-min(P_LV)
        # concentration_LV[end+1:len(td)] = 0
        # td = np.transpose([range(td[0],td[0]+len(concentration_LV)-1)])*TR
        concentration_LV = 0 #concentration_LV*np.exp(td/T1)
    else:
        # measured AIF from @MRXCAT_CMR_PERF
        concentration_LV = get_AIF(Nd)

    # apply time shift
    concentration_LV = timeshift(td, concentration_LV, shift)

    # compress according to desired FWHM
    # upsample
    [t_inf, c_inf] = upsample(td, concentration_LV, 100)
    # compress concentration
    c_inf = compress_stretch(t_inf, c_inf, scale_FWHM)
    # downsample
    concentration_LV = downsample(t_inf, c_inf, td)

    # normalize
    concentration_LV = concentration_LV/max(concentration_LV)
    return concentration_LV

def get_myo_concentration(concentration_LV, td, TR, fermi, delay, MBF):
    # parameter initialization
    alpha = fermi[0]
    beta = fermi[1]

    # td = [*range(td[0], td[0] + len(concentration_LV) - 1)]*TR

     # upsample AIF to "pseudo-continuous" AIF
    [t_inf, c_inf] = upsample(td, concentration_LV, 100)

    # scale flow and calculate impulse residue function
    # MBF = MBF * length(concentration_LV) / round(length(concentration_LV) * TR);
    IRF = fermiF(t_inf, alpha, beta)

    # calculate myocardial concentration
    concentration_myo = convolve(t_inf, c_inf, IRF, MBF)

    # downsample
    concentration_myo = downsample(t_inf, concentration_myo, td)

    # apply time shift
    concentration_myo = timeshift(td, concentration_myo, delay)

    return concentration_myo

def get_RV_concentration(concentration_LV, td, TR, scale_factor, compress_factor, delay):
    # =========================================================================
    #
    # 	TITLE:
    #        get_RV_concentration.m
    #
    # 	DESCRIPTION:
    #        Calculates the pyruvate concentration in the right ventricle (RV)
    #        from the pyruvate concentration in the left ventricle (LV).
    #
    # 	INPUT:
    #        concentration_LV:       concentration dynamic in LV
    #        td:                     time vector [s]
    #        TR:                     TR [s]
    #        scale_factor:           scaling factor of RV w.r.t LV amplitude []
    #        compress_factor:        compression factor of RV w.r.t LV signal []
    #        delay:                  temporal delay between RV and LV peak [s]
    #
    # 	OUTPUT:
    #        concentration_RV:       concentration dynamic in RV
    #
    # 	VERSION HISTORY:
    #        171026JT Initial version
    #        180110JT Update: scale, compress, shift
    #
    # 	    JULIA TRAECHTLER (TRAECHTLER@BIOMED.EE.ETHZ.CH)
    #
    # =========================================================================

    # compress
    td = np.transpose([range(td[0],td[0]+len(concentration_LV)-1)])*TR
    concentration_RV = compress_stretch(td, concentration_LV, compress_factor)

    # apply time shift
    concentration_RV = timeshift(td, concentration_RV, -delay)

    # scale
    concentration_RV = scale_factor*concentration_RV
    return concentration_RV


def fermiF(t,alpha = 0.25, beta = 0.25):
    f = (1 + beta)/(1 + beta*np.exp(alpha * np.array(t)))
    return f

def MIF(t,params):
    myoC = params[0]
    td = params[1]
    TR = params[2]

    # derivative
    tPeak = myoC.argmax()
    print(tPeak)
    dMyoCdt = np.zeros(np.size(myoC))
    dMyoCdt[0:tPeak] = np.gradient(myoC[0:tPeak])/TR

    # MIF
    MIF = max(0, sp.interpolate.interp1d(td[0:len(dMyoCdt)], dMyoCdt)(t))
    return MIF

def timeshift(td, f, dT):
    # =========================================================================
    #
    # 	TITLE:
    #        timeshift.m
    #
    # 	DESCRIPTION:
    #        Shift function f(t) by time dT along time t.
    #
    # 	INPUT:
    #        td:    time vector [s]
    #        f:     function f(t) to be shifted
    #        dT:    time shift [s]
    #
    # 	OUTPUT:
    #        g:  shifted function g(t) = f(t-dT)
    #
    # 	VERSION HISTORY:
    #        171026JT Initial version
    #        180111JT Update
    #
    # 	    JULIA TRAECHTLER (TRAECHTLER@BIOMED.EE.ETHZ.CH)
    #
    # =========================================================================

    # time shift
    g  = sp.interpolate.interp1d((td+dT), f, bounds_error=False)(td)
    return g

def upsample(td, c, factor):
    # =========================================================================
    #
    # 	TITLE:
    #        upsample.m
    #
    # 	DESCRIPTION:
    #        Upsample time-discrete curve c by the factor 'factor'.
    #
    # 	INPUT:
    #        td:             time vector
    #        c:              time-discrete curve (same vector size as td)
    #        factor:         upsampling factor
    #
    # 	OUTPUT:
    #        t_up:           upsampled time vector
    #        c_up:           upsampled curve vector
    #
    # 	VERSION HISTORY:
    #        180110JT Initial version
    #
    # 	    JULIA TRAECHTLER (TRAECHTLER@BIOMED.EE.ETHZ.CH)
    #
    # ========================================================================
    #  upsample
    t_up = [*np.arange(td[0], td[-1], 1/factor)]
    # c_up = interp1(td,c,t_up,'linear',0);
    c_up = sp.interpolate.interp1d(td, c)(t_up)
    return t_up, c_up

def diffEq(C,t,diffCmyo, kPL,kPB,kPA,kLP):
    Cpyr, Clac, Cbic, Clac = C
    dCpyrdt = -(kPL+kPB+kPA)*Cpyr + kLP*Clac + diffCmyo
    dClacdt = - kLP*Clac          + kPL*Cpyr
    dCbicdt =                     + kPB*Cpyr
    dCaladt =                     + kPA*Cpyr
    return dCpyrdt, dClacdt, dCbicdt, dCaladt

# if __name__ == "__main__":
def calcConcentrations(MBF, alpha, beta, kPL, kPB, kPA, kLP, T1):
    td = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,
         54,56,58,60,62,64,66,68,70,72,74,76,78])
    t = np.interp(np.linspace(0, max(td), 500), range(1, len(td)+1), td)
    dt = (t[-1]-t[0])/len(t)
    TR = (td[-1]-td[0])/len(td)
    T1pyr = 30
    #########################################################################################################
#     print(sys.argv)
#     MBF, alpha, beta   = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])  ### parameters to vary ###
#     kPL, kPB, kPA, kLP = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]) 
#     T1                 = float(sys.argv[8])
    #########################################################################################################
    T1dec = np.exp(-t/T1)
    IRF = fermiF(t,alpha, beta)
    Cpyr_LV = [0, 0.89513850, 1.8972744, 10.790848, 30.563053, 57.440395, 80.071205, 90,
               76.130241, 52.073250, 29.491318, 18.227985, 15.567835, 16.515390, 16.924950,
               16.174492, 14.419795, 12.124231, 10.043180, 8.3371353, 7.1633506, 6.4394593,
               5.8426766, 5.2093320, 4.4761543, 3.8443687, 3.3361158, 2.9166200, 2.6802750,
               2.3883140, 2.0865772, 1.8474436, 1.6728854, 1.5486465, 1.3587548, 1.1962585,
               1.1535619, 1.1254175, 0.95994794, 0.84657621]
    Cpyr_myo_meas = [0, 0, 0, 0.0030567343, 0.015033339, 0.051462583, 0.16244209, 0.43260989,
                     0.92704833, 1.6203786, 2.3731024, 2.9627583, 3.1768522, 3.0244234, 2.6150560,
                     2.1052132, 1.6961468, 1.3690621, 1.1038253, 0.88908082, 0.71610618, 0.57700330,
                     0.46540159, 0.37541157, 0.30267227, 0.24387883, 0.19647875, 0.15832572, 0.12760884,
                     0.10285904, 0.082863934, 0.066747583, 0.053710934, 0.043245900, 0.034833867, 0.028072823,
                     0.022625165, 0.018229241, 0.014681842, 0.011828410]
    Cpyr_LV = np.interp(np.linspace(0,max(td),500),range(1,len(td)+1),Cpyr_LV)
    Cpyr_myo_meas = np.interp(np.linspace(0,max(td),500),range(1,len(td)+1),Cpyr_myo_meas)
    Cpyr_LV = Cpyr_LV/np.exp(-t/T1pyr)
    # Cmyo = get_myo_concentration(Cpyr_LV, td, TR, (0.25,0.25), 2, MBF)
    Cmyo = MBF * np.convolve(IRF, Cpyr_LV,'full')*dt
    # diffCmyo = MIF(t,[Cmyo,td,2])
    diffCmyo = np.diff(Cmyo)/TR
    # plt.plot(IRF)
    # plt.title('IRF')
    # plt.show()
    # plt.plot(Cmyo)
    # plt.title('Cmyo')
    # plt.show()
    # plt.plot(diffCmyo)
    # plt.title('diffCmyo')
    # plt.show()
    n = len(t)
    C0 = [0, 0, 0, 0]
    Cpyr = np.empty_like(t)
    Clac = np.empty_like(t)
    Cbic = np.empty_like(t)
    Cala = np.empty_like(t)
    # record initial condi
    # solve ODE
    for i in range(1, n):
        # span for next time step
        tspan = [t[i - 1], t[i]]
        # solve for next step
        C = odeint(diffEq, C0, tspan, args=(Cmyo[i], kPL, kPB, kPA, kLP))
        # store solution for plotting
        Cpyr[i] = C[1][0]
        Clac[i] = C[1][1]
        Cbic[i] = C[1][2]
        Cala[i] = C[1][3]
        # next initial condition
        C0 = C[1]

    # T1 decay
    Cpyr = Cpyr*T1dec
    Clac = Clac*T1dec
    Cbic = Cbic*T1dec
    Cala = Cala*T1dec
    # plt.plot(t, Cpyr_myo_meas, '-',label='pyr myo meas')
    # plt.plot(t, Cpyr, label='pyr myo')
    # plt.plot(t, Clac, label='lac')
    # plt.plot(t, Cbic, label='bic')
    # plt.plot(t, Cala, label='ala')
    # plt.plot(t, Cpyr_LV/10, label='pyr LV meas /10')
    # plt.legend(loc="upper left")
    # plt.xlabel("Time [s]")
    # plt.show()
    return Cpyr, Clac, Cbic, Cala, t, td
