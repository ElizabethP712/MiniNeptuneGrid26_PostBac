import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path

current_directory = Path.cwd()
references_directory_path = "Installation&Setup_Instructions/picasofiles/reference"
PYSYN_directory_path = "Installation&Setup_Instructions/picasofiles/grp/redcat/trds"
print(os.path.join(current_directory, references_directory_path))
print(os.path.join(current_directory, PYSYN_directory_path))

os.environ['picaso_refdata']= os.path.join(current_directory, references_directory_path)
os.environ['PYSYN_CDBS']= os.path.join(current_directory, PYSYN_directory_path)

import picaso.justdoit as jdi
import picaso.justplotit as jpi
import astropy.units as u
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline

from astropy import constants
from photochem.utils import zahnle_rx_and_thermo_files
from photochem.extensions import gasgiants # Import the gasgiant extensions

import json
from astroquery.mast import Observations
from photochem.utils import stars

import star_spectrum
import pickle
import requests

#from gridutils import make_grid
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import tarfile

import h5py
import sys
import logging
import faulthandler
import traceback
import socket
import multiprocessing as mp
from queue import Empty
from datetime import datetime

# Opacity table cache: loaded once per MPI rank process, then inherited by all
# forked subprocesses at zero cost (copy-on-write via fork).
_OPACITY_CK = None

# set up simple logging to stderr with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def setup_rank_debug_logging():
    """Attach rank/host metadata to logs and print uncaught tracebacks per MPI rank."""
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    hostname = socket.gethostname()

    # Emit Python-level fatal tracebacks (including crashes in C extensions when possible)
    faulthandler.enable(all_threads=True)

    def _ranked_excepthook(exc_type, exc_value, exc_tb):
        print(f"[UNCAUGHT EXCEPTION] rank={rank} host={hostname}", file=sys.stderr, flush=True)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
        sys.stderr.flush()

    sys.excepthook = _ranked_excepthook

    formatter = logging.Formatter(f'%(asctime)s %(levelname)s [rank={rank} host={hostname}]: %(message)s')
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

# Calculates the PT Profile Using PICASO; w/ K2-18b & G-star Assumptions for non-changing parameters; change mh, tint, and total_flux.

def calc_semi_major_SUN(Teq):
    """
    Calculates the semi-major distance from the Sun of a planet whose equilibrium temperature can vary.
    
    Parameters:
    
    Teq: float
        This is the equilibrium temperature (in Kelvin) calculated based on total flux (or otherwise) of the planet.

    Results:
    
    distance_AU: float
        Returns the distance from the planet to the Sun to maintain equilibrium temperature in AU.
    
    """
    luminosity_star = 3.846*(10**26) # in Watts for the Sun
    boltzmann_const = 5.670374419*(10**-8) # in W/m^2 * K^4 for the Sun
    distance_m = np.sqrt(luminosity_star / (16 * np.pi * boltzmann_const * (Teq**4)))
    distance_AU = distance_m / 1.496e+11
    return distance_AU

def calc_Teq_SUN(distance_AU):
    luminosity_star = 3.846*(10**26) # in Watts for the Sun
    boltzmann_const = 5.670374419*(10**-8) # in W/m^2 * K^4 for the Sun
    distance_m = distance_AU * 1.496e+11
    Teq = (((distance_m ** 2) * (16 * np.pi * boltzmann_const) / luminosity_star) ** (1/4))**(-1)
    return Teq

def mass_from_radius_chen_kipping_2017(R_rearth):

    """
    Estimate planet mass (Earth masses) from radius (Earth radii) using the
    Chen & Kipping (2017) piecewise power-law (Forecaster) relation,
    using the *inverted* form (given Rp -> Mp) as documented by the

    NASA Exoplanet Archive.

    Parameters
    ----------
    R_rearth : float

        Planet radius in Earth radii.
    Returns
    -------
    M_mearth : float
        Estimated planet mass in Earth masses.
    Notes
    -----
    Uses:
      R = log10(Rp/Re) = C + S * log10(Mp/Me)
      => log10(Mp/Me) = (log10(Rp/Re) - C)/S
    Valid regimes for Rp -> Mp (Archive):
      - Rp < 1.23 Re:         C=0.00346,  S=0.2790
      - 1.23 <= Rp < 11.1 Re: C=-0.0925,  S=0.589
      - 11.1 <= Rp <= 14.3 Re: degenerate (no unique mapping) -> error
      - Rp >= 14.3 Re:        C=-2.85,    S=0.881
    For sub-Neptunes (1.7–4 Re), this uses the 1.23–11.1 Re regime.
    """
    if R_rearth <= 0:
        raise ValueError("R_rearth must be > 0")

    # Degenerate region where Rp does not map uniquely to Mp

    if 11.1 <= R_rearth <= 14.3:
        raise ValueError(
            "Chen & Kipping (2017) inversion is degenerate for 11.1 <= Rp/Re <= 14.3; "
            "mass is not uniquely defined in this radius range."
        )
        
    logR = np.log10(R_rearth)

    if R_rearth < 1.23:
        C, S = 0.00346, 0.2790

    elif R_rearth < 11.1:
        C, S = -0.0925, 0.589

    else:  # R_rearth > 14.3
        C, S = -2.85, 0.881

    logM = (logR - C) / S

    return 10.0 ** logM

def PICASO_PT_Planet(rad_plan=1, log_mh=2.0, tint=60, semi_major_AU=1, ctoO=1, nlevel=91, nofczns=1, nstr_upper=85, rfacv=0.5, outputfile=None, pt_guillot=True, prior_out=None):

    """
    Calculates the semi-major distance from the Sun of a planet whose equilibrium temperature can vary.
    
    Parameters:

    rad_plan = float
        This is the radius of the planet in units of x Earth radius.
    mh = float
        This is the metallicity of the planet in units of log10 x Solar
    tint = float
        This is the internal temperature of the planet in units of Kelvin
    semi_major_AU = float
        This is the orbital distance of the planet from the star in units of AU.
    ctoO = float
        This is the carbon to oxygen ratio of the planet in units of x Solar C/O ratio.
    nlevel = float
        Number of plane-parallel levels in your code
    nofczns = float
        Number of convective zones
    nstr_upper = float
        Top most level of guessed convective zone
    rfacv = float
        Based on Mukherjee et al. Eqn. 20, this tells you how much of the hemisphere(s) is being irradiated; if stellar irradiation is 50% (one hemisphere), rfacv is 0.5 and if just night side then rfacv is 0. If tidally locked planet, rfacv is 1.
        
    Results: CHECK THIS WHEN RUNNING CASES THAT DIDN'T CONVERGE
    
    out: dictionary
        Creates an output file that contains pressure (bars), temperature (Kelvin), and whether the model converged or not (0 = False, 1 = True), along with all input data.
    basecase: dictionary
        Creates an output file that contains the original guesses for pressure and temperature.
    
    """

    print(f'Input Values: rad_plan={rad_plan}, mh={log_mh}, tint={tint}, semi_major_AU={semi_major_AU}, ctoO={ctoO}')
    
    # Values of Planet
    radius_planet = rad_plan*6.371e+6*u.m # Converts from units of xEarth radius to m

    # Use the 2017 M-R relationship to calculate mass in Earth units
    mass_planet_earth = mass_from_radius_chen_kipping_2017(R_rearth=rad_plan)
    mass_planet = mass_planet_earth*5.972e+24*u.kg # Converts from units of x Earth mass to kg
    grav = (const.G * (mass_planet)) / ((radius_planet)**2) # of planet
    
    # Load the opacity table once per process; forked subprocesses inherit it.
    global _OPACITY_CK
    if _OPACITY_CK is None:
        logging.info("Loading opacity table (first call in this process)...")
        _OPACITY_CK = jdi.opannection(method='resortrebin')
    opacity_ck = _OPACITY_CK

    # Values of the Host Star (assuming G-Star)
    T_star = 5778 # K, star effective temperature, the min value is 3500K 
    logg = 4.4 #logg , cgs
    metal = 0.0 # metallicity of star
    r_star = 1 # solar radius

    # Calculate Teq & Semi-Major Axis
    # What is the semi-major axis that is self-consistent?
    Teq = calc_Teq_SUN(semi_major_AU)
        
    # Starting Up the Run
    cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation 
    cl_run.gravity(gravity=grav.value, gravity_unit=u.Unit('m/(s**2)')) # input gravity
    cl_run.effective_temp(tint) # input effective temperature
    cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star, 
            radius_unit=u.R_sun,semi_major= semi_major_AU , semi_major_unit = u.AU )#opacity db, pysynphot database, temp, metallicity, logg

    # Initial T(P) Guess
    nstr_deep = nlevel -2
    nstr = np.array([0,nstr_upper,nstr_deep,0,0,0]) # initial guess of convective zones

    # Try to fix the convergence issue by using other results as best guesses
    #with h5py.File('results/PICASO_climate_fv.h5', 'r') as f:
    #    pressure = np.array(list(f['results']['pressure'][1][0][0]))
    #    temp_guess = np.array(list(f['results']['temperature'][1][0][0]))

    if pt_guillot == True:
        pt = cl_run.guillot_pt(Teq, nlevel=nlevel, T_int = tint, p_bottom=3, p_top=-6)
        temp_guess = pt['temperature'].values
        pressure = pt['pressure'].values
    elif pt_guillot == False:
        temp_guess = prior_out['temperature']
        pressure = prior_out['pressure']
    
    # Try using the T(P) profile from the test case instead of Guillot et al 2010.
    # with open('out_Sun_5778_initP3bar.pkl', 'rb') as file:
    #     out_Gstar = pickle.load(file)
    
    #temp_guess = pt['temperature'].values
    #pressure = pt['pressure'].values

    # Initial Convective Zone Guess
    cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure, 
                      nstr = nstr, nofczns = nofczns , rfacv = rfacv)

    # Set composition
    mh_converted_from_log = 10**log_mh
    cl_run.atmosphere(mh=mh_converted_from_log, cto_relative=ctoO, chem_method='on-the-fly')

    # Run Model
    try:
        out = cl_run.climate(opacity_ck, save_all_profiles=False,with_spec=False) # For now changed with_spec to False since it is an additional calculation after climate solve for the values I am saving, also just saving convergence results (save_all_profiles=False).
        # base_case = jdi.pd.read_csv(jdi.HJ_pt(), delim_whitespace=True)  # not used in final results

    except Exception as e:
        print(f"An exception was raised in PICASO_PT_Planet: {type(e).__name__}: {e}", file=sys.stderr)
        # Return an error dictionary instead of raising to prevent worker crash
        error_out = {
            'pressure': np.full(nlevel, np.nan),
            'temperature': np.full(nlevel, np.nan),
            'converged': 0,
            'status': 'error',
            'error': f"{type(e).__name__}: {e}"
        }

        #base_case = None
        
        return error_out
    
    # Extract only plain numpy-compatible values before returning.
    # The raw PICASO 'out' object contains pandas DataFrames and internal PICASO
    # state that cannot be pickled by the standard multiprocessing.Queue used in
    # _run_picaso_isolated.  Extracting here ensures only serializable data crosses
    # the queue boundary.
    safe_out = {
        'pressure':    np.array(out['pressure']),
        'temperature': np.array(out['temperature']),
        'converged':   np.array([out.get('converged', 0)]),
        'status':      out.get('status', ''),
        'error':       out.get('error', ''),
    }

    if outputfile is not None:
        with open(f'out_{outputfile}.pkl', 'wb') as f:
            pickle.dump(out, f)

    return safe_out
   
def PICASO_fake_climate_model_testing_errors(rad_plan, log_mh, tint, semi_major_AU, ctoO, outputfile=None):
    
    fake_dictionary = {'planet radius': np.full(10, rad_plan), 'log_mh': np.full(10, log_mh) , 'tint': np.full(10, tint), 'semi major': np.full(10, semi_major_AU), 'ctoO': np.full(10, ctoO)}
    return fake_dictionary

def _error_result(nlevel, status, error_message):
    """Build a uniform, h5py-compatible error payload for failed grid points."""
    return {
        'pressure': np.full(nlevel, np.nan),
        'temperature': np.full(nlevel, np.nan),
        'converged': np.array([0]),
        'status': np.array([status], dtype='U'),
        'error': np.array([error_message], dtype='U'),
    }

def _picaso_subprocess_worker(kwargs, queue):
    """Run one PICASO PT solve in an isolated process to contain native crashes."""
    try:
        out = PICASO_PT_Planet(**kwargs)
        queue.put({'ok': True, 'out': out})
    except Exception as exc:
        queue.put({'ok': False, 'error': f"{type(exc).__name__}: {exc}"})

def _run_picaso_isolated(kwargs, timeout_s):
    """
    Execute PICASO in a forked child process.
    This converts segfaults/timeouts into normal Python return states.
    Uses 'fork' (not 'spawn') so the child inherits already-imported modules
    from the parent, avoiding expensive re-imports from the shared filesystem
    that cause severe filesystem contention when many MPI workers spawn
    subprocesses simultaneously on HPC clusters.
    Note: the child process never calls MPI, so forking after MPI_Init is safe.
    """
    ctx = mp.get_context('fork')
    queue = ctx.Queue()
    proc = ctx.Process(target=_picaso_subprocess_worker, args=(kwargs, queue))
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            'ok': False,
            'kind': 'timeout',
            'error': f"PICASO subprocess timed out after {timeout_s}s",
            'exitcode': proc.exitcode,
        }

    if proc.exitcode != 0:
        signal_note = ''
        if proc.exitcode is not None and proc.exitcode < 0:
            signal_note = f" (signal {-proc.exitcode})"
        return {
            'ok': False,
            'kind': 'crash',
            'error': f"PICASO subprocess crashed with exit code {proc.exitcode}{signal_note}",
            'exitcode': proc.exitcode,
        }

    try:
        payload = queue.get(timeout=5)
    except Empty:
        return {
            'ok': False,
            'kind': 'no-payload',
            'error': 'PICASO subprocess exited cleanly but returned no payload',
            'exitcode': proc.exitcode,
        }

    if not payload.get('ok', False):
        return {
            'ok': False,
            'kind': 'python-exception',
            'error': payload.get('error', 'Unknown subprocess exception'),
            'exitcode': proc.exitcode,
        }

    return {
        'ok': True,
        'out': payload['out'],
        'exitcode': proc.exitcode,
    }

def _append_crash_fingerprint(stage, x, reason, details=None):
    """
    Write one-line crash fingerprints to a dedicated file.
    This runs on each worker rank and is intentionally append-only.
    """
    rank = 'unknown'
    host = socket.gethostname()
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        pass

    log_path = os.environ.get('PICASO_ERROR_LOG', 'results/picaso_crash_fingerprints.log')
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    x_list = np.atleast_1d(x).tolist()
    detail_text = '' if details is None else str(details)
    line = (
        f"{timestamp} rank={rank} host={host} stage={stage} "
        f"input={x_list} reason={reason} details={detail_text}\n"
    )

    try:
        with open(log_path, 'a', encoding='utf-8') as fp:
            fp.write(line)
    except Exception:
        # Logging should never break the simulation path.
        pass

def PICASO_climate_model(x):
    
    """
    This takes the values from get_gridvals_PICASO_TP and plugs them into PICASO_PT_Planet for parallel computing,
    then saves the results to new, simplified dictionary.

    Parameter(s):
    x: 1D array of input parameters in the order of total_flux, mh, then tint.
        mh = string like '0.0' in terms of solar metalicity
        tint = float like 70 in terms of Kelvin
        total_flux = float in terms of solar flux

    Results:
    new_out: dictionary
        This simplifies the output of PICASO into a dictionary with three keys,
        pressure at each iterated point in the profile in units of bars,
        temperature at each iterated point in the profile in units of Kelvin,
        Noting that both go from smaller value to larger value,
        and converged representing whether or not results converged (0 = False, 1 = True)
        
    """
    # For Tijuca
    rad_plan_earth_units, log10_planet_metallicity, tint_K, semi_major_AU, ctoO_solar = x
    logging.info(f"starting climate model for inputs: {x}")
    print(f'This is the value of {x} used in the climate model')

    # normalize/convert inputs
    try:
        rad_plan_earth_units = float(rad_plan_earth_units)
    except Exception:
        pass
    try:
        tint_K = float(tint_K)
    except Exception:
        pass
    try:
        semi_major_AU = float(semi_major_AU)
    except Exception:
        pass
    try:
        log10_planet_metallicity = float(log10_planet_metallicity)
    except Exception:
        pass
    try:
        ctoO_solar = float(ctoO_solar)
    except Exception:
        pass

    # Default fallback length for PT arrays (matches PICASO_PT_Planet default nlevel)
    _default_nlevel = 91
    max_retries = 3
    # Default timeout: 3600s (1 hour) per case.  Set PICASO_SUBPROCESS_TIMEOUT_S=0
    # (or 'none') in the environment to disable the timeout entirely.
    timeout_env = os.environ.get('PICASO_SUBPROCESS_TIMEOUT_S', '0').strip().lower()
    if timeout_env in ('', '0', 'none', 'false', 'off'):
        timeout_s = None
    else:
        timeout_s = int(timeout_env)

    # helper to make sure outputs have array dtype suitable for h5py
    def _sanitize(new_out_dict):
        # ensure everything is at least 1-D numpy array and convert U strings to S
        for kk, vv in list(new_out_dict.items()):
            if not isinstance(vv, np.ndarray):
                vv = np.atleast_1d(np.array(vv))
            if vv.dtype.kind == 'U':
                vv = vv.astype('S')
            new_out_dict[kk] = vv
        return new_out_dict

    # Try initial run in isolated subprocess so native crashes do not kill the MPI rank
    first_kwargs = {
        'rad_plan': rad_plan_earth_units,
        'log_mh': log10_planet_metallicity,
        'tint': tint_K,
        'semi_major_AU': semi_major_AU,
        'ctoO': ctoO_solar,
        'outputfile': None,
    }
    initial = _run_picaso_isolated(first_kwargs, timeout_s=timeout_s)
    if not initial.get('ok', False):
        err = initial.get('error', 'Unknown isolated-run failure')
        logging.error("PICASO run failed (initial) for inputs %s: %s", x, err)
        print(f"PICASO run failed (initial): {err}", file=sys.stderr)
        _append_crash_fingerprint(
            stage='initial',
            x=x,
            reason=initial.get('kind', 'isolated-failure'),
            details=err,
        )
        return _sanitize(_error_result(_default_nlevel, status='crash', error_message=err))

    out = initial['out']

    # If PICASO returned an error status, do not attempt retries
    if out.get('status') == 'error':
        print(f"PICASO returned status 'error' after initial run: {out.get('error', '')}")
        _append_crash_fingerprint(
            stage='initial',
            x=x,
            reason='picaso-status-error',
            details=out.get('error', 'Unknown error'),
        )
        return _sanitize(_error_result(_default_nlevel, status='error', error_message=out.get('error', 'Unknown error')))

    # If the model reports not converged, try a few more times using prior outputs
    count = 0
    while out.get('converged', 1) == 0 and out.get('status') != 'error' and count < max_retries:
        count += 1
        print(f"Loop iteration, Recalculating PT Profile: {count}")
        retry_kwargs = {
            'rad_plan': rad_plan_earth_units,
            'log_mh': log10_planet_metallicity,
            'tint': tint_K,
            'semi_major_AU': semi_major_AU,
            'ctoO': ctoO_solar,
            'outputfile': None,
            'pt_guillot': False,
            'prior_out': out,
        }
        retry = _run_picaso_isolated(retry_kwargs, timeout_s=timeout_s)
        if not retry.get('ok', False):
            err = retry.get('error', 'Unknown isolated-run failure')
            logging.error("PICASO run failed (recalc #%d) for inputs %s: %s", count, x, err)
            print(f"PICASO run failed (recalc #{count}): {err}", file=sys.stderr)
            _append_crash_fingerprint(
                stage=f'retry-{count}',
                x=x,
                reason=retry.get('kind', 'isolated-failure'),
                details=err,
            )
            return _sanitize(_error_result(_default_nlevel, status='crash', error_message=err))

        if retry['out'].get('status') == 'error':
            _append_crash_fingerprint(
                stage=f'retry-{count}',
                x=x,
                reason='picaso-status-error',
                details=retry['out'].get('error', 'Unknown error'),
            )

        out = retry['out']

        if count >= max_retries:
            print(f"Hit the maximum amount of loops ({max_retries}) without converging.")
            break

    # Prepare output dictionary with only the desired keys
    desired_keys = ['pressure', 'temperature', 'converged', 'status', 'error']
    new_out = {key: out[key] for key in desired_keys if key in out}
    # Ensure pressure and temperature exist; if missing, fill with NaNs
    if 'pressure' not in new_out:
        new_out['pressure'] = np.full(_default_nlevel, np.nan)
    if 'temperature' not in new_out:
        new_out['temperature'] = np.full(_default_nlevel, np.nan)

    # Ensure converged is an array
    new_out['converged'] = np.array([out.get('converged', 0)])

    # Add status: 'converged', 'not_converged', or 'error'
    if new_out['converged'][0] == 1:
        new_out['status'] = 'converged'
    else:
        new_out['status'] = 'not_converged'

    # ensure 'status' and 'error' are numpy arrays (so gridutils can treat them uniformly)
    new_out['status'] = np.array([new_out.get('status','')], dtype='U')
    new_out['error'] = np.array([new_out.get('error','')], dtype='U')

    # final pass: convert any remaining scalars/strings to numpy arrays and fix dtypes
    for k, v in list(new_out.items()):
        # Ensure scalars become at least 1D arrays (for len() calls in gridutils)
        if not isinstance(v, np.ndarray):
            v = np.atleast_1d(np.array(v))
        # Convert Unicode string dtype to byte string dtype for h5py compatibility
        if v.dtype.kind == 'U':
            v = v.astype('S')
        new_out[k] = v

    return new_out

def get_gridvals_PICASO_TP():

    
    """
    This provides the input parameters to run the climate model over multiple computers (i.e. paralell computing).

    Parameter(s):
    log10_totalflux = np.array of floats
        This is the total flux of the starlight on the planet in units of x Solar
    log10_planet_metallicity = np.array of floats (will be converted to strings internally)
        This is the planet metallicity in units of log10 x Solar
    tint = np.array of floats
        This is the internal temperature of the planet in units of Kelvin
    
    Returns:
        A tuple array of each array of input parameters to run via parallelization and return a 1D climate PT profile.
    
    """
    """
    # True Values to replace after test case:

    Convert float inputs to strings for metallicity and ctoO ratio:
    metal_float = np.linspace(3, 3000, 10)
    metal_string = np.array([str(f) for f in metal_float])

    rad_plan_earth_units = np.linspace(1.6, 4, 5) # in units of xEarth radii
    log10_planet_metallicity = metal_string # in units of solar metallicity, right now should be a list of strings
    tint_K = np.linspace(20, 400, 5) # in Kelvin
    semi_major_AU = np.linspace(0.3, 10, 10) # in AU 
    ctoO_solar = np.array(['0.01', '0.25', '0.5', '0.75', '1']) # in units of solar C/O

    """
    """

    # Test Case: this was the _updatop_test files
    rad_plan_earth_units = np.array([2.61]) # in units of xEarth radii
    log10_planet_metallicity = np.array([0.5]) # in units of solar metallicity
    tint_K = np.array([155]) # in Kelvin
    semi_major_AU = np.array([1]) # in AU 
    ctoO_solar = np.array([1]) # in units of solar C/O


    """

    
    # Full Parameter Exploration
    rad_plan_earth_units = np.array([2]) # in units of xEarth radii
    log10_planet_metallicity = np.linspace(0.5, 3.5, 9) # in units of solar metallicity
    tint_K = np.linspace(50, 400, 8) # in Kelvin
    semi_major_AU = np.array([0.3, 0.7, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]) # in AU 
    ctoO_solar = np.linspace(0.01, 1, 5) # in units of solar C/O

    """

    # Parameter Exploration Refined
    rad_plan_earth_units = np.array([1.6, 4]) # in units of xEarth radii
    log10_planet_metallicity = np.array([0.5, 3.5]) # in units of solar metallicity
    tint_K = np.array([100, 200, 300, 400]) # in Kelvin
    semi_major_AU = np.array([2,4,6,8,10]) # in AU 
    ctoO_solar = np.array([0.01, 1]) # in units of solar C/O

    """

    gridvals = (rad_plan_earth_units, log10_planet_metallicity, tint_K, semi_major_AU, ctoO_solar)
    
    return gridvals

if __name__ == "__main__":
    import gridutils

    """
    To execute running 1D PICASO climate model for the range of values in get_gridvals_PICASO_TP, type the folling command into your terminal:
    # mpiexec -n X python PICASO_Climate_grid_121625.py

    """
    setup_rank_debug_logging()
    
    gridutils.make_grid(
        model_func=PICASO_climate_model, 
        gridvals=get_gridvals_PICASO_TP(), 
        filename='results/PICASO_climate_updatop_full_exploration_reducedrad_solveSegFault.h5', 
        progress_filename='results/PICASO_climate_updatop_full_exploration_reducedrad_solveSegFault.log'
    ) 
