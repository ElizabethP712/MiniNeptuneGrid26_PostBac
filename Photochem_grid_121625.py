import warnings
warnings.filterwarnings('ignore')

import os
import time
from pathlib import Path

current_directory = Path.cwd()
references_directory_path = "Installation&Setup_Instructions/picasofiles/reference"
PYSYN_directory_path = "Installation&Setup_Instructions/picasofiles/grp/redcat/trds"
os.environ['picaso_refdata']= os.path.join(current_directory, references_directory_path)
os.environ['PYSYN_CDBS']= os.path.join(current_directory, PYSYN_directory_path)

import numpy as np
#%matplotlib inline

from photochem.utils import zahnle_rx_and_thermo_files
from photochem.extensions import gasgiants # Import the gasgiant extensions

import star_spectrum

from mpi4py import MPI

#from gridutils import make_grid
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import gridutils
import h5py
import PICASO_Climate_grid_121625 as PICASO_Climate_grid
import sys
import logging
import faulthandler
import traceback
import socket
import builtins
from datetime import datetime

# set up simple logging to stderr with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

_ORIGINAL_PRINT = builtins.print
_PICASO_PT_CACHE = {}
_READY_STELLAR_FLUX_FILES = set()
_CHEMISTRY_FILES_READY = False
_FILE_READY_POLL_S = 0.1
_FILE_READY_TIMEOUT_S = 300.0

def _files_are_ready(paths):
    """A shared file is ready once it exists and has non-zero size."""
    for path in paths:
        if not os.path.exists(path):
            return False
        try:
            if os.path.getsize(path) <= 0:
                return False
        except OSError:
            return False
    return True

def _wait_for_files(paths, timeout_s=_FILE_READY_TIMEOUT_S):
    """Wait until a set of files exists and has non-zero size."""
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while not _files_are_ready(paths):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for files: {paths}")
        time.sleep(_FILE_READY_POLL_S)

def _try_acquire_lock(lock_path):
    """Attempt to create a lock file atomically."""
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(lock_fd)
    return True

def _ensure_stellar_flux_file(planet_Teq):
    """Create each stellar flux file once and reuse it across subsequent model calls."""
    stellar_flux_file = f'sun_flux_file_{planet_Teq}'
    if stellar_flux_file in _READY_STELLAR_FLUX_FILES and _files_are_ready([stellar_flux_file]):
        return stellar_flux_file

    lock_path = f'{stellar_flux_file}.lock'
    temp_path = f'{stellar_flux_file}.tmp.{socket.gethostname()}.{os.getpid()}'
    lock_acquired = False
    try:
        if not _files_are_ready([stellar_flux_file]):
            lock_acquired = _try_acquire_lock(lock_path)
            if lock_acquired:
                try:
                    if not _files_are_ready([stellar_flux_file]):
                        star_spectrum.solar_spectrum(Teq=planet_Teq, outputfile=temp_path)
                        os.replace(temp_path, stellar_flux_file)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                    lock_acquired = False
            else:
                _wait_for_files([stellar_flux_file])

        _wait_for_files([stellar_flux_file])
        _READY_STELLAR_FLUX_FILES.add(stellar_flux_file)
        return stellar_flux_file
    finally:
        if lock_acquired and os.path.exists(lock_path):
            os.remove(lock_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)

def _ensure_photochem_input_files(atoms_names):
    """Generate shared Photochem chemistry files once and reuse them."""
    global _CHEMISTRY_FILES_READY

    rxns_filename = 'photochem_rxns.yaml'
    thermo_filename = 'photochem_thermo.yaml'
    file_paths = [rxns_filename, thermo_filename]
    if _CHEMISTRY_FILES_READY and _files_are_ready(file_paths):
        return rxns_filename, thermo_filename

    lock_path = 'photochem_inputs.lock'
    lock_acquired = False
    try:
        if not _files_are_ready(file_paths):
            lock_acquired = _try_acquire_lock(lock_path)
            if lock_acquired:
                try:
                    if not _files_are_ready(file_paths):
                        zahnle_rx_and_thermo_files(
                        atoms_names=atoms_names,
                        rxns_filename=rxns_filename,
                        thermo_filename=thermo_filename,
                        remove_reaction_particles=True # For gas giants, we should always leave out reaction particles.
                        )
                finally:
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                    lock_acquired = False
            else:
                _wait_for_files(file_paths)

        _wait_for_files(file_paths)
        _CHEMISTRY_FILES_READY = True
        return rxns_filename, thermo_filename
    finally:
        if lock_acquired and os.path.exists(lock_path):
            os.remove(lock_path)

def _build_value_index(values):
    """Map exact grid values to integer indices for fast lookup."""
    return {float(value): index for index, value in enumerate(np.asarray(values, dtype=float))}

def _get_picaso_pt_cache(filename, gridvals):
    """Load the PICASO PT grid once per rank and cache direct index maps."""
    cache = _PICASO_PT_CACHE.get(filename)
    if cache is None:
        with h5py.File(filename, 'r') as f:
            results = f['results']
            cache = {
                'index_maps': {
                    'planet_radius': _build_value_index(gridvals[0]),
                    'planet_metallicity': _build_value_index(gridvals[1]),
                    'tint': _build_value_index(gridvals[2]),
                    'semi_major': _build_value_index(gridvals[3]),
                    'ctoO': _build_value_index(gridvals[4]),
                },
                'pressure': np.array(results['pressure']),
                'temperature': np.array(results['temperature']),
                'converged': np.array(results['converged']),
                'status': np.array(results['status']) if 'status' in results else None,
                'error': np.array(results['error']) if 'error' in results else None,
            }
        _PICASO_PT_CACHE[filename] = cache
    return cache

def setup_rank_debug_logging():
    """Attach rank/host metadata to logs and print uncaught tracebacks per MPI rank."""
    rank = MPI.COMM_WORLD.Get_rank()
    hostname = socket.gethostname()

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

    # Prefix Python print output with the MPI rank to mirror tagged stdout like [1].
    if not getattr(builtins.print, '_rank_wrapped', False):
        def _ranked_print(*args, **kwargs):
            kwargs_local = dict(kwargs)
            sep = kwargs_local.pop('sep', ' ')
            text = sep.join(str(arg) for arg in args)
            _ORIGINAL_PRINT(f'[{rank}] {text}', **kwargs_local)

        _ranked_print._rank_wrapped = True
        builtins.print = _ranked_print

def _append_crash_fingerprint(stage, x, reason, details=None):
    """
    Write one-line crash fingerprints to a dedicated file.
    This runs on each worker rank and is intentionally append-only.
    """
    rank = 'unknown'
    host = socket.gethostname()
    try:
        rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        pass

    log_path = os.environ.get('PHOTOCHEM_ERROR_LOG', 'results/photochem_crash_fingerprints.log')
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
        pass

_PC_NLAYERS = 100  # matches interpolate_photochem_result_to_nlayers

def _photochem_nan_result(nlayers=_PC_NLAYERS):
    """Return NaN-filled sol/soleq dicts and zero convergence arrays for failed grid points."""
    nan_arr = np.full(nlayers, np.nan)
    sol_nan = {'pressure': nan_arr.copy(), 'temperature': nan_arr.copy(), 'Kzz': nan_arr.copy()}
    soleq_nan = {'pressure': nan_arr.copy(), 'temperature': nan_arr.copy(), 'Kzz': nan_arr.copy()}
    zero_arr = np.zeros(nlayers, dtype=np.uint8)
    return sol_nan, soleq_nan, zero_arr, zero_arr

# Finds the associated PT profile and calculates Photochemical Composition of a Planet

def find_PT_grid(filename='results/PICASO_climate_updatop_full_exploration_reducedrad_solveSegFault.h5', rad_plan=None, log10_planet_metallicity=None, tint=None, semi_major=None, ctoO=None, gridvals=PICASO_Climate_grid.get_gridvals_PICASO_TP()):
    """
    This finds the matching PT profile in the PICASO grid to be used for Photochem grid calculation.
    
    Parameters:
    filename: string
        This is the directory path to the PICASO grid being referenced (this is the output of makegrid for climate model using PICASO)
    rad_plan = float
        This is the radius of the planet in units of x Earth radius.
    mh = string
        This is the metallicity of the planet in units of log10 x Solar
    tint = float
        This is the internal temperature of the planet in units of Kelvin
    semi_major_AU = float
        This is the orbital distance of the planet from the star in units of AU.
    ctoO = float
        This is the carbon to oxygen ratio of the planet in units of x Solar C/O ratio
    gridvals: function
        This calls the parameter space used to create the grid whose path was given in filename (i.e. PICASO by default).
        
    Results:
    PT_list: 2D array
        This provides the matching pressures in a list (small to large, bars), then temperatures (small to large, Kelvin) in a list of the matching input values from the PICASO grid.
    convergence_values: 1D array
        This provides whether or not the PICASO model converged (0 = False, 1 = True).
    
    """
    cache = _get_picaso_pt_cache(filename, gridvals)

    try:
        indices = (
            cache['index_maps']['planet_radius'][float(rad_plan)],
            cache['index_maps']['planet_metallicity'][float(log10_planet_metallicity)],
            cache['index_maps']['tint'][float(tint)],
            cache['index_maps']['semi_major'][float(semi_major)],
            cache['index_maps']['ctoO'][float(ctoO)],
        )
    except KeyError:
        print('A match given total flux, planet metallicity, and tint does not exist')
        return None, None, 'PICASO Not Converged/Error Value', 'No matching PICASO grid point found'

    pressure_values = np.array(cache['pressure'][indices])
    temperature_values = np.array(cache['temperature'][indices])
    convergence_values = np.array(cache['converged'][indices])
    PT_list = pressure_values, temperature_values

    # Read PICASO status and error if available; guard against old h5 files.
    picaso_status = b'converged'
    picaso_error = b''
    if cache['status'] is not None:
        picaso_status = np.array(cache['status'][indices]).flat[0]
    if cache['error'] is not None:
        picaso_error = np.array(cache['error'][indices]).flat[0]

    if picaso_status != b'converged' or picaso_error != b'':
        status_str = picaso_status.decode('utf-8') if isinstance(picaso_status, bytes) else str(picaso_status)
        error_str = picaso_error.decode('utf-8') if isinstance(picaso_error, bytes) else str(picaso_error)
        print(f'PICASO did not converge or has an error: status={status_str}, error={error_str}')
        return None, None, f'PICASO: {status_str}', error_str

    print('Was able to successfully find your input parameters in the PICASO TP profile grid!')
    return PT_list, convergence_values, 'PICASO-converged', ''

def linear_extrapolate_TP(P, T):
    
    """
    This extends the pressure and temperature profile from PT deeper into the atmosphere.
    
    Parameters:
    P: 1D np.array
        This is the pressure (currently set to be in PICASO default order & units (ascending, bars).
        
    T: 1D np.array
        This is the temperature (currently set to be in PICASO default order & units (ascending, K)
        
    Results:
    P: 1D np.array
        This returns a pressure array that has been extended based on a linear model mapped to a logarithmic pressure and linear Temperature of the last 5 points that adds 13 additional points up to a pressure of 10^6 bars. Units are still ascending bars.
    T: 1D np.array 
        This returns a temperature array associated with the additional pressure points added. Units are still ascending Kelvin.
    
    """
    
    new_temperature = T[-5:]
    new_pressure = np.log10(P[-5:])
    m, b = np.polyfit(new_temperature, new_pressure, 1)
    x_final = new_pressure[-5:][-1]
    
    add_pres_val = np.linspace(x_final, 6, 13)
    add_pres_val_rm_final = add_pres_val[1:]
    add_temp_val = (add_pres_val_rm_final - b)/m
    
    P = np.concatenate((P, np.array(10**add_pres_val_rm_final)))
    T = np.concatenate((T, np.array(add_temp_val)))

    return P, T

# Make it so the sol and soleq are the same length (needed for saving to h5)
def interpolate_photochem_result_to_nlayers(out, nlayers):

    """
    This makes sure the output arrays are the same length in resolution by interpolating results specific to pressure, temperature, Kzz, and mixing ratios.
    
    Parameters:
    out: dictionary
        This is the output you get from applying Photochem_Gas_Giant, or something similar with keys of np.arrays.
    nlayers: float
        This is how many values you wish to maintain in your grid.
      
    Results:
    sol: dictionary
        Each key's valued array is now the length of nlayers.
    
    """
    
    sol = {}

    # Make a new array of pressures
    sol['pressure'] = np.logspace(np.log10(np.max(out['pressure'])),np.log10(np.min(out['pressure'])),nlayers)
    log10P_new = np.log10(sol['pressure'][::-1]).copy() # log space and flipped of new pressures
    log10P = np.log10(out['pressure'][::-1]).copy() # log space and flipped old pressures

    # Do a log-linear interpolation of temperature
    T = np.interp(log10P_new, log10P, out['temperature'][::-1].copy())
    sol['temperature'] = T[::-1].copy()

    # Do a log-log interpolation of Kzz
    Kzz = np.interp(log10P_new, log10P, np.log10(out['Kzz'][::-1].copy()))
    sol['Kzz'] = 10.0**Kzz[::-1].copy()

    # Do a log-log interpolation of mixing ratios
    for key in out:
        if key not in ['pressure','temperature','Kzz']:
            tmp = np.log10(np.clip(out[key][::-1].copy(),a_min=1e-100,a_max=np.inf))
            mix = np.interp(log10P_new, log10P, tmp)
            sol[key] = 10.0**mix[::-1].copy()

    return sol

# Calculates the Chemical Composition of the Planet using Photochem


def Photochem_Gas_Giant(rad_plan=None, log10_planet_metallicity=None, tint=None, semi_major=None, ctoO=None, log_Kzz=None, PT_filename=None):

    """
    This calculates the 1D photochemical composition of a K218b-like planet around a Sun-like star.
    
    Parameters:
    rad_plan = float
        This is the radius of the planet in units of x Earth radius. Should be same as PICASO grid. 
    log10_planet_metallicity = float
        This is the metallicity of the planet in units of log10 x Solar. Should be same as PICASO grid. 
    tint = float
        This is the internal temperature of the planet in units of Kelvin. Should be same as PICASO grid. 
    semi_major_AU = float
        This is the orbital distance of the planet from the star in units of AU. Should be same as PICASO grid. 
    ctoO = float
        This is the carbon to oxygen ratio of the planet in units of x Solar C/O ratio. Should be same as PICASO grid. 
    log_Kzz: float
        This is the exponent of the eddy diffusion coefficient you want to use in units of cm^2/s.
    PT_filename: string
        This is the path to the PICASO PT grid you would like to use.
      
    Results:
    sol: dictionary of same length 1D arrays
        These are the molecular abundances %/total # of elements (all would be a value of 1), the pressures with each layer (this time in descending order in dynes/cm^2), the temperatures with each layer (this time in descending order in Kelvin) when a steady state was reached.
    soleq: dictionary of same length 1D arrays
        Same as sol, but when the molecules were in chemical equilibrium.
    pc: IDK
        This kept track of the inputs into the .EvoAtmosphereGasGiants function from Photochem.
    convergence_values: 1D array
        These were the values where PICASO either converged (i.e. 1 = True) or did not (i.e. 0 = False). 
    converged: 1D array
        These were the values where Photochem either converged (i.e. 1 = True) or did not (i.e. 0 = False). 
    
    """

    # Planet Parameters
    atoms_names = ['H', 'He', 'N', 'O', 'C'] # We select a subset of the atoms in zahnle_earth.yaml (leave out Cl), remove Sulpher for faster convergence

    # Calculate the Mass of the Planet and Teq
    mass_planet_earth = PICASO_Climate_grid.mass_from_radius_chen_kipping_2017(R_rearth=rad_plan)
    mass_planet = mass_planet_earth * (5.972e+24) * 1e3 # of planet, but in grams
    radius_planet = rad_plan * (6.371e+6) * 1e2 # of planet but in cm
    solar_zenith_angle = 60 # Used in Tsai et. al. (2023), in degrees
    planet_Teq = PICASO_Climate_grid.calc_Teq_SUN(distance_AU=semi_major)

    # Dependent constant variables
    stellar_flux_file = _ensure_stellar_flux_file(planet_Teq)
    rank = MPI.COMM_WORLD.Get_rank()

    PT_list, convergence_values, picaso_status, picaso_error = find_PT_grid(filename=PT_filename, rad_plan=rad_plan, log10_planet_metallicity=log10_planet_metallicity, tint=tint, semi_major=semi_major, ctoO=ctoO)

    # If PICASO did not converge or had an error, return NaN result immediately.
    if PT_list is None:
        sol_nan, soleq_nan, conv_nan, conv_pc_nan = _photochem_nan_result()
        return sol_nan, soleq_nan, None, conv_nan, conv_pc_nan, picaso_status, picaso_error

    # Test Data - This works fine.
    #with open('out_Sun_5778_initp3bar.pkl', 'rb') as file:
    #    out_reopened = pickle.load(file)
    #    pressure = out_reopened['pressure']
    #    temperature = out_reopened['temperature']
    #PT_list = np.array(pressure), np.array(temperature)
    #convergence_values = np.array([1])

    # Define P-T Profile (convert from PICASO to Photochem)
    P_extended, T_extended = linear_extrapolate_TP(PT_list[0], PT_list[1]) # Extend the end to bypass BOA Error of mismatching boundary conditions.
    #P = np.flip(np.array(PT_list[0]) * (10**6)).copy()
    #T = np.flip(np.array(PT_list[1])).copy()
    P = np.flip(np.array(P_extended) * (10**6)).copy() # Convert from bars to dynes/cm^2
    T = np.flip(np.array(T_extended)).copy()
    
    # Check if numpy array is sorted (investigating error)
    # sorted_P = np.flip(np.sort(P)).copy()
    # unsorted_indices = np.where(P != sorted_P)[0]
    
    rxns_filename, thermo_filename = _ensure_photochem_input_files(atoms_names)

    try:
        # Initialize ExoAtmosphereGasGiant
        # Assigns 
        pc = gasgiants.EvoAtmosphereGasGiant(
            mechanism_file=rxns_filename,
            stellar_flux_file=stellar_flux_file,
            planet_mass=mass_planet,
            planet_radius=radius_planet,
            solar_zenith_angle=solar_zenith_angle,
            thermo_file=thermo_filename
        )
        # Adjust convergence parameters:
        pc.var.conv_longdy = 0.03 # converges at 3% (change of mixing ratios over long time)
        pc.gdat.max_total_step = 10000 # assumes convergence after 10,000 steps
        
        pc.gdat.verbose = True # printing
        
        # Define the host star composition
        molfracs_atoms_sun = np.ones(len(pc.gdat.gas.atoms_names))*1e-10 # This is for the Sun
        comp = {
            'H' : 9.21e-01,
            'N' : 6.23e-05,
            'O' : 4.51e-04,
            'C' : 2.48e-04,
            'S' : 1.21e-05,
            'He' : 7.84e-02
        }

        tot = sum(comp.values())
        for key in comp:
            comp[key] /= tot
        for i,atom in enumerate(pc.gdat.gas.atoms_names):
            molfracs_atoms_sun[i] = comp[atom]
        
        pc.gdat.gas.molfracs_atoms_sun = molfracs_atoms_sun

        # Assume a default radius for particles 1e-5cm was default, so we increased the size but think of these in microns
        particle_radius = pc.var.particle_radius
        particle_radius[:,:] = 1e-3 #cm or 10 microns
        pc.var.particle_radius = particle_radius

        # Assumed Kzz (cm^2/s) in Tsai et al. (2023)
        Kzz_zero_grid = np.ones(P.shape[0])
        Kzz = Kzz_zero_grid*(10**log_Kzz) #Note Kzz_fac was meant to be the power of 10 since we are in log10 space

        # Initialize the PT based on chemical equilibrium 
        pc.gdat.BOA_pressure_factor = 3
        pc.initialize_to_climate_equilibrium_PT(P, T, Kzz, 10**(log10_planet_metallicity), ctoO)
        
        # Integrate to steady state
        converged_PC = pc.find_steady_state()

        # Check if the model converged after 10,000 steps
        if not converged_PC:
            assert pc.gdat.total_step_counter > pc.gdat.max_total_step - 10
            
        sol_raw = pc.return_atmosphere()
        soleq_raw = pc.return_atmosphere(equilibrium=True)

        # Call the interpolation of the grid 
        sol = interpolate_photochem_result_to_nlayers(out=sol_raw, nlayers=100)
        soleq = interpolate_photochem_result_to_nlayers(out=soleq_raw, nlayers=100)
        convergence_values = np.full(
            len(sol['pressure']),
            np.uint8(convergence_values[0]),
            dtype=np.uint8,
        )
        converged_PC_arr = np.full(
            len(sol['pressure']),
            np.uint8(1 if converged_PC else 0),
            dtype=np.uint8,
        )

        # Print out the lengths of arrays: Save the size of the grid for future reference.
        print(f"This is for the input value of planet radius:{rad_plan}, metal:{float(log10_planet_metallicity)}, tint:{tint}, semi major:{semi_major}, ctoO: {ctoO}, log_Kzz: {log_Kzz}")

        # Add nan's to fit the grid if underestimated, and make sure list goes from largest to smallest.

        if converged_PC:
            pc_status = 'Photochem-converged'
        else:
            pc_status = 'Photochem-not-converged'

        return sol, soleq, pc, convergence_values, converged_PC_arr, pc_status, ''

    except Exception as e:
        err_str = f"{type(e).__name__}: {e}"
        print(f"Photochem exception for inputs rad_plan={rad_plan}, metal={log10_planet_metallicity}, tint={tint}, semi_major={semi_major}, ctoO={ctoO}, log_Kzz={log_Kzz}: {err_str}", file=sys.stderr)
        sol_nan, soleq_nan, conv_nan, conv_pc_nan = _photochem_nan_result()
        return sol_nan, soleq_nan, None, conv_nan, conv_pc_nan, 'Photochem-error', err_str

def get_gridvals_Photochem():

    """
    This provides the input parameters to run the photochemical model over multiple computers (i.e. paralell computing).

    Parameter(s):
    rad_plan = np.array of floats
        This is the radius of the planet in units of x Earth radius.
    log10_planet_metallicity = np.array of strings
        This is the planet metallicity in units of log10 x Solar (i.e. 0.5 inputed means 10^0.5 ~ 3x Solar Metallicity)
    tint = np.array of floats
        This is the internal temperature of the planet in units of Kelvin
    semi_major = np.array of floats
        This is the orbital distance of the planet from the star in units of AU.
    ctoO = np.array of floats
        This is the carbon to oxygen ratio of the planet in units of x Solar C/O ratio.
    log_kzz = np.array of floats
        This is the eddy diffusion coefficient (the power of 10) in cm^2/s
    
    Returns:
        A tuple array of each array of input parameters to run via parallelization and return a 1D photochemical model.
    
    """

    # True Values to replace after test case:
    """
    # Convert metallicity to a list of string values
    metal_float = np.linspace(3, 3000, 10)
    metal_string = np.array([str(f) for f in metal_float])

    rad_plan_earth_units = np.linspace(1.6, 4, 5) # in units of xEarth radii
    log10_planet_metallicity = metal_string # in units of log solar metallicity
    tint_K = np.linspace(20, 400, 5) # in Kelvin
    semi_major_AU = np.linspace(0.3, 10, 10) # in AU 
    ctoO_solar = [0.01, 0.25, 0.5, 0.75, 1] # in units of solar C/O
    log_Kzz = np.array([7, 9]) # in cm^2/s 
    
    """
    """
    # Test Case:
    rad_plan_earth_units = np.array([2]) # in units of xEarth radii
    log10_planet_metallicity = np.array([3.5]) # in units of solar metallicity
    tint_K = np.array([50]) # in Kelvin
    semi_major_AU = np.array([0.3, 0.7, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]) # in AU 
    ctoO_solar = np.array([0.01]) # in units of solar C/O
    log_Kzz = np.array([5])
    
    """
    """
    
    # Parameter Exploration
    rad_plan_earth_units = np.array([2]) # in units of xEarth radii
    log10_planet_metallicity = np.array([3.5]) # in units of solar metallicity
    tint_K = np.array([50]) # in Kelvin
    semi_major_AU = np.array([0.3, 0.7, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]) # in AU 
    ctoO_solar = np.array([0.01]) # in units of solar C/O
    log_Kzz = np.array([5]) # In units of logspace (so 5 means 10^5 cm^2/s)
    
    """

    # Parameter Exploration Refined
    rad_plan_earth_units = np.array([2]) # in units of xEarth radii
    log10_planet_metallicity = np.linspace(0.5, 3.5, 9) # in units of solar metallicity
    tint_K = np.linspace(50, 400, 8) # in Kelvin
    semi_major_AU = np.array([0.3, 0.7, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]) # in AU 
    ctoO_solar = np.linspace(0.01, 1, 5) # in units of solar C/O
    log_Kzz = np.array([5, 7, 9]) # In units of logspace (so 5 means 10^5 cm^2/s)
    
    gridvals = (rad_plan_earth_units, log10_planet_metallicity, tint_K, semi_major_AU, ctoO_solar, log_Kzz)

    return gridvals
    
def Photochem_1D_model(x):

    """
    This runs Photochem_Gas_Giant on Tijuca for parallel computing.

    Parameters:
        x needs to be in the order of total flux, planet metallicity, tint, and kzz!
        total flux = units of solar (float)
        planet metallicity = units of solar but needs to be a float/integer NOT STRING
        tint = units of Kelvin (float)
        kzz = units of cm^2/s (float)

    Results:
    combined_dict: dictionary
        This gives you all the results of Photochem_Gas_Giant, except matches the length of convergence arrays with molecular abundances at steady state and renames where or not PICASO converged as "converged_TP" and whether or not Photochem converged as "converged_PC". Both are in the binary equivalent of the boolean True/False (i.e. 1/0). 
        
    """

    # For Tijuca
    rad_plan_earth_units, log10_planet_metallicity, tint_K, semi_major_AU, ctoO_solar, log_Kzz = x
    logging.info(f"starting photochem model for inputs: {x}")

    sol, soleq, pc, convergence_values, converged_PC, status, error = Photochem_Gas_Giant(rad_plan=rad_plan_earth_units, log10_planet_metallicity=log10_planet_metallicity, tint=tint_K, semi_major=semi_major_AU, ctoO=ctoO_solar, log_Kzz=log_Kzz, PT_filename='results/PICASO_climate_updatop_full_exploration_reducedrad_solveSegFault.h5')

    if status not in ('Photochem-converged', 'Photochem-not-converged'):
        # PICASO failure or Photochem exception: log the fingerprint
        _append_crash_fingerprint(
            stage='photochem',
            x=x,
            reason=status,
            details=error,
        )

    # Merge the sol & soleq & convergence arrays into a single dictionary
    modified_sol_dict = {key + "_sol": value for key, value in sol.items()}
    modified_soleq_dict = {key + "_soleq": value for key, value in soleq.items()}
    combined_dict = {**modified_sol_dict, **modified_soleq_dict}
    combined_dict['converged_TP'] = np.asarray(convergence_values, dtype=np.uint8)
    combined_dict['converged_PC'] = np.asarray(converged_PC, dtype=np.uint8)
    combined_dict['status'] = np.array([str(status)], dtype='S64')
    combined_dict['error'] = np.array([str(error)], dtype='S1024')

    return combined_dict 

if __name__ == "__main__":
    """
    To execute running 1D Photochemical model for the range of values in get_gridvals_Photochem, type the folling command into your terminal:
   
    # mpiexec -n X python Photochem_grid_121625.py
    
    """
    setup_rank_debug_logging()
    gridutils.make_grid(
        model_func=Photochem_1D_model,
        gridvals=get_gridvals_Photochem(),
        filename='results/Photochem_1D_updatop_paramext_reducedrad_full_try2.h5',
        progress_filename='results/Photochem_1D_updatop_paramext_reducedrad_full_try2.log'
    )
