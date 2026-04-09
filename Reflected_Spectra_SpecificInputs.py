import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path

current_directory = Path.cwd()
references_directory_path = "Installation&Setup_Instructions/picasofiles/reference"
PYSYN_directory_path = "Installation&Setup_Instructions/picasofiles/grp/redcat/trds"
print(os.path.join(current_directory, references_directory_path))
print(os.path.join(current_directory, PYSYN_directory_path))

os.environ['picaso_refdata'] = os.path.join(current_directory, references_directory_path)
os.environ['PYSYN_CDBS'] = os.path.join(current_directory, PYSYN_directory_path)

import picaso.justdoit as jdi
import astropy.units as u
import astropy.constants as const
import numpy as np
import pandas as pd

from mpi4py import MPI
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import h5py
import copy
import dill as pickle
import json
import logging
import socket
import sys
import faulthandler
import traceback
from tqdm import tqdm

import Photochem_grid_121625 as Photochem_grid

# Set up logging with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ---------------------------------------------------------------------------
# Worker-only initializations (rank 0 is the MPI master and never runs the
# model, so it skips the expensive opacity and Photochem loads entirely).
# Each worker loads these once at startup and reuses them for every case.
# ---------------------------------------------------------------------------
_comm = MPI.COMM_WORLD
_rank = _comm.Get_rank()

OPACITY = None
PHOTOCHEM_INPUTS   = None   # float array (N_rows, 6): [rad, metal, tint, semi, ctoO, kzz]
PHOTOCHEM_RESULTS  = None   # dict key -> ndarray with full grid shape

if _rank != 0:
    # --- Opacity ---
    opacity_file_path = "Installation&Setup_Instructions/picasofiles/reference/opacities/opacities_photochem_0.1_250.0_R15000.db"
    opacity_path = os.path.join(current_directory, opacity_file_path)
    logging.info("rank %d: loading opacity from %s", _rank, opacity_path)
    OPACITY = jdi.opannection(filename_db=opacity_path, wave_range=[0.3, 2.5])
    logging.info("rank %d: opacity loaded", _rank)

    # --- Photochem results (loaded once, kept in RAM) ---
    _photochem_file = 'results/Photochem_1D_updatop_paramext_reducedrad_full_try3.h5'
    logging.info("rank %d: loading Photochem data from %s", _rank, _photochem_file)
    with h5py.File(_photochem_file, 'r') as _f:
        PHOTOCHEM_INPUTS  = np.array(_f['inputs'])
        PHOTOCHEM_RESULTS = {key: np.array(_f['results'][key]) for key in _f['results'].keys()}
    logging.info("rank %d: Photochem data loaded (%d rows)", _rank, len(PHOTOCHEM_INPUTS))

# ---------------------------------------------------------------------------
# Planet/star physics helpers (defined locally to avoid cross-module issues)
# ---------------------------------------------------------------------------

def calc_Teq_SUN(distance_AU):
    """Equilibrium temperature (K) of a planet at distance_AU from the Sun."""
    luminosity_star = 3.846e26  # W
    boltzmann_const = 5.670374419e-8  # W m^-2 K^-4
    distance_m = distance_AU * 1.496e11
    Teq = (((distance_m**2) * (16 * np.pi * boltzmann_const) / luminosity_star) ** (1/4))**(-1)
    return Teq

def mass_from_radius_chen_kipping_2017(R_rearth):
    """
    Estimate planet mass (Earth masses) from radius (Earth radii) using Chen & Kipping (2017).
    """
    if R_rearth <= 0:
        raise ValueError("R_rearth must be > 0")
    if 11.1 <= R_rearth <= 14.3:
        raise ValueError("Chen & Kipping (2017) inversion is degenerate for 11.1 <= Rp/Re <= 14.3.")
    logR = np.log10(R_rearth)
    if R_rearth < 1.23:
        C, S = 0.00346, 0.2790
    elif R_rearth < 11.1:
        C, S = -0.0925, 0.589
    else:
        C, S = -2.85, 0.881
    logM = (logR - C) / S
    return 10.0**logM

# ---------------------------------------------------------------------------
# Photochem lookup helpers (ported from Reflected_Spectra_grid_13026.py)
# ---------------------------------------------------------------------------

def find_Photochem_match(
    rad_plan=None, log10_planet_metallicity=None, tint=None,
    semi_major=None, ctoO=None, Kzz=None,
    gridvals=Photochem_grid.get_gridvals_Photochem()
):
    """
    Find the matching Photochem solution for the given input parameters.
    Searches the in-memory PHOTOCHEM_INPUTS / PHOTOCHEM_RESULTS globals
    (loaded once at worker startup) instead of reopening the HDF5 file.

    Returns (sol_dict, soleq_dict, PT_list, convergence_PC, convergence_TP).
    All are None if no match is found.
    """
    gridvals_metal = [float(s) for s in gridvals[1]]
    gridvals_ctoO  = [float(s) for s in gridvals[4]]

    planet_metallicity = float(log10_planet_metallicity)
    input_list = np.array([rad_plan, planet_metallicity, tint, semi_major, ctoO, Kzz])

    # Search in-memory inputs array — no file I/O
    row_matches = np.all(PHOTOCHEM_INPUTS == input_list, axis=1)
    matching_indicies = np.where(row_matches)[0]

    if matching_indicies.size == 0:
        print(f'No Photochem match found for inputs: {input_list}')
        return None, None, None, None, None

    # Find per-axis indices for multi-dimensional result lookup
    gridvals_dict = {
        'rad_plan':           np.array(gridvals[0]),
        'planet_metallicity': np.array(gridvals_metal),
        'tint':               np.array(gridvals[2]),
        'semi_major':         np.array(gridvals[3]),
        'ctoO':               np.array(gridvals_ctoO),
        'Kzz':                np.array(gridvals[5]),
    }
    ri = np.where(gridvals_dict['rad_plan']           == input_list[0])[0]
    mi = np.where(gridvals_dict['planet_metallicity'] == input_list[1])[0]
    ti = np.where(gridvals_dict['tint']               == input_list[2])[0]
    si = np.where(gridvals_dict['semi_major']         == input_list[3])[0]
    ci = np.where(gridvals_dict['ctoO']               == input_list[4])[0]
    ki = np.where(gridvals_dict['Kzz']                == input_list[5])[0]

    sol_dict   = {}
    soleq_dict = {}
    for key, arr in PHOTOCHEM_RESULTS.items():
        if key.endswith('sol'):
            sol_dict[key]   = arr[ri[0]][mi[0]][ti[0]][si[0]][ci[0]][ki[0]]
        elif key.endswith('soleq'):
            soleq_dict[key] = arr[ri[0]][mi[0]][ti[0]][si[0]][ci[0]][ki[0]]

    sol_dict   = {k.removesuffix('_sol')   if k.endswith('_sol')   else k: v for k, v in sol_dict.items()}
    soleq_dict = {k.removesuffix('_soleq') if k.endswith('_soleq') else k: v for k, v in soleq_dict.items()}

    pressure_values    = PHOTOCHEM_RESULTS['pressure_sol'][ri[0]][mi[0]][ti[0]][si[0]][ci[0]][ki[0]]
    temperature_values = PHOTOCHEM_RESULTS['temperature_sol'][ri[0]][mi[0]][ti[0]][si[0]][ci[0]][ki[0]]
    convergence_PC     = PHOTOCHEM_RESULTS['converged_PC'][ri[0]][mi[0]][ti[0]][si[0]][ci[0]][ki[0]]
    convergence_TP     = PHOTOCHEM_RESULTS['converged_TP'][ri[0]][mi[0]][ti[0]][si[0]][ci[0]][ki[0]]
    PT_list = pressure_values, temperature_values

    print(f'Photochem match found for inputs: {input_list}')
    return sol_dict, soleq_dict, PT_list, convergence_PC, convergence_TP


def find_pbot(sol=None, solaer=None, tol=0.9):
    """
    Find the pressure at the bottom of the H2O cloud (in bars).
    Returns None if no water cloud is present.
    """
    pressure = sol['pressure']
    H2Oaer   = solaer['H2Oaer']

    if np.max(H2Oaer) < 1e-20:
        return None

    H2Oaer_normalized = H2Oaer / np.max(H2Oaer)
    ind = None
    for i, val in enumerate(H2Oaer_normalized):
        if val > tol:
            ind = i
            break

    if ind is None:
        raise Exception('Could not find the bottom of the water cloud.')

    return pressure[ind]


def make_picaso_atm(sol):
    """
    Convert Photochem output dict to PICASO-ready dict.
    Pressure is converted from dynes/cm^2 to bars and all arrays are flipped.
    Returns (atm_dict, aerosol_dict).
    """
    sol_dict_noaer = {}
    sol_dict_aer   = {}
    for key in sol.keys():
        if key.endswith('aer'):
            sol_dict_aer[key] = sol[key]
        else:
            sol_dict_noaer[key] = sol[key]

    atm = copy.deepcopy(sol_dict_noaer)
    atm['pressure'] /= 1e6  # dynes/cm^2 -> bars
    for key in atm:
        atm[key] = atm[key][::-1].copy()

    sol_dict_aer = copy.deepcopy(sol_dict_aer)
    for key in sol_dict_aer:
        sol_dict_aer[key] = sol_dict_aer[key][::-1].copy()

    return atm, sol_dict_aer

# ---------------------------------------------------------------------------
# Core reflected-spectrum calculation
# ---------------------------------------------------------------------------

def reflected_spectrum_planet_Sun(
    rad_plan=None, planet_metal=None, tint=None, semi_major=None,
    ctoO=None, Kzz=None, phase_angle=None,
    atmosphere_kwargs={}
):
    """
    Calculate the reflected spectrum of a mini-Neptune around a Sun-like star,
    using composition from the Photochem grid.

    Photochem data is read from the PHOTOCHEM_INPUTS / PHOTOCHEM_RESULTS globals
    loaded once at worker startup — no file I/O per call.

    Parameters
    ----------
    rad_plan : float
        Planet radius in Earth radii.
    planet_metal : float
        log10(metallicity / Solar).
    tint : float
        Internal temperature (K).
    semi_major : float
        Semi-major axis (AU).
    ctoO : float
        C/O ratio in Solar units.
    Kzz : float
        log10(eddy diffusion coefficient / cm^2 s^-1).
    phase_angle : float
        Phase angle in radians.
    atmosphere_kwargs : dict
        Optional; pass {'exclude_mol': ['O2']} to zero out a species.

    Returns
    -------
    wno, fpfs, alb, clouds : np.ndarray (each length ~150)
    """
    opacity = OPACITY
    planet_metal = float(planet_metal)

    start_case = jdi.inputs()

    # Planet physical properties
    planet_radius = rad_plan * 6.371e6 * u.m
    planet_mass_earth = mass_from_radius_chen_kipping_2017(R_rearth=rad_plan)
    planet_mass = planet_mass_earth * 5.972e24  # kg
    planet_grav = (const.G.value * planet_mass) / (planet_radius.value**2)  # m/s^2

    # Star parameters (Sun)
    stellar_Teff  = 5778   # K
    stellar_logg  = 4.4    # log10(g / cgs)
    stellar_metal = 0.0    # log10(Z/Solar)
    stellar_radius = 1.0   # Solar radii

    start_case.phase_angle(phase_angle, num_tangle=8, num_gangle=8)

    jupiter_mass   = const.M_jup.value          # kg
    jupiter_radius = 69911e3                    # m
    start_case.gravity(
        gravity=planet_grav, gravity_unit=jdi.u.Unit('m/(s**2)'),
        radius=planet_radius.value / jupiter_radius, radius_unit=jdi.u.Unit('R_jup'),
        mass=planet_mass / jupiter_mass, mass_unit=jdi.u.Unit('M_jup')
    )
    start_case.star(
        opannection=opacity,
        temp=stellar_Teff, logg=stellar_logg,
        semi_major=semi_major, metal=stellar_metal,
        radius=stellar_radius, radius_unit=jdi.u.R_sun,
        semi_major_unit=jdi.u.au
    )

    # Look up Photochem composition (searches in-memory globals, no file I/O)
    sol_dict, soleq_dict, PT_list, convergence_PC, convergence_TP = find_Photochem_match(
        rad_plan=rad_plan, log10_planet_metallicity=planet_metal,
        tint=tint, semi_major=semi_major, ctoO=ctoO, Kzz=Kzz
    )

    if sol_dict is None:
        raise ValueError(
            f'No Photochem match for rad={rad_plan}, metal={planet_metal}, '
            f'tint={tint}, semi_major={semi_major}, ctoO={ctoO}, Kzz={Kzz}'
        )

    atm, sol_dict_aer = make_picaso_atm(sol_dict)
    df_atmo = jdi.pd.DataFrame(atm)

    if 'exclude_mol' in atmosphere_kwargs:
        sp = atmosphere_kwargs['exclude_mol'][0]
        if sp in df_atmo:
            df_atmo[sp] *= 0

    start_case.atmosphere(df=df_atmo)
    df_cldfree = start_case.spectrum(opacity, calculation='reflected', full_output=True)
    wno_cf = df_cldfree['wavenumber']
    alb_cf = df_cldfree['albedo']
    fpfs_cf = df_cldfree['fpfs_reflected']
    _, alb_cf_grid  = jdi.mean_regrid(wno_cf, alb_cf,  R=150)
    wno_cf_grid, fpfs_cf_grid = jdi.mean_regrid(wno_cf, fpfs_cf, R=150)

    # Optionally add H2O water cloud
    if 'H2Oaer' in sol_dict_aer:
        pbot = find_pbot(sol=atm, solaer=sol_dict_aer)
        if pbot is not None:
            logpbot = np.log10(pbot)
            ptop_earth, pbot_earth = 0.6, 0.7
            logdp = np.log10(pbot_earth) - np.log10(ptop_earth)
            start_case.clouds(w0=[0.99], g0=[0.85], p=[logpbot], dp=[logdp], opd=[10])
            df_cld = start_case.spectrum(opacity, full_output=True)
            wno_c   = df_cld['wavenumber']
            alb_c   = df_cld['albedo']
            fpfs_c  = df_cld['fpfs_reflected']
            _, alb  = jdi.mean_regrid(wno_cf, 0.5*alb_cf  + 0.5*alb_c,  R=150)
            wno, fpfs = jdi.mean_regrid(wno_cf, 0.5*fpfs_cf + 0.5*fpfs_c, R=150)
            clouds = np.ones(len(wno))
            return wno, fpfs, alb, clouds

    wno   = wno_cf_grid.copy()
    alb   = alb_cf_grid.copy()
    fpfs  = fpfs_cf_grid.copy()
    clouds = np.zeros(len(wno))
    return wno, fpfs, alb, clouds


def make_case_RSM(rad_plan=None, planet_metal=None, tint=None, semi_major=None,
                  ctoO=None, Kzz=None, phase_angle=None):
    """
    Thin wrapper around reflected_spectrum_planet_Sun that returns a result dict.
    """
    wno, fpfs, albedo, clouds = reflected_spectrum_planet_Sun(
        rad_plan=rad_plan, planet_metal=planet_metal, tint=tint,
        semi_major=semi_major, ctoO=ctoO, Kzz=Kzz,
        phase_angle=phase_angle
    )
    return {'wno': wno, 'fpfs': fpfs, 'albedo': albedo, 'clouds': clouds}

# ---------------------------------------------------------------------------
# Specific case loading (replaces get_gridvals_RSM / gridutils.make_grid)
# ---------------------------------------------------------------------------

def _resolve_case_file(case_ref):
    """Search common locations for a case-list file and return its path."""
    case_ref = str(case_ref).strip()
    candidates = [
        case_ref,
        f"{case_ref}.npy",
        os.path.join("results", case_ref),
        os.path.join("results", f"{case_ref}.npy"),
        os.path.join(str(Path.cwd()), case_ref),
        os.path.join(str(Path.cwd()), f"{case_ref}.npy"),
        os.path.join(str(Path.cwd()), "results", case_ref),
        os.path.join(str(Path.cwd()), "results", f"{case_ref}.npy"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            logging.info("Resolved case-list file: %s", p)
            return p
    raise FileNotFoundError(
        f"Could not find case-list file '{case_ref}'.\nSearched:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


def _load_cases_from_file(case_ref):
    """
    Load specific cases from a .npy, .json, or .csv file.

    Expected column order (7 columns):
        [rad, metal, tint, semi_major, ctoO, kzz, phase_angle]

    Returns float array of shape (N, 7).
    """
    case_file = _resolve_case_file(case_ref)
    ext = os.path.splitext(case_file)[1].lower()

    if ext == ".npy":
        arr = np.load(case_file, allow_pickle=True)
    elif ext == ".json":
        with open(case_file, "r", encoding="utf-8") as fp:
            arr = np.array(json.load(fp), dtype=float)
    elif ext == ".csv":
        arr = np.loadtxt(case_file, delimiter=",", dtype=float, comments='#')
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .npy, .json, or .csv.")

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(
            f"Expected shape (N, 7), got {arr.shape}.\n"
            "Column order: [rad, metal, tint, semi_major, ctoO, kzz, phase_angle]"
        )

    logging.info("Loaded case list: %s  shape=%s", case_file, arr.shape)
    print(f"Loaded case list: {case_file}  ({arr.shape[0]} cases)")
    return arr


def get_specific_inputs_RSM():
    """
    Return the array of specific input cases to run.

    Each row is:
        [rad (R_earth), metal (log10 solar), tint (K), semi_major (AU),
         ctoO (x solar), kzz (log10 cm^2/s), phase_angle (radians)]

    To run specific cases, set the environment variable RSM_CASE_LIST to the
    path of a .npy, .json, or .csv file with columns in the order above.

    If RSM_CASE_LIST is not set, the default test cases below are used.

    To generate a case file from a set of base parameters at all phase angles,
    use something like:

        import numpy as np
        base = np.array([[2.61, 3.5, 155, 1, 0.01, 5]])  # shape (N, 6)
        phases = np.linspace(0, np.pi, 19)[:-1]           # 18 phase angles
        cases = np.array([[*row, phi] for row in base for phi in phases])
        np.save('my_cases.npy', cases)
    """
    case_ref = os.environ.get("RSM_CASE_LIST", "").strip()
    if case_ref:
        return _load_cases_from_file(case_ref)

    # Default: K2-18b-like planet at 18 phase angles (0 to 170 degrees)
    phase_angles = np.linspace(0, np.pi, 19)[:-1]
    base_params = [2, 3.5, 155, 1.0, 0.01, 5]   # rad, metal, tint, semi_major, ctoO, kzz
    cases = np.array([[*base_params, phi] for phi in phase_angles])
    print(f"No RSM_CASE_LIST set — using default test cases ({len(cases)} cases).")
    return cases

# ---------------------------------------------------------------------------
# Flat-list parallel infrastructure (replaces gridutils.make_grid)
# ---------------------------------------------------------------------------

def _initialize_hdf5_specific(filename, N, n_params=7):
    """Create the HDF5 file skeleton for N specific cases."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('inputs',    shape=(N, n_params), dtype=float)
        f['inputs'][:] = np.nan
        f.create_dataset('completed', shape=(N,), dtype='bool')
        f['completed'][:] = False
        f.create_group('results')


def _save_result_specific(filename, index, x, res, N):
    """Save one completed case to the HDF5 file."""
    with h5py.File(filename, 'a') as f:
        # inputs
        if 'inputs' not in f:
            f.create_dataset('inputs', shape=(N, len(x)), dtype=float)
            f['inputs'][:] = np.nan
        f['inputs'][index] = x

        # results
        if 'results' not in f:
            f.create_group('results')

        for key, val in res.items():
            val = np.atleast_1d(np.array(val))
            if val.dtype.kind == 'U':
                val = val.astype('S')
            length = len(val)
            if key not in f['results']:
                f['results'].create_dataset(
                    key, shape=(N, length),
                    maxshape=(N, None),
                    dtype=val.dtype
                )
            elif f['results'][key].shape[1] < length:
                f['results'][key].resize((N, length))
            f['results'][key][index] = val

        # completed flag
        if 'completed' not in f:
            f.create_dataset('completed', shape=(N,), dtype='bool')
            f['completed'][:] = False
        f['completed'][index] = True


def _load_completed_specific(filename):
    """Return array of already-completed case indices."""
    if os.path.isfile(filename):
        with h5py.File(filename, 'r') as f:
            if 'completed' in f:
                return np.where(f['completed'][:])[0]
    return np.array([], dtype=int)


def _assign_job(comm, rank, serialized_model, job_iter, cases):
    try:
        i = next(job_iter)
        comm.send((serialized_model, i, cases[i]), dest=rank, tag=1)
        return True
    except StopIteration:
        comm.send(None, dest=rank, tag=0)
        return False


def _master(model_func, cases, filename, progress_filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    N = len(cases)

    serialized_model = pickle.dumps(model_func)

    if not os.path.exists(filename):
        print(f"Initializing HDF5 output for {N} cases...")
        _initialize_hdf5_specific(filename, N, n_params=cases.shape[1])

    completed = _load_completed_specific(filename)
    job_indices = [i for i in range(N) if i not in completed]
    job_iter = iter(job_indices)

    print(f"Cases total: {N} | Completed: {len(completed)} | Remaining: {len(job_indices)}")

    with open(progress_filename, 'w') as log_file:
        pbar = tqdm(total=len(job_indices), file=log_file, dynamic_ncols=True)
        status = MPI.Status()

        active_workers = 0
        for rank in range(1, size):
            if _assign_job(comm, rank, serialized_model, job_iter, cases):
                active_workers += 1

        while active_workers > 0:
            index, x, res = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
            worker_rank = status.Get_source()
            _save_result_specific(filename, index, x, res, N)
            pbar.update(1)
            log_file.flush()
            if not _assign_job(comm, worker_rank, serialized_model, job_iter, cases):
                active_workers -= 1

        pbar.close()


def _worker():
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    n_wno = 150

    while True:
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == 0:
            break  # shutdown signal

        serialized_model, index, x = data
        model_func = pickle.loads(serialized_model)
        try:
            res = model_func(x)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            res = {
                'wno':    np.full(n_wno, np.nan),
                'fpfs':   np.full(n_wno, np.nan),
                'albedo': np.full(n_wno, np.nan),
                'clouds': np.zeros(n_wno),
                'status': np.array([f'worker_exception: {err}'], dtype='S'),
            }

        comm.send((index, x, res), dest=0, tag=2)


def make_specific_grid(model_func, cases, filename, progress_filename):
    """
    Run a flat-list parallel computation using MPI, saving results to an HDF5 file.

    Unlike gridutils.make_grid (which meshgrids parameter axes into all combinations),
    this function runs each row of `cases` as one independent model call, allowing
    arbitrary non-grid combinations of input parameters.

    The output HDF5 file has the structure:
        /inputs       float (N, 7)  — [rad, metal, tint, semi_major, ctoO, kzz, phase_angle]
        /completed    bool  (N,)
        /results/
            wno       float (N, ~150)
            fpfs      float (N, ~150)
            albedo    float (N, ~150)
            clouds    float (N, ~150)
            status    bytes (N, 1)  — 'ok', 'no_photochem_match', or error message

    Parameters
    ----------
    model_func : callable
        Takes a 1D array x of shape (7,) and returns a dict of numpy arrays.
    cases : np.ndarray, shape (N, 7)
        Each row is one set of inputs [rad, metal, tint, semi_major, ctoO, kzz, phase_angle].
    filename : str
        Output HDF5 file path.
    progress_filename : str
        Path for the tqdm progress log.

    Notes
    -----
    Must be launched with an MPI runner, e.g.:
        mpiexec -n 40 python Reflected_Spectra_SpecificInputs.py
    Interrupted runs resume automatically by skipping already-completed indices.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print(f"make_specific_grid: {len(cases)} cases, saving to '{filename}'")
        _master(model_func, cases, filename, progress_filename)
    else:
        _worker()

# ---------------------------------------------------------------------------
# Model function (called by each MPI worker for one case row)
# ---------------------------------------------------------------------------

def Reflected_Spectra_model_specific(x):
    """
    Compute the reflected spectrum for one specific set of input parameters.

    Parameters
    ----------
    x : array-like, shape (7,)
        [rad (R_earth), metal (log10 Solar), tint (K), semi_major (AU),
         ctoO (x Solar), kzz (log10 cm^2/s), phase_angle (radians)]

    Returns
    -------
    dict with keys: wno, fpfs, albedo, clouds, status (all np.ndarray)
    """
    rad, metal, tint, semi_major, ctoO, kzz, phase_angle = [float(v) for v in x]
    n_wno = 150

    def _error_res(status_str):
        return {
            'wno':    np.full(n_wno, np.nan),
            'fpfs':   np.full(n_wno, np.nan),
            'albedo': np.full(n_wno, np.nan),
            'clouds': np.zeros(n_wno),
            'status': np.array([status_str], dtype='S'),
        }

    logging.info("Starting RSM for inputs: rad=%.3f, metal=%.2f, tint=%.1f, "
                 "semi_major=%.3f, ctoO=%.4f, kzz=%.1f, phase=%.4f",
                 rad, metal, tint, semi_major, ctoO, kzz, phase_angle)

    try:
        res = make_case_RSM(
            rad_plan=rad,
            planet_metal=metal,
            tint=tint,
            semi_major=semi_major,
            ctoO=ctoO,
            Kzz=kzz,
            phase_angle=phase_angle,
        )
        out = {
            'wno':    np.array(res['wno']),
            'fpfs':   np.array(res['fpfs']),
            'albedo': np.array(res['albedo']),
            'clouds': np.array(res['clouds']),
            'status': np.array(['ok'], dtype='S'),
        }
        return out

    except ValueError as e:
        # Likely a missing Photochem match
        msg = str(e)
        logging.warning("No Photochem match: %s", msg)
        return _error_res(f'no_photochem_match: {msg[:120]}')

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logging.error("RSM failed: %s", msg)
        return _error_res(f'error: {msg[:120]}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run the reflected spectra model for a list of specific input cases in parallel.

    Usage on UW Hyak (via SBATCH):
        mpiexec -n <N_TASKS> python Reflected_Spectra_SpecificInputs.py

    To specify a custom case list, set the RSM_CASE_LIST environment variable
    before launching (see get_specific_inputs_RSM for file format details).

    To run the default K2-18b test cases:
        mpiexec -n 4 python Reflected_Spectra_SpecificInputs.py
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        from mpi4py import MPI as _MPI
        import faulthandler as _fh
        _fh.enable(all_threads=True)

    cases = get_specific_inputs_RSM()

    make_specific_grid(
        model_func=Reflected_Spectra_model_specific,
        cases=cases,
        filename='results/ReflectedSpectra_SpecificInputsModernEarth_fv.h5',
        progress_filename='results/ReflectedSpectra_SpecificInputsModernEarth_fv.log',
    )
