import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy
import pandas as pd
from scipy import optimize
from matplotlib import pyplot as plt
import pickle
from itertools import cycle
import matplotlib.colors as mcolors
import astropy.units as u
import astropy.constants as const


from photochem.utils import stars
import PICASO_Climate_grid_121625 as picaso_grid
import Photochem_grid_121625 as Photochem_grid
import Reflected_Spectra_grid_13026 as Reflected_Spectra
from picaso.photochem import EquilibriumChemistry
import GraphsKey

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


def find_pbot(sol=None, solaer=None, tol=0.9):

    """
    Parameters:
    pressures: ndarray
        Pressure at each atmospheric layer in dynes/cm^2
    H2Oaer: ndarray
        Mixing ratio of H2O aerosols.
    tol: float, optional
        The threshold value for which we define the beginning of the cloud,
        by default 1e-4.

    Returns:
    P_bottom: float
        The cloud bottom pressure in dynes/cm^2

    """

    pressure = sol['pressure']
    H2Oaer = solaer['H2Oaer']

    # There is no water cloud in the model, so we return None
    # For the cloud bottom of pressure

    if np.max(H2Oaer) < 1e-20:
        return None

    # Normalize so that max value is 1
    H2Oaer_normalized = H2Oaer/np.max(H2Oaer)

    # loop from bottom to top of atmosphere, cloud bottom pressure
    # defined as the index level where the normalized cloud mixing ratio
    # exeeds tol .

    ind = None

    for i, val in enumerate(H2Oaer_normalized):
        if val > tol:
            ind = i
            break

    if ind is None:
        raise Exception('A problem happened when trying to find the bottom of the cloud.')

    # Bottom of the cloud
    pbot = pressure[ind]

    return pbot


def _add_cloud_deck(case, ptop_bar, pbot_bar, w0=0.99, g0=0.85, opd=10):
    """Add a grey cloud deck between ptop_bar and pbot_bar (pressures in bars)."""
    logdp = np.log10(pbot_bar) - np.log10(ptop_bar)
    case.clouds(w0=[w0], g0=[g0], p=[np.log10(pbot_bar)], dp=[logdp], opd=[opd])


# Calculates the reflected spectrum of the mini-Neptune planet but allows you to adjust the cloud fraction, the abundance of certain molecules (combine atmosphere_kwargs w/ kwarg_factor variables)

def reflected_spectrum_gas_planet_Sun(rad_plan=None, planet_metal=None, tint=None, semi_major=None, ctoO=None, Kzz=None, phase_angle=None, sol_path=None, soleq_path=None, Photochem_file=False, atmosphere_kwargs={}, kwarg_factor=0,  outputfile=None, forced_nocld=False, cloud_frac=0.5, no_rayleigh=False, surface_albedo=None, plot_pt=False, exclude_all_gas=False):

    """
    This finds the reflected spectra of a planet similar to K218b around a Sun.
start_case.inputs['atmosphere']['exclude_mol'] = {'CH4': 0}
    Parameters:
    rad_plan: float
        This is the radius of the planet in units of Earth radii.
    planet_metal: float
        This is the planet's metallicity in units of log10 x Solar metallicity.
    tint: float
        This is the planet's internal temperature in Kelvin.
    semi_major: float
        This is the semi major axis of your planet's orbit in units of AU.
    ctoO: float
        This is the carbon to oxygen ratio of your planet's atmosphere in units of xSolar c/o.
    Kzz: float
        This is the eddy diffusion coefficient in logspace (i.e. the power of 10) in cm/s^2.
    phase_angle: float
        This is the phase of orbit the planet is in relative to its star and the observer (i.e. how illuminated it is), units of radians.
    Photochem_file: string
        This is the path to the Photochem grid you would like to pull composition information from.
    atmosphere_kwargs: dict 'exclude_mol': value where value is a string
        If left empty, all molecules are included, but can limit how many molecules are calculated.

    Results: IDK for sure though
    wno: grid of 150 points
        ???
    fpfs: grid of 150 points
        This is the relative flux of the planet and star (fp/fs).
    alb: grid of 150 points
        ???
    np.array(clouds): grid of 150 points
        This is a grid of whether or not a cloud was used to make the reflective spectra using the binary equivalent to booleans (True=1, False=0).

    """

    # Create and empty dictionary for later results

    current_directory = Path.cwd()
    opacity_file_path = "Installation&Setup_Instructions/picasofiles/reference/opacities/opacities_photochem_0.1_250.0_R15000.db"
    opacity_path=os.path.join(current_directory, opacity_file_path)
    print(opacity_path)
    opacity = jdi.opannection(filename_db=opacity_path, wave_range=[0.1,2.5])

    planet_metal = float(planet_metal)

    start_case = jdi.inputs()

    # Then calculate the composition from the TP profile
    class planet:

        planet_radius = (rad_plan*6.371e+6*u.m) # in meters
        planet_mass = picaso_grid.mass_from_radius_chen_kipping_2017(R_rearth=rad_plan)*(5.972e+24) # in kg
        planet_Teq = picaso_grid.calc_Teq_SUN(distance_AU=semi_major) # Equilibrium temp (K)
        planet_grav = (const.G * (planet_mass)) / ((planet_radius)**2) # of K2-18b in m/s^2
        planet_ctoO = ctoO # in xSolar

    class Sun:

        stellar_radius = 1 # Solar radii
        stellar_Teff = 5778 # K
        stellar_metal = 0.0 # log10(metallicity)
        stellar_logg = 4.4 # log10(gravity), in cgs units

    solar_zenith_angle = 60 # Used in Tsai et al. (2023)

    # Star and Planet Parameters (Stay the Same & Should Match Photochem & PICASO)
    start_case.phase_angle(phase_angle, num_tangle=8, num_gangle=8) #radians, using default here

    jupiter_mass = const.M_jup.value # in kg
    jupiter_radius = 69911e+3 # in m
    start_case.gravity(gravity=planet.planet_grav, gravity_unit=jdi.u.Unit('m/(s**2)'), radius=(planet.planet_radius.value)/jupiter_radius, radius_unit=jdi.u.Unit('R_jup'), mass=(planet.planet_mass)/jupiter_mass, mass_unit=jdi.u.Unit('M_jup'))

    # star temperature, metallicity, gravity, and opacity (default opacity is opacity.db in the reference folder)
    start_case.star(opannection=opacity, temp=Sun.stellar_Teff, logg=Sun.stellar_logg, semi_major=semi_major, metal=Sun.stellar_metal, radius=Sun.stellar_radius, radius_unit=jdi.u.R_sun, semi_major_unit=jdi.u.au)

    # Match Photochemical Files
    if Photochem_file is not True:
        sol_dict, soleq_dict, PT_list, convergence_PC, convergence_TP = Reflected_Spectra.find_Photochem_match(filename=Photochem_file, rad_plan=rad_plan, log10_planet_metallicity=planet_metal, tint=tint, semi_major=semi_major, ctoO=ctoO,Kzz=Kzz)

    elif Photochem_file is True:
        with open(sol_path, 'rb') as file:
            sol_dict = pickle.load(file)
        with open(soleq_path, 'rb') as file:
            soleq_dict = pickle.load(file)


    # Determine Planet Atmosphere & Composition

    atm, sol_dict_aer = Reflected_Spectra.make_picaso_atm(sol_dict) # Converted Pressure of Photochem, in dynes/cm^2, back to bars and flip all arrays before placing into PICASO
    print(type(atm['pressure']))
    print(f'Length of pressure vs other element: {len(atm['pressure'])} vs {len(atm['He'])}')

    # Limit atmosphere to pressures 1000 bars and below.

    atm_filtered = {}

    # Threshold value
    threshold = 1000

    # Filter the list for a specific key (e.g., 'key1')
    # Convert list to NumPy array for efficient filtering
    arr = np.array(atm['temperature'])

    # Use boolean indexing to filter values less than the threshold
    filtered_arr = arr[arr < threshold]

    # Update the dictionary (optional: convert back to list)
    atm_filtered['temperature'] = filtered_arr # or keep as a NumPy array: data['key1'] = filtered_arr
    atm_filtered['pressure'] = atm['pressure'][:len(atm_filtered['temperature'])] # filter temperature by index

   # Define keys to ignore
    exclude = {'pressure', 'temperature'}

    for key, value in atm.items():
        if key not in exclude:
            atm_filtered[key] = value[:len(atm_filtered['temperature'])]
            #print(f'updated with {key}: {atm_filtered}')


    #print(f"Original dictionary: {atm.keys()}")
    #print(f"Filtered 'pressure' values (below {threshold}): {len(atm_filtered['pressure'])}, then temperature should be the same length as new pressure: {len(atm_filtered['temperature'])}")
    #print(f"Quick check with PT association of old and new dictionaries: filtered pres: {atm_filtered['pressure'][5]}, filtered temp: {atm_filtered['temperature'][5]}, original pres: {atm['pressure'][5]}, original temp: {atm['temperature'][5]}")

    df_atmo = jdi.pd.DataFrame(atm_filtered)

    if plot_pt:
        plt.gca().invert_yaxis()
        plt.semilogy(df_atmo['temperature'], df_atmo['pressure'], ls='-', c='red')
        plt.semilogy(atm['temperature'], atm['pressure'], ls='--', c='blue')
        plt.title("Filtered PT Profile")
        plt.xlabel("Temperature in K")
        plt.ylabel("Pressure in bars")
        plt.legend()
        plt.show()

    if 'exclude_mol' in atmosphere_kwargs:
        for sp in atmosphere_kwargs['exclude_mol']:
            print(f'This should show the species you are excluding: {sp}')

            if sp in df_atmo:
                df_atmo[sp] *= kwarg_factor
                print(df_atmo[sp])

    if exclude_all_gas:
        gas_names = [col for col in df_atmo.columns if col not in ['pressure', 'temperature']]
        start_case.atmosphere(df=df_atmo, exclude_mol=gas_names)
    else:
        start_case.atmosphere(df=df_atmo)

    if surface_albedo is not None:
        start_case.surface_reflect(surface_albedo, opacity.wno)

    if no_rayleigh:
        for mol in opacity.rayleigh_opa:
            opacity.rayleigh_opa[mol] = np.zeros_like(opacity.rayleigh_opa[mol])

    df_cldfree = start_case.spectrum(opacity, calculation='reflected', full_output=True)
    wno_cldfree, alb_cldfree, fpfs_cldfree = df_cldfree['wavenumber'], df_cldfree['albedo'], df_cldfree['fpfs_reflected']
    _, alb_cldfree_grid = jdi.mean_regrid(wno_cldfree, alb_cldfree, R=150)
    wno_cldfree_grid, fpfs_cldfree_grid = jdi.mean_regrid(wno_cldfree, fpfs_cldfree, R=150)

    print(f'This is the length of the grids created: {len(wno_cldfree_grid)}, {len(fpfs_cldfree_grid)}')

    # Build pickle filename suffix from active options
    if outputfile is not None:
        _suffix = ''
        if exclude_all_gas:
            _suffix += '_nogas'
        elif 'exclude_mol' in atmosphere_kwargs and kwarg_factor == 0:
            _suffix += '_no' + ''.join(atmosphere_kwargs['exclude_mol'])
        if forced_nocld:
            _suffix += '_nocld'
        else:
            _suffix += f'_cld{cloud_frac}'
        if no_rayleigh:
            _suffix += '_noray'
        if surface_albedo is not None:
            _suffix += f'_surf{surface_albedo}'
        _out = Path(outputfile)
        pkl_name = str(_out.parent / f'RLS_{_out.name}{_suffix}')
        Path(pkl_name).parent.mkdir(parents=True, exist_ok=True)

    # Determine Whether to Add Clouds or Not?

    if "H2Oaer" in sol_dict_aer and forced_nocld == False:
        # What if we added Grey Earth-like Clouds?

        # Calculate pbot:
        pbot = find_pbot(sol = atm, solaer=sol_dict_aer)

        if pbot is not None:
            print(f'pbot was calculated, there is H2Oaer and a cloud was implemented')
            logpbot = np.log10(pbot)

            # Calculate logdp:
            ptop_earth = 0.6
            pbot_earth = 0.7
            logdp = np.log10(pbot_earth) - np.log10(ptop_earth)

            # Default opd (optical depth), w0 (single scattering albedo), g0 (asymmetry parameter)
            start_case.clouds(w0=[0.99], g0=[0.85],
                              p = [logpbot], dp = [logdp], opd=[10])
            # Cloud spectrum
            df_cld = start_case.spectrum(opacity,full_output=True)

            # Average the two spectra - This differs between Calculating Earth Reflected Spectra
            wno_c, alb_c, fpfs_c, albedo_c = df_cld['wavenumber'],df_cld['albedo'],df_cld['fpfs_reflected'], df_cld['albedo']
            _, alb = jdi.mean_regrid(wno_cldfree, (1 -cloud_frac)*alb_cldfree+cloud_frac*albedo_c,R=150)
            wno, fpfs = jdi.mean_regrid(wno_cldfree, (1 -cloud_frac)*fpfs_cldfree+cloud_frac*fpfs_c,R=150)

            # Match the length of the clouds array with the length of wno or alb (fpfs is different length)
            clouds = [1] * len(wno)


            if outputfile == None:
                return wno, fpfs, alb, np.array(clouds), df_cld, df_cldfree
            else:
                RSM_outputs = {"wno":wno,
                               "fpfs":fpfs,
                               "alb":alb,
                               "clouds":np.array(clouds),
                               "df_cld":df_cld,
                               "df_cldfree":df_cldfree}

                with open(f'{pkl_name}.pkl', 'wb') as f:
                    pickle.dump(RSM_outputs, f)
                return wno, fpfs, alb, np.array(clouds), df_cld, df_cldfree

        else:
            print(f'pbot is empty and/or forced_nocld is False, so no cloud is implemented')
            wno = wno_cldfree_grid.copy()
            alb = alb_cldfree_grid.copy()
            fpfs = fpfs_cldfree_grid.copy()

            # Match the length of the clouds array with the length of wno or alb (fpfs is different length)
            clouds = [0] * len(wno)

            print(f'This is the length of the values I want to save: wno {len(wno)}, alb {len(alb)}, fpfs {len(fpfs)}, clouds {len(clouds)}')

            if outputfile == None:
                return wno, fpfs, alb, np.array(clouds), df_cld, df_cldfree
            else:
                RSM_outputs = {"wno":wno,
                               "fpfs":fpfs,
                               "alb":alb,
                               "clouds":np.array(clouds),
                               "df_cld":df_cld,
                               "df_cldfree":df_cldfree}

                with open(f'{pkl_name}.pkl', 'wb') as f:
                    pickle.dump(RSM_outputs, f)
                return wno, fpfs, alb, np.array(clouds), df_cld, df_cldfree

    else:
        if forced_nocld:
            print(f'forced_nocld=True, skipping cloud calculation')
        else:
            print(f'H2Oaer is not in solutions, no cloud implemented')
        wno = wno_cldfree_grid.copy()
        alb = alb_cldfree_grid.copy()
        fpfs = fpfs_cldfree_grid.copy()
        print(f'For the inputs: {rad_plan}, {planet_metal}, {tint}, {semi_major}, {ctoO}, {Kzz}, {phase_angle}, The length should match: wno - {len(wno)}, alb - {len(alb)}, fpfs - {len(fpfs)}')

        # Match the length of the clouds array with the length of wno or alb (fpfs is different length)
        clouds = [0] * len(wno) # This means that there are no clouds

        df_cld = None

        if outputfile == None:
                return wno, fpfs, alb, np.array(clouds), df_cld, df_cldfree
        else:
            RSM_outputs = {"wno":wno,
                           "fpfs":fpfs,
                           "alb":alb,
                           "clouds":np.array(clouds),
                           "df_cld":df_cld,
                           "df_cldfree":df_cldfree}

            with open(f'{pkl_name}.pkl', 'wb') as f:
                pickle.dump(RSM_outputs, f)

            return wno, fpfs, alb, np.array(clouds), df_cld, df_cldfree


# Default Modern and Archean Earth Mixing Ratio Compositions
df_mol_archean_earth = {
        "N2":0.945,
        "CO2":0.05,
        "CO":0.0005,
        "CH4":0.005,
        "H2O":0.003
    }

df_mol_modern_earth = {
        "N2":0.79,
        "O2":0.21,
        "O3":7e-7,
        "H2O":3e-3,
        "CO2":300e-6,
        "CH4":1.7e-6
    }


def earth_spectrum(
    opacity_path=None,
    df_mol_earth=None,
    phase=0.0,
    atmosphere_kwargs=None,
    cloud_frac=0.5,
    p_surface_bar=1.0,
    nlevel=90,
    cloud_ptop_bar=0.6,
    cloud_pbot_bar=0.7,
    no_rayleigh=False,
    outputfile=None,
    exclude_all_gas=False,
):
    """
    Calculates Earth reflected spectrum around the Sun.

    Parameters:
    opacity_path: str
        Path to the opacity file.
    df_mol_earth: dict or None
        Molecular mixing ratios. If None, defaults to modern Earth composition.
    phase: float
        Phase angle in radians.
    atmosphere_kwargs: dict or None
        Optional. Keys: 'exclude_mol' — list of molecule names to zero out.
        All listed species are zeroed out (loops over the full list).
    cloud_frac: float
        Cloud fraction (0–1) for weighted average of cloudy/cloud-free spectra.
    p_surface_bar: float
        Surface pressure in bars. Default 1.0 matches modern Earth.
        Increase to simulate a deeper/thicker atmosphere (e.g. gas planet analogue).
    nlevel: int
        Number of atmospheric pressure levels.
    cloud_ptop_bar: float
        Cloud top pressure in bars.
    cloud_pbot_bar: float
        Cloud bottom pressure in bars.

    Returns:
    wno, fpfs, albedo, df_cld, df_cldfree
    """
    if atmosphere_kwargs is None:
        atmosphere_kwargs = {}

    earth = jdi.inputs()
    earth.phase_angle(phase, num_tangle=8, num_gangle=8)
    earth.gravity(radius=1, radius_unit=jdi.u.Unit('R_earth'),
                  mass=1, mass_unit=jdi.u.Unit('M_earth'))
    earth.approx(raman="none")

    opacity = jdi.opannection(filename_db=opacity_path, wave_range=[0.3, 2.5])
    earth.star(opannection=opacity, temp=5778, logg=4.4, semi_major=1,
               metal=0.0, semi_major_unit=u.Unit('au'))

    P = np.logspace(-6, np.log10(p_surface_bar), nlevel)
    df_atmo = earth.TP_line_earth(P, nlevel=nlevel)
    df_pt_earth = pd.DataFrame({
        'pressure': df_atmo['pressure'].values,
        'temperature': df_atmo['temperature'].values,
    })

    if df_mol_earth is None:
        df_mol_earth = {
            "N2": 0.79, "O2": 0.21, "O3": 7e-7,
            "H2O": 3e-3, "CO2": 300e-6, "CH4": 1.7e-6,
        }

    df_mol_grid = pd.DataFrame({key: P * 0 + val for key, val in df_mol_earth.items()})
    df_atmo_earth = df_pt_earth.join(df_mol_grid, how='inner')

    if 'exclude_mol' in atmosphere_kwargs:
        for sp in atmosphere_kwargs['exclude_mol']:
            if sp in df_atmo_earth:
                df_atmo_earth[sp] *= 0

    if exclude_all_gas:
        gas_names = [col for col in df_atmo_earth.columns if col not in ['pressure', 'temperature']]
        earth.atmosphere(df=df_atmo_earth, exclude_mol=gas_names)
    else:
        earth.atmosphere(df=df_atmo_earth)
    earth.surface_reflect(0.1, opacity.wno)

    if no_rayleigh:
        for mol in opacity.rayleigh_opa:
            opacity.rayleigh_opa[mol] = np.zeros_like(opacity.rayleigh_opa[mol])

    df_cldfree = earth.spectrum(opacity, calculation='reflected', full_output=True)

    _add_cloud_deck(earth, cloud_ptop_bar, cloud_pbot_bar)
    df_cld = earth.spectrum(opacity, full_output=True)

    wno = df_cldfree['wavenumber']
    fpfs_cf = df_cldfree['fpfs_reflected']
    albedo_cf = df_cldfree['albedo']
    fpfs_c = df_cld['fpfs_reflected']
    albedo_c = df_cld['albedo']

    _, albedo = jdi.mean_regrid(wno, (1 - cloud_frac) * albedo_cf + cloud_frac * albedo_c, R=150)
    wno, fpfs = jdi.mean_regrid(wno, (1 - cloud_frac) * fpfs_cf + cloud_frac * fpfs_c, R=150)

    if outputfile is not None:
        _suffix = f'_phase{phase:.4f}_cld{cloud_frac}_psurf{p_surface_bar}'
        if exclude_all_gas:
            _suffix += '_nogas'
        elif 'exclude_mol' in atmosphere_kwargs:
            _suffix += '_no' + ''.join(atmosphere_kwargs['exclude_mol'])
        if no_rayleigh:
            _suffix += '_noray'
        _out = Path(outputfile)
        _save_path = _out.parent / f'EarthSpectrum_{_out.name}{_suffix}.pkl'
        _save_path.parent.mkdir(parents=True, exist_ok=True)
        earth_outputs = {
            'wno': wno,
            'fpfs': fpfs,
            'albedo': albedo,
            'df_cld': df_cld,
            'df_cldfree': df_cldfree,
        }
        with open(_save_path, 'wb') as f:
            pickle.dump(earth_outputs, f)
        print(f'Saved {_save_path}')

    return wno, fpfs, albedo, df_cld, df_cldfree


def make_case_earth(
    opacity_path=opacity_path,
    df_mol_earth=None,
    phase=0,
    species=None,
    cloud_frac=0.5,
    p_surface_bar=1.0,
    nlevel=90,
    cloud_ptop_bar=0.6,
    cloud_pbot_bar=0.7,
):
    """
    Returns a dict of wno, albedo, fpfs results from earth_spectrum.

    Provide species as a list to also compute exclude_mol variants:
    species = ['O2', 'H2O', 'CO2', 'O3', 'CH4']
    """
    kwargs = dict(
        opacity_path=opacity_path,
        df_mol_earth=df_mol_earth,
        phase=phase,
        cloud_frac=cloud_frac,
        p_surface_bar=p_surface_bar,
        nlevel=nlevel,
        cloud_ptop_bar=cloud_ptop_bar,
        cloud_pbot_bar=cloud_pbot_bar,
    )
    res = {}
    res['all'] = earth_spectrum(**kwargs)

    if species is not None:
        for sp in species:
            tmp = earth_spectrum(**{**kwargs, 'atmosphere_kwargs': {'exclude_mol': [sp]}})
            res[sp] = tmp[:2]
        return res

    return res


def save_array_or_string(group, key, arr):
    """
    Save numeric arrays, string arrays, or object arrays safely.
    If the array contains nested lists/objects, convert to groups recursively.
    """
    # If it's a numpy scalar
    if np.isscalar(arr):
        group.attrs[key] = arr
        return

    # Convert to numpy array if list/tuple
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr, dtype=object)

    # If numeric array
    if np.issubdtype(arr.dtype, np.number):
        group.create_dataset(key, data=arr, compression="gzip", chunks=True)

    # If string array
    elif arr.dtype.kind in {"U", "S"}:
        dt = h5py.string_dtype(encoding="utf-8")
        data = np.array(arr.tolist(), dtype=object)  # force Python str
        group.create_dataset(key, data=data, dtype=dt)

    # If object array
    elif arr.dtype.kind == "O":
        # check if all elements are strings
        if all(isinstance(x, str) for x in arr.flat):
            dt = h5py.string_dtype(encoding="utf-8")
            data = np.array(arr.tolist(), dtype=object)
            group.create_dataset(key, data=data, dtype=dt)
        else:
            # Otherwise, create a group for each element
            subgrp = group.create_group(key)
            for i, val in enumerate(arr):
                subkey = f"{i}"
                # Recurse
                save_array_or_string(subgrp, subkey, val)

    else:
        raise TypeError(f"Cannot save array of dtype {arr.dtype} for key {key}")

def save_dict_to_hdf5(group, dictionary):

    for key, value in dictionary.items():
        key = str(key)

        # ---- Nested dictionary ----
        if isinstance(value, dict):
            subgrp = group.create_group(key)
            save_dict_to_hdf5(subgrp, value)

        # ---- Pandas DataFrame ----
        elif isinstance(value, pd.DataFrame):
            df_grp = group.create_group(key)
            df_grp.attrs["columns"] = list(value.columns)
            df_grp.attrs["index"] = value.index.to_numpy()
            for col in value.columns:
                col_data = value[col].to_numpy()
                save_array_or_string(df_grp, col, col_data)

        # ---- NumPy array ----
        elif isinstance(value, np.ndarray):
            save_array_or_string(group, key, value)

        # ---- List or tuple ----
        elif isinstance(value, (list, tuple)):
            arr = np.array(value)
            save_array_or_string(group, key, arr)

        # ---- Python string ----
        elif isinstance(value, str):
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(key, data=value, dtype=dt)

        # ---- Scalar numeric ----
        elif np.isscalar(value):
            group.attrs[key] = value

        else:
            raise TypeError(f"Unsupported type for key '{key}': {type(value)}")


def calc_RSM_earth_phases(df_mol_earth=None, phase_earth=None, earth_type='Archean', cloud_frac = 0.5):

    if phase_earth is None:
        phase_earth = np.linspace(0, np.pi, 19)
        phase_angle = phase_earth[:-1]
    else:
        phase_angle = phase_earth

    if df_mol_earth is None:
        df_mol_earth = {
            "N2": 0.945,
            "CO2": 0.05,
            "CO": 0.0005,
            "CH4": 0.005,
            "H2O": 0.003
        }

    filename = f"{earth_type}_earth_diff_phases_cldfrac{cloud_frac}.h5"

    with h5py.File(filename, "w") as f:

        f.attrs["earth_type"] = earth_type
        f.attrs["num_phases"] = len(phase_angle)

        for phase in phase_angle:

            res_earth = make_case_earth(df_mol_earth=df_mol_earth, phase=phase, cloud_frac=cloud_frac)

            wv = res_earth['all'][0]
            fpfs = res_earth['all'][1]
            alb = res_earth['all'][2]
            df_cld = res_earth['all'][3]        # dictionary
            df_cldfree = res_earth['all'][4]    # dictionary

            phase_str = f"{phase:.4f}"
            grp = f.create_group(f"phase_{phase_str}")
            grp.attrs["phase_radians"] = phase

            # Save main arrays
            grp.create_dataset("wv", data=wv, compression="gzip", chunks=True)
            grp.create_dataset("fpfs", data=fpfs, compression="gzip", chunks=True)
            grp.create_dataset("alb", data=alb, compression="gzip", chunks=True)

            # Save nested dictionaries
            cld_grp = grp.create_group("df_cld")
            save_dict_to_hdf5(cld_grp, df_cld)

            cldfree_grp = grp.create_group("df_cldfree")
            save_dict_to_hdf5(cldfree_grp, df_cldfree)

    return print(f"{filename} recorded successfully.")
