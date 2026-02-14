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
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline

from astropy import constants
from photochem.utils import zahnle_rx_and_thermo_files
from photochem.extensions import gasgiants # Import the gasgiant extensions

import json
from astroquery.mast import Observations
from photochem.utils import stars

import pickle
import requests

from mpi4py import MPI

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import tarfile
import gridutils
import star_spectrum
import Photochem_grid_121625 as Photochem_grid
import h5py
import numpy as np
import pandas as pd
import copy
    
def earth_spectrum(opacity_path, df_mol_earth, phase, atmosphere_kwargs={}):

    """
    Calculates the Modern Earth Reflected Spectrum at full phase around the same star (Sun). 

    Parameters:
    opacity_path: string
        This provides the path to the opacity file you wish to use (we recommend v3 from Batalha et. al. 2025 on zenodo titled "Resampled Opacity Database for PICASO".
    atmosphere_kwargs: 'key': value
        If you wish to exclude any molecules, you can create a key titled 'exclude_mol' and add a list of molecules you do not wish to computer the reflected spectra of.
    df_mol_earth: dictionary with allowable abundances of molecules from the period of Earth you want
        
        EXAMPLE:

        df_mol_earth = {"N2": 0.79,
            "O2": 0.21,
            "O3": 7e-7,
            "H2O": 3e-3,
            "CO2": 300e-6,
            "CH4": 1.7e-6
        }

    Results:
    wno: grid of 150 values
        This is something, idk.
    fpfs: grid of 150 values
        This is the relative flux of the planet and star (fp/fs). 
    albedo: grid of 150 values
        This is something, idk.
    
    """

    earth = jdi.inputs()
    
    # Phase angle 
    earth.phase_angle(phase, num_tangle=8, num_gangle=8) #radians
    
    # Define planet gravity
    earth.gravity(radius=1, radius_unit=jdi.u.Unit('R_earth'),
                 mass =1, mass_unit=jdi.u.Unit('M_earth')) #any astropy units available
    earth.approx(raman="none")
    
    # Define star (same as used in K218b grid calculations)
    stellar_radius = 1 # Solar radii
    stellar_Teff = 5778 # K
    stellar_metal = 0.0 # log10(metallicity)
    stellar_logg = 4.4 # log10(gravity), in cgs units
    opacity = jdi.opannection(method=opacity_path, wave_range=[0.3,2.5])
    
    earth.star(opannection=opacity,temp=stellar_Teff,logg=stellar_logg,semi_major=1, metal=stellar_metal,
               semi_major_unit=u.Unit('au')) 

    # P-T-Composition
    nlevel = 90 
    P = np.logspace(-6, 0, nlevel)
    df_atmo = earth.TP_line_earth(P , nlevel = nlevel)
    df_pt_earth =  pd.DataFrame({
        'pressure':df_atmo['pressure'].values,
        'temperature':df_atmo['temperature'].values})

    if df_mol_earth == None:
        df_mol_earth_modern_default = pd.DataFrame({
                "N2":P*0+0.79,
                "O2":P*0+0.21,
                "O3":P*0+7e-7,
                "H2O":P*0+3e-3,
                "CO2":P*0+300e-6,
                "CH4":P*0+1.7e-6
            })
        
        df_atmo_earth = df_pt_earth.join(df_mol_earth_modern_default, how='inner')
        print(df_atmo_earth)

    else:
        df_mol_earth_grid_dict = {}
        df_mol_earth_grid = pd.DataFrame({})
        for key in df_mol_earth:
            df_mol_earth_grid_dict[key] = df_mol_earth[key] + (P*0)
            for key in df_mol_earth_grid_dict:
                df_mol_earth_grid[key] = pd.Series(df_mol_earth_grid_dict[key])

        df_atmo_earth = df_pt_earth.join(df_mol_earth_grid, how='inner')
        print(df_atmo_earth)
            
    if 'exclude_mol' in atmosphere_kwargs:
        sp = atmosphere_kwargs['exclude_mol'][0]
        if sp in df_atmo_earth:
            df_atmo_earth[sp] *= 0
            
    # earth.atmosphere(df=df_atmo_earth, **atmosphere_kwargs)
    
    earth.atmosphere(df=df_atmo_earth)
    earth.surface_reflect(0.1,opacity.wno)

    # Cloud free spectrum
    df_cldfree = earth.spectrum(opacity,calculation='reflected',full_output=True)

    # Clouds
    ptop = 0.6
    pbot = 0.7
    logdp = np.log10(pbot) - np.log10(ptop)  
    log_pbot = np.log10(pbot)
    earth.clouds(w0=[0.99], g0=[0.85], 
                 p = [log_pbot], dp = [logdp], opd=[10])

    # Cloud spectrum
    df_cld = earth.spectrum(opacity,full_output=True)

    # Average the two spectra
    wno, alb, fpfs, albedo = df_cldfree['wavenumber'],df_cldfree['albedo'],df_cldfree['fpfs_reflected'], df_cldfree['albedo']
    wno_c, alb_c, fpfs_c, albedo_c = df_cld['wavenumber'],df_cld['albedo'],df_cld['fpfs_reflected'], df_cld['albedo']
    _, albedo = jdi.mean_regrid(wno, 0.5*albedo+0.5*albedo_c,R=150)
    wno, fpfs = jdi.mean_regrid(wno, 0.5*fpfs+0.5*fpfs_c,R=150)
    

    return wno, fpfs, albedo

# How do we want to deal with opacity in reflected spectra from PICASO?

def make_case_earth(df_mol_earth=None, phase=0, species=None):

    """
    This calculates a dictionary of wno, albedo, and fpfs results from earth_spectrum.

    Provide a list if you wish to limit the species calculated by the reflected light spectra.
    species = ['O2','H2O','CO2','O3','CH4']
    
    """
    res = {}
    res['all'] = earth_spectrum(df_mol_earth, phase) # in order of wno, fpfs, alb
    
    if species is not None:
        for sp in species:
            tmp = earth_spectrum(atmosphere_kwargs={'exclude_mol': [sp]})
            res[sp] = tmp[:2]
        return res

    else:
        return res
    
def calc_semi_major_SUN(Teq):

    """
    This calculates the semi major axis (AU) between the Sun and a planet given the planet's equilibrium temperature (K).
    """
    luminosity_star = 3.846*(10**26) # in Watts for the Sun
    boltzmann_const = 5.670374419*(10**-8) # in W/m^2 * K^4
    distance_m = np.sqrt(luminosity_star / (16 * np.pi * boltzmann_const * (Teq**4)))
    distance_AU = distance_m / 1.496e+11
    return distance_AU
    
def find_Photochem_match(filename='results/Photochem_1D_fv.h5',  rad_plan=None, log10_planet_metallicity=None, tint=None, semi_major=None, ctoO=None, Kzz=None, gridvals= Photochem_grid.get_gridvals_Photochem()):
    
    """
    This finds the Photochem match on the grid based on inputs into the Reflected Spectra grid.

    Parameters:
    filename: string
        this is the file path to the output of makegrid for Photochemical model
    rad_plan: float
        This is the radius of your planet in units of x Earth Radii.
    log10_planet_metallicity: float
        This is the planet's metallicity in units of log10 x Solar metallicity.
    tint: float
        This is the planet's internal temperature in Kelvin.
    semi_major: float
        This is the semi major axis of your planet's orbit in units of AU.
    ctoO: float
        This is the carbon to oxygen ratio of your planet's atmosphere in units of xSolar c/o.
    Kzz: float
        This is the eddy diffusion coefficient in logspace (i.e. the power of 10) in cm/s^2.
    gridvals: tuple of 1D arrays
        Input values for total_flux, planet metallcity, tint, and kzz used to make the Photochemical grid.

    Results:
    sol_dict_new: dictionary of np.arrays
        This provides the matching solutions dictionary from photochem matching total_flux, metallicity, tint, and kzz inputs.
    soled_dict_new: dictionary of np.arrays
        This provides the matching solutions dictionary (from chemical equilibrium) matching total_flux, metallicity, tint, and kzz inputs.
    PT_list: 2D array
        This provides the matching pressure (in dynes/cm^2), temperature (Kelvin) from the Photochemical grid solution (not PICASO, since this involved some extrapolation and interpolation). 
    convergence_PC: 1D array
        This provides information on whether or not the Photochem model converged, using the binary equivalent of booleans (1=True, 0=False)
    convergence_TP: 1D array
        This provides information on whether or not the PICASO model used in Photochem was converged, using binary equivalent of booleans (1=True, 0=False)
        
    """
    gridvals_metal = [float(s) for s in gridvals[1]]
    gridvals_ctoO = [float(s) for s in gridvals[4]]

    planet_metallicity = float(log10_planet_metallicity)
    gridvals_dict = {'rad_plan':gridvals[0], 
                     'planet_metallicity':gridvals_metal, 
                     'tint':gridvals[2], 
                     'semi_major':gridvals[3],
                     'ctoO':gridvals_ctoO,
                     'Kzz':gridvals[5]}


    with h5py.File(filename, 'r') as f:
        input_list = np.array([rad_plan, planet_metallicity, tint, semi_major, ctoO, Kzz])
        matches = (list(f['inputs'] == input_list))
        row_matches = np.all(matches, axis=1)
        matching_indicies = np.where(row_matches)

        matching_indicies_rad_plan = np.where(list(gridvals_dict['rad_plan'] == input_list[0]))
        matching_indicies_metal = np.where(list(gridvals_dict['planet_metallicity'] == input_list[1]))
        matching_indicies_tint = np.where(list(gridvals_dict['tint'] == input_list[2]))
        matching_indicies_semi_major = np.where(list(gridvals_dict['semi_major'] == input_list[3]))
        matching_indicies_ctoO = np.where(list(gridvals_dict['ctoO'] == input_list[4]))
        matching_indicies_kzz = np.where(list(gridvals_dict['Kzz'] == input_list[5]))

        rad_plan_index, metal_index, tint_index, semi_major_index, ctoO_index, kzz_index = matching_indicies_rad_plan[0], matching_indicies_metal[0], matching_indicies_tint[0], matching_indicies_semi_major[0], matching_indicies_ctoO[0], matching_indicies_kzz[0]

        if matching_indicies[0].size == 0:
            print(f'A match given planet radius, metallicity, tint, semi-major axis, ctoO, and Kzz does not exist')
            sol_dict_new = None
            soleq_dict_new = None
            PT_list = None
            convergence_PC = None
            convergence_TP = None
            return sol_dict_new, soleq_dict_new, PT_list, convergence_PC, convergence_TP
            
        else:
            sol_dict = {}
            soleq_dict = {}
            for key in list(f['results']):
                if key.endswith("sol"):
                    sol_dict[key] = np.array(f['results'][key][rad_plan_index[0]][metal_index[0]][tint_index[0]][semi_major_index[0]][ctoO_index[0]][kzz_index[0]])
                elif key.endswith("soleq"):
                    soleq_dict[key] = np.array(f['results'][key][rad_plan_index[0]][metal_index[0]][tint_index[0]][semi_major_index[0]][ctoO_index[0]][kzz_index[0]])

            sol_dict_new = {key.removesuffix('_sol') if key.endswith('_sol') else key: value 
    for key, value in sol_dict.items()}

            soleq_dict_new = {key.removesuffix('_soleq') if key.endswith('_soleq') else key: value 
    for key, value in soleq_dict.items()}
                        
            pressure_values = np.array(f['results']['pressure_sol'][rad_plan_index[0]][metal_index[0]][tint_index[0]][semi_major_index[0]][ctoO_index[0]][kzz_index[0]])
            temperature_values = np.array(f['results']['temperature_sol'][rad_plan_index[0]][metal_index[0]][tint_index[0]][semi_major_index[0]][ctoO_index[0]][kzz_index[0]])
            convergence_PC = np.array(f['results']['converged_PC'][rad_plan_index[0]][metal_index[0]][tint_index[0]][semi_major_index[0]][ctoO_index[0]][kzz_index[0]])
            convergence_TP = np.array(f['results']['converged_TP'][rad_plan_index[0]][metal_index[0]][tint_index[0]][semi_major_index[0]][ctoO_index[0]][kzz_index[0]])
            PT_list = pressure_values, temperature_values
            print(f'Was able to successfully find your input parameters in the PICASO TP profile grid!')
            
            return sol_dict_new, soleq_dict_new, PT_list, convergence_PC, convergence_TP

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


# Make a Global Variable

current_directory = Path.cwd()
opacity_file_path = "Installation&Setup_Instructions/picasofiles/reference/opacities/opacities_photochem_0.1_250.0_R15000.db"
print(os.path.join(current_directory, opacity_file_path))

opacity_path=os.path.join(current_directory, opacity_file_path)
OPACITY = jdi.opannection(filename_db=opacity_path, wave_range=[0.3,2.5])

# Flip the data between PICASO and Photochem

def make_picaso_atm(sol):
    """
    Takes in a dictionary from Photochem output, converts pressure from dynes/cm^2 to bars, and flips all data for PICASO, and gets rid of any aer molecule abundances. Returns a dictionary.
    
    """
    sol_dict_noaer = {}
    sol_dict_aer = {}
    for key in sol.keys():
        if not key.endswith('aer'):
            sol_dict_noaer[key] = sol[key]
        elif key.endswith('aer'):
            sol_dict_aer[key] = sol[key]
        else:
            continue
        
    atm = copy.deepcopy(sol_dict_noaer)
    atm['pressure'] /= 1e6 # in bars
    for key in atm:
        atm[key] = atm[key][::-1].copy()

    sol_dict_aer = copy.deepcopy(sol_dict_aer)
    for key in sol_dict_aer:
        sol_dict_aer[key] = sol_dict_aer[key][::-1].copy()
    
    return atm, sol_dict_aer

# This calculates the spectrum (for now, without clouds)
def reflected_spectrum_planet_Sun(rad_plan=None, planet_metal=None, tint=None, semi_major=None, ctoO=None, Kzz=None, phase_angle=None, Photochem_file='results/Photochem_1D_fv.h5', atmosphere_kwargs={}):

    """
    This finds the reflected spectra of a planet similar to K218b around a Sun.

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

    opacity = OPACITY

    planet_metal = float(planet_metal)
    
    start_case = jdi.inputs()

    # Then calculate the composition from the TP profile
    class planet:
        
        planet_radius = (rad_plan*6.371e+6*u.m) # in meters
        planet_mass = PICASO_Climate_grid.mass_from_radius_chen_kipping_2017(R_rearth=rad_plan)*(5.972e+24) # in kg
        planet_Teq = PICASO_Climate_grid.calc_Teq_SUN(distance_AU=semi_major) # Equilibrium temp (K)
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
    start_case.gravity(gravity=planet.planet_grav, gravity_unit=jdi.u.Unit('m/(s**2)'), radius=(planet.planet_radius.value)/jupiter_radius, radius_unit=jdi.u.Unit('R_jup'), mass=(planet.planet_mass.value)/jupiter_mass, mass_unit=jdi.u.Unit('M_jup'))
    
    # star temperature, metallicity, gravity, and opacity (default opacity is opacity.db in the reference folder)
    start_case.star(opannection=opacity, temp=Sun.stellar_Teff, logg=Sun.stellar_logg, semi_major=semi_major, metal=Sun.stellar_metal, radius=Sun.stellar_radius, radius_unit=jdi.u.R_sun, semi_major_unit=jdi.u.au)

    # Match Photochemical Files
    sol_dict, soleq_dict, PT_list, convergence_PC, convergence_TP = find_Photochem_match(filename=Photochem_file, rad_plan=rad_plan, log10_planet_metallicity=planet_metal, tint=tint, semi_major=semi_major, ctoO=ctoO,Kzz=Kzz)

    # Determine Planet Atmosphere & Composition

    atm, sol_dict_aer = make_picaso_atm(sol_dict) # Converted Pressure of Photochem, in dynes/cm^2, back to bars and flip all arrays before placing into PICASO
    df_atmo = jdi.pd.DataFrame(atm)

    if 'exclude_mol' in atmosphere_kwargs:
        sp = atmosphere_kwargs['exclude_mol'][0]
        if sp in df_atmo:
            df_atmo[sp] *= 0
            
    start_case.atmosphere(df = df_atmo) 
    df_cldfree = start_case.spectrum(opacity, calculation='reflected', full_output=True)
    wno_cldfree, alb_cldfree, fpfs_cldfree = df_cldfree['wavenumber'], df_cldfree['albedo'], df_cldfree['fpfs_reflected']
    _, alb_cldfree_grid = jdi.mean_regrid(wno_cldfree, alb_cldfree, R=150)
    wno_cldfree_grid, fpfs_cldfree_grid = jdi.mean_regrid(wno_cldfree, fpfs_cldfree, R=150)

    print(f'This is the length of the grids created: {len(wno_cldfree_grid)}, {len(fpfs_cldfree_grid)}')

    # Determine Whether to Add Clouds or Not?

    if "H2Oaer" in sol_dict_aer:
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
            _, alb = jdi.mean_regrid(wno_cldfree, 0.5*alb_cldfree+0.5*albedo_c,R=150)
            wno, fpfs = jdi.mean_regrid(wno_cldfree, 0.5*fpfs_cldfree+0.5*fpfs_c,R=150)

            # Match the length of the clouds array with the length of wno or alb (fpfs is different length)
            clouds = [1] * len(wno)

            return wno, fpfs, alb, np.array(clouds)

        else:
            print(f'pbot is empty, so no cloud is implemented')
            wno = wno_cldfree_grid.copy()
            alb = alb_cldfree_grid.copy()
            fpfs = fpfs_cldfree_grid.copy()

            # Match the length of the clouds array with the length of wno or alb (fpfs is different length)
            clouds = [0] * len(wno)

            print(f'This is the length of the values I want to save: wno {len(wno)}, alb {len(alb)}, fpfs {len(fpfs)}, clouds {len(clouds)}')

            return wno, fpfs, alb, np.array(clouds)

    else:
        print(f'H2Oaer is not in solutions')
        wno = wno_cldfree_grid.copy()
        alb = alb_cldfree_grid.copy()
        fpfs = fpfs_cldfree_grid.copy()
        print(f'For the inputs: {rad_plan}, {planet_metal}, {tint}, {semi_major}, {ctoO}, {Kzz}, {phase_angle}, The length should match: wno - {len(wno)}, alb - {len(alb)}, fpfs - {len(fpfs)}')
        
        # Match the length of the clouds array with the length of wno or alb (fpfs is different length)
        clouds = [0] * len(wno) # This means that there are no clouds

        return wno, fpfs, alb, np.array(clouds)

def make_case_RSM(rad_plan=None, planet_metal=None, tint=None, semi_major=None, ctoO=None, Kzz=None, phase_angle=None, limit_sp = False, species=['O2','H2O','CO2','O3','CH4']):

    """
    This calculates a dictionary of wno, albedo, and fpfs results from reflected_spectrum_planet_Sun. When limit_sp is True, it will exclude species O2, H2O, CO2, O3, and CH4, but by default just puts outputs into a dictionary with the keys wno, fpfs, albedo, and clouds.
    
    """
    
    res = {}
    
    wno, fpfs, albedo, clouds = reflected_spectrum_planet_Sun(rad_plan=rad_plan, planet_metal=planet_metal, tint=tint, semi_major=semi_major, ctoO=ctoO, Kzz=Kzz, phase_angle=phase_angle)
    
    res['wno'] = wno
    res['fpfs'] = fpfs
    res['albedo'] = albedo
    res['clouds'] = clouds

    if limit_sp == True:
        for sp in species:
            tmp = reflected_spectrum_planet_Sun(rad_plan=rad_plan, planet_metal=planet_metal, tint=tint, semi_major=semi_major, ctoO=ctoO, Kzz=Kzz, phase_angle=phase_angle, atmosphere_kwargs={'exclude_mol': [sp]})
            res[sp] = tmp[:2]
            
        return res
    else:
        return res

def get_gridvals_RSM():

    """
    This provides the input parameters to run the reflected spectra model over multiple computers (i.e. paralell computing).

    Parameter(s):
    rad_plan = np.array of floats
        This is the radius of the planet in units of x Earth Radii
    log10_planet_metallicity = np.array of strings
        This is the planet metallicity in units of log10 x Solar
    tint = np.array of floats
        This is the internal temperature of the planet in units of Kelvin
    semi_major = np.array of floats
        This is the semi major axis of the planet's orbit in units of AU
    ctoO = np.array of floats
        This is the carbon to oxygen ratio of the planet's atmosphere in units of xSolar c/o
    log_kzz = np.array of floats
        This is the eddy diffusion coefficient (the power of 10) in cm^2/s
    phase_angle = np.array of floats
        This is the phase of orbit the planet is relative to its star & the observer in radians
    
    Returns:
        A tuple array of each array of input parameters to run via parallelization and return a reflected spectra grid.

    """

    """
    # Test Case:
    rad_plan_earth_units = np.array([2.61]) # in units of xEarth radii
    log10_planet_metallicity = np.array(['3.5']) # in units of solar metallicity
    tint_K = np.array([155]) # in Kelvin
    semi_major_AU = np.array([1]) # in AU 
    ctoO_solar = np.array([0.01]) # in units of solar C/O
    log_Kzz = np.array([5])
    phase_angle_list = np.linspace(0, np.pi, 19)
    phase_angle = phase_angle_list[:-1] # in radians, this goes in 10 degree intervals from 0 to 170 degrees
    """

    
    # Parameter Exploration
    rad_plan_earth_units = np.array([1.6, 4]) # in units of xEarth radii
    log10_planet_metallicity = np.array([0.5, 3.5]) # in units of solar metallicity
    tint_K = np.array([20, 400]) # in Kelvin
    semi_major_AU = np.array([0.3, 10]) # in AU 
    ctoO_solar = np.array([0.01, 1]) # in units of solar C/O
    log_Kzz = np.array([5, 9]) # In units of logspace (so 5 means 10^5 cm^2/s)
    phase_angle_list = np.linspace(0, np.pi, 19)
    phase_angle = phase_angle_list[:-1] # in radians, this goes in 10 degree intervals from 0 to 170 degrees
    
    """
    # True Values to replace after test case (what is being run on Tijuca) --> old inputs
    log10_totalflux = np.array([0.1, 0.5, 1, 1.5, 2])
    log10_planet_metallicity = np.array(['0.5', '1.0', '1.5', '2.0']) # in solar, the opacity files are min 0 and max 2, so I cannot do 2.5 and 3.0!
    tint = np.array([20, 40, 60, 100, 120, 140, 160, 200]) # in Kelvin
    log_Kzz = np.array([5, 7, 9]) # in cm^2/s 
    phase_angle_list = np.linspace(0, np.pi, 19)
    phase_angle = phase_angle_list[:-1] # in radians, this goes in 10 degree intervals from 0 to 170 degrees
    """

    gridvals = (rad_plan_earth_units, log10_planet_metallicity, tint_K, semi_major_AU, ctoO_solar, log_Kzz, phase_angle)
    
    return gridvals

def Reflected_Spectra_model(x):

    """
    This runs Photochem_Gas_Giant on Tijuca for parallel computing.

    Parameters:
        x needs to be in the order of total flux, planet metallicity, tint, and kzz!
        total flux = units of solar (float)
        planet metallicity = units of log10 solar but needs to be a float/integer NOT STRING
        tint = units of Kelvin (float)
        kzz = units of cm^2/s (float)
        phase_angle = units of radians (float)

    Results:
    res: dictionary
        This gives you all the results of reflected_spectrum_K218b_Sun.

    """
    # For Tijuca
    rad_plan_earth_units, log10_planet_metallicity, tint_K, semi_major_AU, ctoO_solar, log_Kzz, phase_angle = x
    
    res = make_case_RSM(rad_plan_earth_units=rad_plan_earth_units, planet_metal=log10_planet_metallicity, tint=tint_K, semi_major_AU=semi_major_AU, ctoO_solar=ctoO_solar, Kzz=log_Kzz, phase_angle=phase_angle)

    return res

if __name__ == "__main__":
    """
    To execute running Reflected Spectra model for the range of values in get_gridvals_RSM, type the folling command into your terminal:
   
    # mpiexec -n X python Reflected_Spectra_grid.py
    
    """
    gridutils.make_grid(
        model_func=Reflected_Spectra_model, 
        gridvals=get_gridvals_RSM(), 
        filename='results/ReflectedSpectra_fv.h5', 
        progress_filename='results/ReflectedSpectra_fv.log'
    )
