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

from mpi4py import MPI

#from gridutils import make_grid
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import tarfile
import gridutils

import h5py

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

def PICASO_PT_Planet(rad_plan=1, log_mh=2.0, tint=60, semi_major_AU=1, ctoO='1', nlevel=91, nofczns=1, nstr_upper=85, rfacv=0.5, outputfile=None, pt_guillot=True, prior_out=None):

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
    ctoO = string
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
    
    # Opacity is independent from log_mh and ctoO, instead of downloading opacities
    opacity_ck = jdi.opannection(method='resortrebin') # grab your opacities

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
        out = cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True)
        base_case = jdi.pd.read_csv(jdi.HJ_pt(), delim_whitespace=True)

    except Exception as e:
        print(f"An exception was raised: {type(e).__name__}: {e}")
        raise
    
    # Saves out and base_case to python file to be re-loaded.
    if outputfile == None:
        return out, base_case
                
    else:
        with open(f'out_{outputfile}.pkl', 'wb') as f:
            pickle.dump(out, f)
        with open(f'basecase_{outputfile}.pkl', 'wb') as f:
            pickle.dump(base_case, f)
        return out, base_case
   
def PICASO_fake_climate_model_testing_errors(rad_plan, log_mh, tint, semi_major_AU, ctoO, outputfile=None):
    
    fake_dictionary = {'planet radius': np.full(10, rad_plan), 'log_mh': np.full(10, log_mh) , 'tint': np.full(10, tint), 'semi major': np.full(10, semi_major_AU), 'ctoO': np.full(10, ctoO)}
    return fake_dictionary

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
    print(f'This is the value of {x} used in the climate model')
    out, base_case = PICASO_PT_Planet(rad_plan=rad_plan_earth_units, log_mh=log10_planet_metallicity, tint=tint_K, semi_major_AU = semi_major_AU, ctoO=ctoO_solar, outputfile=None)

    count = 0
    while out['converged'] == 0:  # An infinite loop that will be broken out of explicitly
        count += 1
        
        print(f"Loop iteration, Recalculating PT Profile: {count}")
        
        out, base_case = PICASO_PT_Planet(rad_plan=rad_plan_earth_units, log_mh=log10_planet_metallicity, tint=tint_K, semi_major_AU = semi_major_AU, ctoO=ctoO_solar, outputfile=None, pt_guillot=False, prior_out = out)

        if count == 3:
            print(f"Hit the maximum amount of loops without converging.")
            break  # Exit the loop when count reaches 3

    desired_keys = ['pressure', 'temperature', 'converged']
    new_out = {key: out[key] for key in desired_keys if key in out} # Only picks out some array results from Photochem b/c not all were arrays
    new_out['converged'] = np.array([new_out['converged']])
    # Try specifying the dictionary w/ inputs and outputs

    # Testing (with a simple dictionary, the code works)
    # out = PICASO_fake_climate_model_testing_errors(log_mh=log10_planet_metallicity, tint=tint, total_flux=log10_totalflux, outputfile=None)

    return new_out

def get_gridvals_PICASO_TP():

    
    """
    This provides the input parameters to run the climate model over multiple computers (i.e. paralell computing).

    Parameter(s):
    log10_totalflux = np.array of floats
        This is the total flux of the starlight on the planet in units of x Solar
    log10_planet_metallicity = np.array of strings
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
    semi_major_AU = np.array([1.04]) # in AU 
    ctoO_solar = np.array([1]) # in units of solar C/O
    
    """
    """
    
    # Parameter Exploration
    rad_plan_earth_units = np.array([1.6, 4]) # in units of xEarth radii
    log10_planet_metallicity = np.array([0.5, 3.5]) # in units of solar metallicity
    tint_K = np.array([20, 400]) # in Kelvin
    semi_major_AU = np.array([0.3, 10]) # in AU 
    ctoO_solar = np.array([0.01, 1]) # in units of solar C/O

    """

    # Parameter Exploration Refined
    rad_plan_earth_units = np.array([1.6, 4]) # in units of xEarth radii
    log10_planet_metallicity = np.array([3.5]) # in units of solar metallicity
    tint_K = np.array([50, 75, 100]) # in Kelvin
    semi_major_AU = np.array([0.3, 10]) # in AU 
    ctoO_solar = np.array([0.1, 1]) # in units of solar C/O

    gridvals = (rad_plan_earth_units, log10_planet_metallicity, tint_K, semi_major_AU, ctoO_solar)
    
    return gridvals

if __name__ == "__main__":

    """
    To execute running 1D PICASO climate model for the range of values in get_gridvals_PICASO_TP, type the folling command into your terminal:
    # mpiexec -n X python PICASO_Climate_grid_121625.py

    """
    
    gridutils.make_grid(
        model_func=PICASO_climate_model, 
        gridvals=get_gridvals_PICASO_TP(), 
        filename='results/PICASO_climate_updatop_paramext_refineTINT_CO.h5', 
        progress_filename='results/PICASO_climate_updatop_paramext_refineTINT_CO.log'
    ) 



