import gridutils
import Photochem_grid_121625 as Photochem_grid
import PICASO_Climate_grid_121625 as PICASO_Climate_grid
import GraphsKey
import numpy as np
from scipy import interpolate
import h5py


def find_PT_sol(filepath='/mnt/c/Users/lily/Documents/NASAUWPostbac/MiniNeptuneGrid26_PostBac/results/PICASO_climate_updatop_paramext_K218b.h5',rad_plan=None, log10_planet_metallicity=None, tint=None, semi_major=None, ctoO=None, grid_gridvals=PICASO_Climate_grid.get_gridvals_PICASO_TP()):

    """
    Inputs:
    TBD
    """

    # This takes the inputs that define the grid
    gridshape = tuple(len(a) for a in grid_gridvals)

    print(f"Make sure your inputs are within the following ranges, rad_plan: {np.min(grid_gridvals[0])} - {np.max(grid_gridvals[0])} xEarth Radii, planet metallicity: {np.min((grid_gridvals[1]).astype(float))} - {np.max((grid_gridvals[1]).astype(float))} xsolar, tint: {np.min(grid_gridvals[2])} - {np.max(grid_gridvals[2])} K, semi_major: {np.min(grid_gridvals[3])} - {np.max(grid_gridvals[3])} AU, ctoO: {np.min(grid_gridvals[4])} - {np.max(grid_gridvals[4])}")

    # Check to see if there is a solution that already exists
    PT_list, convergence_values = Photochem_grid.find_PT_grid(filename=filepath, rad_plan=rad_plan, log10_planet_metallicity=log10_planet_metallicity, tint=tint, semi_major=semi_major, ctoO=ctoO)

    if PT_list is not None:
        print(f'All inputs chosen were directly on the grid!')
        comb_results = {}
        comb_results['pressure'] = PT_list[0]
        comb_results['temperature'] = PT_list[1]
        # comb_results['converged'] = convergence_values
        return comb_results
        

    elif PT_list is None:

        print(f'Interpolating results...')
        
        # This notes the grid and associated inputs used to make it as the data
        PhotCh_grid = gridutils.GridInterpolator(filename=filepath, gridvals=grid_gridvals)

        # This interpolates the results based on the user input
        interp_results = {}
    
        # New grid values to interpolate
        user_gridvals = (rad_plan, log10_planet_metallicity, tint, semi_major, ctoO)
    
        for key in PhotCh_grid.data.keys():
            if key.startswith('pressure'):
                interp_function_pressure = PhotCh_grid.make_interpolator(key=key, logspace=True)
                interp_results[key] = interp_function_pressure(user_gridvals)
            elif key.startswith('temperature'):
                interp_function_temperature = PhotCh_grid.make_interpolator(key=key, logspace=False)
                interp_results[key] = interp_function_temperature(user_gridvals)
            elif key.startswith('converged'):
                continue
            else:
                interp_function = PhotCh_grid.make_interpolator(key=key, logspace=True)
                interp_results[key] = interp_function(user_gridvals)
                
        return interp_results

def find_Photochem_match(filename='results/Photochem_1D_updatop_paramext_K218b.h5', rad_plan=None, log10_planet_metallicity=None, tint=None, semi_major=None, ctoO=None, Kzz=None, gridvals= Photochem_grid.get_gridvals_Photochem()):
    
    """
    This finds the Photochem match on the grid based on inputs into the Reflected Spectra grid.

    Parameters:
    filename: string
        this is the file path to the output of makegrid for Photochemical model
    rad_plan: float
        This is the radius of the planet in units of xEarth Radii.
    log10_planet_metallicity: float
        This is the planet's metallicity in units of log10 x Solar metallicity.
    tint: float
        This is the planet's internal temperature in Kelvin.
    semi_major: float
        This is the planet's distance from its host star in units of AU. 
    ctoO: float
        This is the carbon to oxygen ratio in the atmosphere of the planet (unitless). 
    Kzz: float
        This is the eddy diffusion coefficient in logspace (i.e. the power of 10) in cm/s^2.
    gridvals: tuple of 1D arrays
        Input values for rad_plan, planet metallcity, tint, semi_major, ctoO, and kzz used to make the Photochemical grid.

    Results:
    sol_dict_new: dictionary of np.arrays
        This provides the matching solutions dictionary from photochem matching radius of planet, metallicity, tint, semi major axis, carbon to oxygen ratio, and kzz inputs.
    soled_dict_new: dictionary of np.arrays
        This provides the matching solutions dictionary (from chemical equilibrium) matching radius of planet, metallicity, tint, semi major axis, carbon to oxygen ratio, and kzz inputs.
    PT_list: 2D array
        This provides the matching pressure (in dynes/cm^2), temperature (Kelvin) from the Photochemical grid solution (not PICASO, since this involved some extrapolation and interpolation). 
    convergence_PC: 1D array
        This provides information on whether or not the Photochem model converged, using the binary equivalent of booleans (1=True, 0=False)
    convergence_TP: 1D array
        This provides information on whether or not the PICASO model used in Photochem was converged, using binary equivalent of booleans (1=True, 0=False)
        
    """
    gridvals_metal = [float(s) for s in gridvals[1]]
    planet_metallicity = float(log10_planet_metallicity)
    gridvals_dict = {'rad_plan':gridvals[0], 'planet_metallicity':gridvals_metal, 'tint':gridvals[2], 'semi_major':gridvals[3], 'ctoO_solar':gridvals[4], 'Kzz':gridvals[5]}

    with h5py.File(filename, 'r') as f:
        input_list = np.array([rad_plan, planet_metallicity, tint, semi_major, ctoO, Kzz])
        matches = (list(f['inputs'] == input_list))
        row_matches = np.all(matches, axis=1)
        matching_indicies = np.where(row_matches)

        matching_indicies_rad_plan = np.where(gridvals_dict['rad_plan'] == input_list[0])
        matching_indicies_metal = np.where(gridvals_dict['planet_metallicity'] == input_list[1])
        matching_indicies_tint = np.where(gridvals_dict['tint'] == input_list[2])
        matching_indicies_semi_major = np.where(gridvals_dict['semi_major'] == input_list[3])
        matching_indicies_ctoO = np.where(gridvals_dict['ctoO_solar'] == input_list[4])
        matching_indicies_kzz = np.where(gridvals_dict['Kzz'] == input_list[5])

        rad_plan_index, metal_index, tint_index, semi_major_index, ctoO_index, kzz_index = matching_indicies_rad_plan[0], matching_indicies_metal[0], matching_indicies_tint[0], matching_indicies_semi_major[0], matching_indicies_ctoO[0], matching_indicies_kzz[0]

        if matching_indicies[0].size == 0:
            print(f'A match given rad plan, planet metallicity, tint, semi-major axis, and ctoO does not exist')
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

        
def find_Photochem_sol(filepath='/mnt/c/Users/lily/Documents/NASAUWPostbac/MiniNeptuneGrid26_PostBac/results/Photochem_1D_updatop_paramext_K218b.h5', rad_plan=None, log10_planet_metallicity=None, tint=None, semi_major=None, ctoO=None, Kzz=None, grid_gridvals=Photochem_grid.get_gridvals_Photochem()):

    """
    Inputs:
    TBD
    """

    # This takes the inputs that define the grid
    gridshape = tuple(len(a) for a in grid_gridvals)

    print(f"Make sure your inputs are within the following ranges, rad_plan: {np.min(grid_gridvals[0])} - {np.max(grid_gridvals[0])} xEarth Radii, planet metallicity: {np.min((grid_gridvals[1]).astype(float))} - {np.max((grid_gridvals[1]).astype(float))} xsolar, tint: {np.min(grid_gridvals[2])} - {np.max(grid_gridvals[2])} K, semi_major: {np.min(grid_gridvals[3])} - {np.max(grid_gridvals[3])} AU, ctoO: {np.min(grid_gridvals[4])} - {np.max(grid_gridvals[4])}, kzz: {np.min(grid_gridvals[3])} - {np.max(grid_gridvals[3])} log10 of cm^2/s.")

    # Check to see if there is a solution that already exists
    sol_dict_new, soleq_dict_new, PT_list, convergence_PC, convergence_TP = find_Photochem_match(filename=filepath, rad_plan=rad_plan, log10_planet_metallicity=log10_planet_metallicity, tint=tint, semi_major=semi_major, ctoO=ctoO, Kzz=Kzz)

    if sol_dict_new is not None:
        print(f'All inputs chosen were directly on the grid!')
        print(f"This is for the input value of planet radius:{rad_plan}, metal:{float(log10_planet_metallicity)}, tint:{tint}, semi major:{semi_major}, ctoO: {ctoO}, log_Kzz: {Kzz}")
        comb_results = {}
        sol_dict = {}
        soleq_dict = {}
        for key, value in sol_dict_new.items():
            new_key = key + '_sol'
            sol_dict[new_key] = value
        for key, value in soleq_dict_new.items():
            new_key = key + '_soleq'
            soleq_dict[new_key] = value
        comb_results.update(sol_dict)
        comb_results.update(soleq_dict)
        #comb_results['pressure'] = PT_list[
        # comb_results['converged_PC'] = convergence_PC
        # comb_results['converged_TP'] = convergence_TP
        return comb_results
        

    else:

        print(f'Interpolating results...')
        
        # This notes the grid and associated inputs used to make it as the data
        PhotCh_grid = gridutils.GridInterpolator(filename=filepath, gridvals=grid_gridvals)

        # This interpolates the results based on the user input
        interp_results = {}
    
        # New grid values to interpolate
        user_gridvals = (rad_plan, log10_planet_metallicity, tint, semi_major, ctoO, Kzz)
    
        for key in PhotCh_grid.data.keys():
            if key.startswith('pressure'):
                interp_function_pressure = PhotCh_grid.make_interpolator(key=key, logspace=True)
                interp_results[key] = interp_function_pressure(user_gridvals)
            elif key.startswith('temperature'):
                interp_function_temperature = PhotCh_grid.make_interpolator(key=key, logspace=False)
                interp_results[key] = interp_function_temperature(user_gridvals)
            elif key.startswith('converged'):
                continue
            else:
                interp_function = PhotCh_grid.make_interpolator(key=key, logspace=True) # This is giving us photochem abundances
                interp_results[key] = interp_function(user_gridvals)
                
        return interp_results



