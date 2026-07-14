import warnings
warnings.filterwarnings('ignore')
from picaso import photochem as picasochem
import os
import numpy as np

import os
from pathlib import Path

current_directory = Path.cwd()
references_directory_path = "Installation&Setup_Instructions/picasofiles/reference"
PYSYN_directory_path = "Installation&Setup_Instructions/picasofiles/grp/redcat/trds"
print(os.path.join(current_directory, references_directory_path))
print(os.path.join(current_directory, PYSYN_directory_path))

def main():

    thermofile = os.path.join(references_directory_path,'chemistry','thermo_data','thermo-sonora-component.yaml') 
    thermofile_photochem = 'photochem_thermo.yaml'
    gas = picasochem.EquilibriumChemistry(thermofile=thermofile, method='sonora-approx')

    T = 1650 # K
    P = 100.0 # bar
    log10mh = 3 # log10 metallicity relative to solar
    CtoO = 0.01 # C/O relative to solar

    # Equilibrium solve
    gases, condensates = gas.equilibrate_atmosphere(np.array([P]), np.array([T]), log10mh, CtoO)

    # Print results
    species = list(gases.keys())
    gases_arr = np.empty(len(species))
    for i,sp in enumerate(species):
        gases_arr[i] = gases[sp][0]

    print(f'{"NH3":<15} {gases['NH3'][0]:.1e}')
    print(f'{"O3":<15} {gases['O3'][0]:.1e}')

    print('Top 20 most abundant species')
    inds = np.argsort(gases_arr)[::-1]
    for i in range(25):
        j = inds[i]
        print(f'{species[j]:<15} {gases_arr[j]:.1e}')

if __name__ == "__main__":
    main()
