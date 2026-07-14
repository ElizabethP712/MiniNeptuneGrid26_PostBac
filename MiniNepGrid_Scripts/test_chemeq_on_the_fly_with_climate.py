import warnings
warnings.filterwarnings('ignore')

import picaso.justdoit as jdi
import astropy.units as u
import numpy as np
import pickle
import matplotlib.pyplot as plt

def run_climate(filename, opacity_ck, chem_method):

    # Initialize
    cl_run = jdi.inputs(calculation="planet", climate=True)

    # Gravity and tint
    tint = 200
    cl_run.gravity(gravity=4.5, gravity_unit=u.Unit('m/(s**2)')) # input gravity
    cl_run.effective_temp(tint) # input effective temperature

    # The star
    cl_run.star(
        opacity_ck, 
        temp=5326.6,
        metal=-0.03, 
        logg=4.38933, 
        radius=0.932, 
        radius_unit=u.R_sun,
        semi_major=0.0486, 
        semi_major_unit=u.AU
    )

    # Guess P-T
    nlevel = 91
    pt = cl_run.guillot_pt(Teq=1000, nlevel=nlevel, T_int=tint, p_bottom=2, p_top=-6)
    temp_guess = pt['temperature'].values 
    pressure = pt['pressure'].values

    # Convection
    nofczns = 1
    nstr_upper = 85
    nstr_deep = nlevel -2
    nstr = np.array([0,nstr_upper,nstr_deep,0,0,0])
    rfacv = 0.5

    # Set inputs
    cl_run.inputs_climate(
        temp_guess=temp_guess, 
        pressure=pressure, 
        nstr=nstr, 
        nofczns=nofczns, 
        rfacv=rfacv
    )

    # Set composition
    mh = 10
    cto_relative = 1
    cl_run.atmosphere(mh=mh, cto_relative=cto_relative, chem_method=chem_method)

    # Run model
    out = cl_run.climate(opacity_ck, save_all_profiles=True, with_spec=True)

    with open(filename,'wb') as f:
        pickle.dump(out, f)

def run():
    opacity_ck = jdi.opannection(method='resortrebin')
    run_climate('on_the_fly.pkl', opacity_ck, chem_method='on-the-fly')

def plot():

    with open('on_the_fly.pkl','rb') as f:
        on_the_fly = pickle.load(f)

    fig, ax = plt.subplots()

    species = ['H2','CO2','CO','O2','H2O']
    for i,sp in enumerate(species):
        ax.plot(on_the_fly['ptchem_df'][sp], on_the_fly['pressure'], c='C'+str(i), lw=3, ls='--', label=sp)
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e2,1e-6)
    ax.set_xlim(1e-8,1.1)
    ax.set_xlim()
    ax.set_ylabel('Pressure (bar)')
    ax.set_xlabel('Volume Mixing Ratio')
    ax.legend(bbox_to_anchor=(1.0,1.0),loc='upper left')

    ax1 = ax.twiny()
    ax1.plot(on_the_fly['temperature'], on_the_fly['pressure'], c='k', lw=3, ls='--')
    ax1.set_xlabel('Temperature (K)')
    ax1.legend(bbox_to_anchor=(1.0,0.0),loc='lower left')

    plt.savefig('example.png', dpi=150, bbox_inches='tight')
    
def main():
    run()
    plot()

if __name__ == '__main__':
    main()