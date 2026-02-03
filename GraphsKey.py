import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import Photochem_grid_121625 as Photochem_grid
import PICASO_Climate_grid_121625 as PICASO_Climate_grid
import FilterGrids
from photochem.utils import stars
import pickle



def find_all_plotting_values(rad_plan=None, planet_metal=None, tint=None, semi_major=None, ctoO=None, kzz=None, calc_PT=True, calc_PhotCh=True, PT_filepath='/mnt/c/Users/lily/Documents/NASAUWPostbac/MiniNeptuneGrid26_PostBac/results/PICASO_climate_updatop_paramext_K218b.h5'):

    # Results from PICASO Grid (Interpolates if within range of grid)
    if calc_PT == True:
        PT_results_dict = FilterGrids.find_PT_sol(filepath=PT_filepath, rad_plan=rad_plan, log10_planet_metallicity=planet_metal, tint=tint, semi_major=semi_major, ctoO=ctoO)

        PT_list = [PT_results_dict['pressure'], PT_results_dict['temperature']]

    elif calc_PT == False:
        PT_list = None

    # Results from Photochem Grid (Interpolates if within range of grid)
    if calc_PhotCh == True:
        PhotCh_results_dict = FilterGrids.find_Photochem_sol(rad_plan=rad_plan, log10_planet_metallicity=planet_metal, tint=tint, semi_major=semi_major, ctoO=ctoO, Kzz=kzz)
    
        sol_dict = {}
        soleq_dict = {}
        for key in PhotCh_results_dict.keys():
            if key.endswith('_sol'):
                sol_dict[key] = PhotCh_results_dict[key].copy()
            elif key.endswith('_soleq'):
                soleq_dict[key] = PhotCh_results_dict[key].copy()
            else:
                continue
        
        PT_list_Photochem = [PhotCh_results_dict['pressure_sol'], PhotCh_results_dict['temperature_sol']]

    elif calc_PhotCh == False:
        PT_list_Photochem = None
        sol_dict = None
        soleq_dict = None
    
    return PT_list, sol_dict, soleq_dict, PT_list_Photochem
    

def plot_PT(rad_plan=None, planet_metal=None, tint=None, semi_major=None, ctoO=None, kzz=None, calc_PT=True, calc_PhotCh=True, PT_filepath='/mnt/c/Users/lily/Documents/NASAUWPostbac/MiniNeptuneGrid26_PostBac/results/PICASO_climate_updatop_paramext_K218b.h5'):
    
    PT_list, sol_dict, soleq_dict, PT_list_Photochem = find_all_plotting_values(rad_plan=rad_plan, planet_metal=planet_metal, tint=tint, semi_major=semi_major, ctoO=ctoO, kzz=kzz, calc_PT=calc_PT, calc_PhotCh=calc_PhotCh, PT_filepath=PT_filepath)
    
    # Plot the PT Profile from PICASO
    
    plt.figure(figsize=(10,10))
    plt.ylabel("Pressure [Bars]", fontsize=25)
    plt.xlabel('Temperature [K]', fontsize=25)
    plt.gca().invert_yaxis()
    #plt.ylim(500,1e-4)
    #plt.xlim(250,1750)
    
    plt.semilogy(PT_list[1],PT_list[0],color="r",linewidth=3, label='PICASO PT')

    if calc_PhotCh is True:
        
        plt.semilogy(PT_list_Photochem[1],PT_list_Photochem[0]/(10**6),color="blue",linewidth=3, linestyle='--', label='Photochem PT')
        print(f'Includes Photochemistry extension of PT profile.')

    else:
        print(f'Does not include Photochemistry extension of PT profile')
        
    plt.minorticks_on()
    plt.tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=23)
    plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

    plt.legend()

    #Tef = stars.equilibrium_temperature(total_flux*1361, 0)
    #plt.title(f"Teff = {Tef} K, log(g)=4.4")

    plt.show()
        

def plot_PT_Photochem(rad_plan=None, planet_metal=None, tint=None, semi_major=None, ctoO=None, kzz=None, calc_PT=True, calc_PhotCh=True):
    
    PT_list, sol_dict, soleq_dict, PT_list_Photochem = find_all_plotting_values(rad_plan=rad_plan, planet_metal=planet_metal, tint=tint, semi_major=semi_major, ctoO=ctoO, kzz=kzz, calc_PT=calc_PT, calc_PhotCh=calc_PhotCh)
    
    # Plot the Composition from Photochem
    fig, ax1 = plt.subplots(1,1,figsize=[5,4])
    #species = ['CO2','H2O','CH4','CO','NH3','H2','HCN','H2Oaer']
    species_sol = ['CO2_sol','H2O_sol','CH4_sol','CO_sol','NH3_sol','H2_sol','HCN_sol','H2Oaer_sol']
    species_soleq = ['CO2_soleq','H2O_soleq','CH4_soleq','CO_soleq','NH3_soleq','H2_soleq','HCN_soleq','H2Oaer_soleq']
    
    #for i,sp in enumerate(species):
    #    ax1.plot(sol_dict[sp],sol_dict['pressure']/1e6,label=sp, c='C'+str(i))
    #    ax1.plot(soleq_dict[sp],soleq_dict['pressure']/1e6, ls=':', c='C'+str(i), alpha=0.4)
    
    for i,sp in enumerate(species_soleq):
        ax1.plot(soleq_dict[sp],soleq_dict['pressure_soleq']/1e6, ls=':', c='C'+str(i), alpha=0.4)
    for i,sp in enumerate(species_sol):
        ax1.plot(sol_dict[sp],sol_dict['pressure_sol']/1e6, c='C'+str(i), label=sp)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e-8,1)
    ax1.set_ylim(1000,1e-7)
    ax1.grid(alpha=0.4)
    ax1.legend(ncol=1,bbox_to_anchor=(1,1.0),loc='upper left')
    ax1.set_xlabel('Mixing Ratio')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_yticks(10.0**np.arange(-6,2))
    #ax1.text(0.02, 1.04, 't = '+'%.2e'%pc.wrk.tn, \
    #    size = 15,ha='left', va='bottom',transform=ax1.transAxes)
    
    ax2 = ax1.twiny()
    ax2.set_xlabel('Temperature (K)')
    #print(((PT_list_Photochem[1])/(10**6)),  PT_list_Photochem[0])
    #print(np.flip(PT_list[1]), np.flip(PT_list[0]))
    ax2.plot(PT_list_Photochem[1], (PT_list_Photochem[0]/(10**6)), c='blue', ls='--',label='PT PICASO Profile')
    ax2.plot(np.flip(PT_list[1]), np.flip(PT_list[0]), c='black', ls='--',label='PT PICASO Profile')
    
    plt.title('K2-18b Around Sun (G-Star)')
    
    plt.legend()

    plt.show()
    
    # Save the plot as a PNG image
    # plt.savefig('output_graph_GStar_K218b.png', bbox_inches='tight')
    
    # Close the plot to free up memory (important if you're generating many plots)
    # plt.close()

def plot_solar_spectra(solar_file_name='GJ176_spectrum_278K.txt'):
    
    # Determine solar data:
    data_star = pd.read_csv(solar_file_name, sep='/t', engine='python')
    data_star[['Wavelength (nm)', 'Solar flux (mW/m^2/nm)']] = data_star['Wavelength(nm)      SolarFlux(mW/m^2/nm)'].str.split(expand=True)
    df_star = data_star.drop('Wavelength(nm)      SolarFlux(mW/m^2/nm)', axis=1)
    df_star['Wavelength (nm)'] = df_star['Wavelength (nm)'].astype(float)
    df_star['Solar flux (mW/m^2/nm)'] = df_star['Solar flux (mW/m^2/nm)'].astype(float)

    fig, ax1 = plt.subplots(1,1,figsize=[5,4])

    ax1.plot(df_star['Wavelength (nm)'], df_star['Solar flux (mW/m^2/nm)'], color='black', ls='--', linewidth=0.25)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Solar flux (mW/m^2/nm)')
    ax1.set_xlim(250,1400)
    
    # Plot vertical lines for EM spectral boundaries (UV - IR)
    
    # Fill the area between two vertical lines
    plt.axvspan(250, 400, color='lightblue', alpha=0.15, label='UV')
    
    # Visible
    plt.axvspan(380, 700, color='grey', alpha=0.15, label='Visible')
    
    # IR
    plt.axvspan(700, 1400, color='red', alpha=0.15, label='IR')
    
    plt.legend()
    plt.title('Star Spectra')
    plt.show()

# Just keeping this for the plot reference; will have to calculate the wno, albedo, and fpfs desired (no grid available for this in this repository). 
def plot_Reflected_Spectra(total_flux=None, planet_metal=None, tint=None, kzz=None, phase=None, calc_PT=False, calc_PhotCh=False, calc_RSM=True):

    PT_list, sol_dict, soleq_dict, wno, albedo, fpfs, PT_list_Photochem = find_all_plotting_values(total_flux=total_flux, planet_metal=planet_metal, tint=tint, kzz=kzz, phase=phase, calc_PT=calc_PT, calc_PhotCh=calc_PhotCh, calc_RSM=calc_RSM)

    fig,ax = plt.subplots(1,1,figsize=[5,4])

    ax.plot(1e4/wno, fpfs, c='k', lw=1.5, label='With Cloud')
    #for key in res:
    #    if key == 'all':
    #        continue
    #    _, fpfs1 = res[key]
    #    ax.fill_between(1e4/wno,fpfs,fpfs1,label=key,alpha=0.3)
    
    #wno_no_cloud_grid_150, fpfs_no_cloud_grid_150 = jdi.mean_regrid(wno_no_cloud, fpfs_no_cloud, R=150)
    ax.plot(1e4/wno, fpfs, c='red', lw=.05, ls='--')
    #ax.set_xlim(0.3,1)
    #ax.set_ylim(0e-10,10e-9)
    ax.set_ylabel('Planet-to-star flux ratio')
    ax.set_xlabel('Wavelength (microns)')
    ax.set_title('K2-18b Reflected Spectra around G-Star')
    ax.legend()
    #plt.savefig('old_opacities.pdf',bbox_inches='tight')
    plt.show()

# Just keeping this for the plot reference; will have to calculate the wno, albedo, and fpfs desired (no grid available for this in this repository). 
def plot_Reflected_Spectra_comp(total_flux=None, planet_metal=None, tint=None, kzz=None, phase=None, calc_PT=False, calc_PhotCh=False, calc_RSM=True):

    PT_list, sol_dict, soleq_dict, wno, albedo, fpfs, PT_list_Photochem = find_all_plotting_values(total_flux=total_flux, planet_metal=planet_metal, tint=tint, kzz=kzz, phase=phase, calc_PT=calc_PT, calc_PhotCh=calc_PhotCh, calc_RSM=calc_RSM)

    res1 = Reflected_Spectra.make_case_earth()
    wno_earth, fpfs_earth, albedo_earth = res1['all']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(1e4/wno, fpfs, c='k', lw=1.5, label='K218b/G-Star')
    ax1.plot(1e4/wno_earth, fpfs_earth, c='red', lw=1.5, label='Earth/G-Star')
    ax1.set_xlim(0.3,1)
    ax1.set_ylim(0e-10,10e-10)
    ax1.set_ylabel('Planet-to-star flux ratio')
    ax1.set_xlabel('Wavelength (microns)')
    ax1.set_title('Reflected Spectra w/ Earth Clouds')
    ax1.legend()

    ax2.plot(1e4/wno, fpfs, c='k', lw=1.5, label='K218b/G-Star')
    ax2.plot(1e4/wno_earth, fpfs_earth, c='red', lw=1.5, label='Earth/G-Star')
    ax2.set_xlim(0.3,1)
    ax2.set_ylim(0e-10,10e-9)
    ax2.set_ylabel('Planet-to-star flux ratio')
    ax2.set_xlabel('Wavelength (microns)')
    ax2.set_title('Reflected Spectra w/ Earth Clouds')
    ax2.legend()
    
    #plt.savefig('Earth_K218b_clouds_RS.pdf',bbox_inches='tight')
    
    #plt.savefig('Earth_K218b_clouds_RS_zoomedin.pdf',bbox_inches='tight')
    plt.show()
