import matplotlib
matplotlib.use('Agg')  # non-interactive backend for terminal use

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import os

# --- Set up paths ---
current_directory = Path.cwd()
os.environ['picaso_refdata'] = str(current_directory / "Installation&Setup_Instructions/picasofiles/reference")
os.environ['PYSYN_CDBS']     = str(current_directory / "Installation&Setup_Instructions/picasofiles/grp/redcat/trds")

import picaso.justdoit as jdi
import astropy.units as u

# --- Opacity ---
opacity_path = str(
    current_directory /
    "Installation&Setup_Instructions/picasofiles/reference/opacities/opacities_photochem_0.1_250.0_R15000.db"
)
print(f"Loading opacities from: {opacity_path}")
OPACITY_EARTH = jdi.opannection(filename_db=opacity_path, wave_range=[0.3, 2.5])
print("Opacities loaded.")

# --- Helper functions ---
def _add_cloud_deck(case, ptop_bar, pbot_bar, w0=0.99, g0=0.85, opd=10):
    logdp = np.log10(pbot_bar) - np.log10(ptop_bar)
    case.clouds(w0=[w0], g0=[g0], p=[np.log10(pbot_bar)], dp=[logdp], opd=[opd])

def earth_spectrum(
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
    if atmosphere_kwargs is None:
        atmosphere_kwargs = {}

    earth = jdi.inputs()
    earth.phase_angle(phase, num_tangle=8, num_gangle=8)
    earth.gravity(radius=1, radius_unit=jdi.u.Unit('R_earth'),
                  mass=1, mass_unit=jdi.u.Unit('M_earth'))
    earth.approx(raman="none")
    earth.star(opannection=OPACITY_EARTH, temp=5778, logg=4.4, semi_major=1,
               metal=0.0, semi_major_unit=u.Unit('au'))

    P = np.logspace(-6, np.log10(p_surface_bar), nlevel)
    df_atmo = earth.TP_line_earth(P, nlevel=nlevel)
    df_pt_earth = pd.DataFrame({
        'pressure':    df_atmo['pressure'].values,
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
        _excl = {sp: ['line', 'continuum'] for sp in df_atmo_earth.columns if sp not in ['pressure', 'temperature']}
        earth.atmosphere(df=df_atmo_earth, exclude_mol=_excl)
    elif no_rayleigh:
        _excl = {sp: ['rayleigh'] for sp in df_atmo_earth.columns if sp not in ['pressure', 'temperature']}
        earth.atmosphere(df=df_atmo_earth, exclude_mol=_excl)
    else:
        earth.atmosphere(df=df_atmo_earth)

    earth.surface_reflect(0.1, OPACITY_EARTH.wno)

    if no_rayleigh:
        # Zero scattering causes 1/0 NaN; ghost cloud with negligible opacity prevents this
        _add_cloud_deck(earth, cloud_ptop_bar, cloud_pbot_bar, opd=1e-10)

    df_cldfree = earth.spectrum(OPACITY_EARTH, calculation='reflected', full_output=True)
    _add_cloud_deck(earth, cloud_ptop_bar, cloud_pbot_bar)  # real cloud overwrites ghost
    df_cld = earth.spectrum(OPACITY_EARTH, full_output=True)

    wno = df_cldfree['wavenumber']
    fpfs_cf  = df_cldfree['fpfs_reflected']
    albedo_cf = df_cldfree['albedo']
    fpfs_c   = df_cld['fpfs_reflected']
    albedo_c  = df_cld['albedo']

    _, albedo = jdi.mean_regrid(wno, (1 - cloud_frac) * albedo_cf + cloud_frac * albedo_c, R=150)
    wno, fpfs = jdi.mean_regrid(wno, (1 - cloud_frac) * fpfs_cf  + cloud_frac * fpfs_c,  R=150)

    return wno, fpfs, albedo, df_cld, df_cldfree

# --- Run test ---
print("\nRunning earth_spectrum with no_rayleigh=True ...")
df_mol_modern_earth = {
    "N2": 0.79, "O2": 0.21, "O3": 7e-7,
    "H2O": 3e-3, "CO2": 300e-6, "CH4": 1.7e-6,
}

wno, fpfs, albedo, df_cld, df_cldfree = earth_spectrum(
    df_mol_earth=df_mol_modern_earth,
    phase=0.0,
    no_rayleigh=True,
)

# --- Check 1: did it run? ---
print("CHECK 1 PASSED: earth_spectrum ran without error.")

# --- Diagnostics: which component has NaN? ---
cf_nan = np.sum(np.isnan(df_cldfree['albedo']))
cld_nan = np.sum(np.isnan(df_cld['albedo']))
print(f"\nDIAGNOSTICS:")
print(f"  Cloud-free albedo NaN count : {cf_nan} / {len(df_cldfree['albedo'])}")
print(f"  Cloudy albedo NaN count     : {cld_nan} / {len(df_cld['albedo'])}")

# --- Check 2: no NaN in final albedo ---
nan_count = np.sum(np.isnan(albedo))
if nan_count == 0:
    print("\nCHECK 2 PASSED: final albedo contains no NaN values.")
else:
    print(f"\nCHECK 2 FAILED: final albedo contains {nan_count} NaN values out of {len(albedo)}.")

print(f"Albedo stats: min={np.nanmin(albedo):.4f}, max={np.nanmax(albedo):.4f}, mean={np.nanmean(albedo):.4f}")

# --- Save results ---
save_path = current_directory / "test_earth_spectrum_results.pkl"
results = {
    'wno': wno,
    'fpfs': fpfs,
    'albedo': albedo,
    'df_cld': df_cld,
    'df_cldfree': df_cldfree,
}
with open(save_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved to: {save_path}")
print("Load in Jupyter with:")
print("  import pickle")
print(f"  with open('test_earth_spectrum_results.pkl', 'rb') as f: results = pickle.load(f)")
