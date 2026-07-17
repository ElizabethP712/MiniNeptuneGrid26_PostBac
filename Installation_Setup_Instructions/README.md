# Installation & Setup Instructions

## Environments

Two conda environments are used in this project:

| Environment | picaso version | Use for |
|---|---|---|
| `subneptune` | Nick Wogan's fork (1abab28) | Photochem grid (`Photochem_grid_121625.py`) |
| `subneptune_nb_picaso` | Natasha Batalha's fork (4569a09) | New low-metal PICASO grid (`PICASO_Climate_grid_121625.py`) |

Both environments use `photochem=0.8.4`. Do **not** use `subneptune_picaso_updated` for grid scripts — its picaso 4.0.1 is incompatible with `inputs_climate(nstr=...)`.

The `picasofiles/` reference data (opacities, star spectra) lives at:
```
Installation_Setup_Instructions/picasofiles/
```
and is shared by both environments.

---

## Fresh install (Hyak or new machine)

`setup_automated.sh` builds `subneptune_nb_picaso` end-to-end: creates the conda env, installs picaso from the zip, copies reference data, downloads star spectra, and runs `setup_picaso.py` for opacity tables.

Run it from the `Installation_Setup_Instructions/` directory:

```sh
cd Installation_Setup_Instructions
bash setup_automated.sh
```

**When `setup_picaso.py` runs, it will pause and prompt for input three times.** Enter the following responses:

```
ck_tables
by-molecule
yes
```

To set up the original `subneptune` environment instead, run manually:

```sh
conda env create -f Installation_Setup_Instructions/environment_subneptune_current.yaml
conda activate subneptune

cd Installation_Setup_Instructions
wget https://github.com/Nicholaswogan/picaso/archive/1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37.zip
unzip 1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37.zip
cd picaso-1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37
python -m pip install . -v
cd ..
mkdir -p picasofiles
cp -r picaso-1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37/reference picasofiles/reference
rm -rf picaso-1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37
rm 1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37.zip

wget http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz
tar -xvzf synphot3.tar.gz
mv grp picasofiles/
rm synphot3.tar.gz

python setup_picaso.py
# When prompted, enter:
#   ck_tables
#   by-molecule
#   yes
```

---

## Local machine (clone from existing env)

If `subneptune` already works locally, cloning is faster than a full install:

```sh
# Build subneptune_nb_picaso by cloning subneptune and swapping picaso
conda create --name subneptune_nb_picaso --clone subneptune
conda activate subneptune_nb_picaso
pip install https://github.com/natashabatalha/picaso/archive/4569a09eea10e41f6f31a3a20d0bfb06b69d9ea2.zip

# Only needed if the new picaso version requires different opacity files:
# cd Installation_Setup_Instructions && python setup_picaso.py
```

---

## Recovering picasofiles/ only

If you deleted `picasofiles/` but your conda environments are still intact, run this from the `Installation_Setup_Instructions/` directory:

```sh
cd Installation_Setup_Instructions
conda activate subneptune_nb_picaso

# Download the picaso source zip and extract just the reference folder
# (reference/ is not installed into site-packages — it only comes from the source zip)
wget https://github.com/natashabatalha/picaso/archive/4569a09eea10e41f6f31a3a20d0bfb06b69d9ea2.zip
unzip 4569a09eea10e41f6f31a3a20d0bfb06b69d9ea2.zip
mkdir -p picasofiles
cp -r picaso-4569a09eea10e41f6f31a3a20d0bfb06b69d9ea2/reference picasofiles/reference
rm -rf picaso-4569a09eea10e41f6f31a3a20d0bfb06b69d9ea2
rm 4569a09eea10e41f6f31a3a20d0bfb06b69d9ea2.zip

# Download star spectra
wget http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz
tar -xvzf synphot3.tar.gz
mv grp picasofiles/
rm synphot3.tar.gz

# Re-download opacity tables into the reference dir
export picaso_refdata=$(pwd)/picasofiles/reference
python setup_picaso.py
# When prompted, enter:
#   ck_tables
#   by-molecule
#   yes
```

This skips recreating the conda environment entirely.

---

## Every session

The grid scripts set `picaso_refdata` and `PYSYN_CDBS` automatically when imported.
For interactive use (notebooks, one-off checks) run these from the **project root**:

```sh
export picaso_refdata=$(pwd)/Installation_Setup_Instructions/picasofiles/reference
export PYSYN_CDBS=$(pwd)/Installation_Setup_Instructions/picasofiles/grp/redcat/trds
```

---

## Verify the installation

After installing either environment, confirm picaso imported correctly and that `inputs_climate` accepts `nstr` (required by the grid scripts):

```sh
# Set env vars first (see above), then:
python -c "import inspect, picaso.justdoit as jdi; print(inspect.signature(jdi.inputs.inputs_climate))"
```

For `subneptune_nb_picaso` (Natasha Batalha's picaso), the output should include `rcb_guess` in the signature. For `subneptune` (Nick Wogan's picaso), it should include `nstr`.

---

## Environment files

| File | Description |
|---|---|
| `environment_subneptune_current.yaml` | Exported working state of the `subneptune` env |
| `environment_subneptune_nb_picaso.yaml` | Spec for the `subneptune_nb_picaso` env |
| `environment.yaml` | Original environment file (may be outdated) |
| `setup_automated.sh` | Automated setup script (currently builds `subneptune_nb_picaso`) |
| `setup_picaso.py` | Downloads picaso opacity tables via `picaso.data.get_data()` |
