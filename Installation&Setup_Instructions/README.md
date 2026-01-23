
# First time setup

```sh
conda env create -f environment.yaml
conda activate subneptune

# Install picaso
wget https://github.com/Nicholaswogan/picaso/archive/1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37.zip
unzip 1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37.zip
cd picaso-1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37
python -m pip install . -v
# Get reference
cd ../
mkdir picasofiles
cp -r picaso-1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37/reference picasofiles/reference
rm -rf picaso-1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37
rm 1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37.zip

# Get the star stuff
wget http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz
tar -xvzf synphot3.tar.gz
mv grp picasofiles/
rm synphot3.tar.gz

# setup
export picaso_refdata=$(pwd)"/picasofiles/reference/" 
export PYSYN_CDBS=$(pwd)"/picasofiles/grp/redcat/trds"

```

# Every other time

```sh
conda activate subneptune
export picaso_refdata=$(pwd)"/picasofiles/reference/"
export PYSYN_CDBS=$(pwd)"/picasofiles/grp/redcat/trds"
```
