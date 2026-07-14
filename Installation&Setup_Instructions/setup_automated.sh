#!/bin/bash

# Define variables (adjust as needed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/environment.yaml"
CONDA_ENV_NAME="subneptune"
PICASO_DOWNLOAD_URL="https://github.com/Nicholaswogan/picaso/archive/1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37.zip"
PICASO_DOWNLOAD_FILE="1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37.zip"
PICASO_EXTRACT_DIR="picaso-1abab282f0fa8a7ca6c7ddb330cbcfdc08d75f37"
Starstuff_DOWNLOAD_URL="http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz"
Starstuff_DOWNLOAD_File="synphot3.tar.gz"

# --- Automation Commands ---

# 1. Create Conda Environment

# Define the path to your environment YAML file
ENV_FILE="$SCRIPT_DIR/environment.yaml"

# Check if the file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file not found at $ENV_FILE"
    exit 1
fi

# Create the Conda environment using the YAML file
echo "Creating Conda environment from $ENV_FILE..."
conda env create -f "$ENV_FILE"

# Optional: Check the exit status of the previous command
if [ $? -eq 0 ]; then
    echo "Environment created successfully."
    # Optional: Activate the environment after creation (this works for the current shell session)
    # echo "Activating the environment..."
    # conda activate <ENV_NAME> # Replace <ENV_NAME> with the name specified in your YAML file
else
    echo "Failed to create Conda environment."
fi

# 2. Activate Conda environment
# Note: Activating conda in a non-interactive script requires sourcing
# the conda initialization script.
echo "Activating Conda environment..."
source ~/.bashrc  # or ~/.zshrc, or path to your conda.sh
conda activate $CONDA_ENV_NAME

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment. Exiting."
    exit 1
fi

# 3. Use wget to download picaso
echo "Downloading picaso data..."
wget $PICASO_DOWNLOAD_URL -O $PICASO_DOWNLOAD_FILE

# 3. Unzip the downloaded file
echo "Unzipping picaso data..."
unzip $PICASO_DOWNLOAD_FILE

# 4. Change directory to the extracted data folder
cd $PICASO_EXTRACT_DIR

# 5. Run a Python script (assuming it's in the current directory now)
echo "Running Python script..."
python -m pip install . -v

# 6. Get the reference folder from the copied folder
cd ..
mkdir -p picasofiles
cp -r $PICASO_EXTRACT_DIR/reference picasofiles/reference
rm -rf $PICASO_EXTRACT_DIR
rm $PICASO_DOWNLOAD_FILE

# 7. Download star stuff
wget $Starstuff_DOWNLOAD_URL -O $Starstuff_DOWNLOAD_File

# 8. Unzip the star stuff
echo "Unzipping star stuff data"
tar -xvzf $Starstuff_DOWNLOAD_File 
mv grp picasofiles/
rm $Starstuff_DOWNLOAD_File

# 6. Export an environment variable (often used for configuration)
export picaso_refdata=$(pwd)"/picasofiles/reference/" 
export PYSYN_CDBS=$(pwd)"/picasofiles/grp/redcat/trds"
echo "Set picaso_refdata and PYSYN_CDBS paths (only persists for the duration of this script)"

# 7. Run the setup_picaso script to download opacities required
echo "Finish setting up picaso by downloading opacities"
python setup_picaso.py

echo "Automation script finished."