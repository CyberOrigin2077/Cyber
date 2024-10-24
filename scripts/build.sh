#!/bin/bash

# Get the directory where Miniconda/Anaconda is installed
CONDA_BASE=$(conda info --base)

# Source the conda.sh file
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Now you can use conda commands
conda create -n cyber python=3.10 -y
conda activate cyber
conda install pytorch==2.3.0 torchvision==0.18.0 cudatoolkit=11.1 -c pytorch -c nvidia -y

# Install the cyber package
pip install -e .
