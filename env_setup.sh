#!/bin/bash

tmp_env=$1

# If environment is given, activate it, otherwise print info message.
if [ ! -z "$tmp_env" ]
then
	source activate $tmp_env
else
	echo "INFO: A source-able environement can be provided as an argument to this script."
fi

# Display the target environment
echo "Installing the relevant packages to the following environment after 5s:"
which python

# Wait for ten seconds
sleep 5

# Install packages relevant for core function
conda install python=3.6.5
# conda install -c conda-forge hdbscan=0.8.13=py36_0
conda install numba

# These packages are solely for demonstration purposes and the example notebook,
#   and might be excluded on demand.
conda install ipykernel
conda install matplotlib
