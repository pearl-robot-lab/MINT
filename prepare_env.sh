#!/bin/bash
eval "$(conda shell.bash hook)"
# Activate the environment
conda activate mint
# Install the package
pip install -e .
# install the extensions
python setup_ext.py install
pip install torchsummary
pip install imitation
pip install magical-il