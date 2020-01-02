#!/bin/bash
conda create -n wrf_subset_regrid -c conda-forge -y \
    python=3.7 \
    esmpy=7.1.0 \
    numpy xesmf \
    xarray \
    dask \
    pandas

conda activate wrf_subset_regrid

python setup.py install
