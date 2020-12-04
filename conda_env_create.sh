#!/bin/bash
conda create -n wrf_subset_regrid -c conda-forge --strict -y \
    python=3.7 \
    esmpy=7.1.0 \
    numpy xesmf \
    xarray \
    dask \
    pandas \
    tqdm

conda activate wrf_subset_regrid

python -m pip install parflowio
