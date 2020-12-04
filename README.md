# WRF Forcings Subset/Regrid Tool

Assist with regridding and subsetting WRF forcings in NetCDF format to ParFlow forcings in PFB format.

## Requirements

* Python >= 3.5
* esmpy
* numpy
* xesmf
* xarray
* dask
* pandas
* tqdm
* dask-jobqueue
* parflowio

## Usage

general usage:
```
python subset.py -i <path_to_inputs> -o <output_path> -x <NX> -y <NY> -s <start_date> -e <end_date> -l <lat_lon_file> -d <starting_day_number_for_outputs> -n <number_of_days_from_span> -r <domain_resolution>
```

See detailed example in `wrf_to_pf_subset2.bash` file.




