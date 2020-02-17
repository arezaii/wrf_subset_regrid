import xarray as xr
import numpy as np
import pandas as pd
from varlist import var_list_parflow, var_list_wh
import time
import glob
import os
import copy
import xesmf as xe
import dask
import write_pfb

# TODO replace with arguments
water_year = 2015
out_dir = '/home/arezaii/wrf_out_for_parflow'
#out_filename = 'd01_parflow_vars.nc'
#input_files = sorted(glob.glob(f'/scratch/arezaii/wrf_out/wy_{water_year}/d01/wrfout_d01_*'))
input_files = sorted(glob.glob(f'/home/arezaii/wrf_history/vol11/wrf_out//wy_{water_year}/d01/wrfout_d01_*'))
#regrid_filename = 'd01_parflow_vars_regridded_1_day.nc'
#avg_filename = 'd01_parflow_vars_avg.nc'
#dest_grid = xr.open_dataset('/home/arezaii/git/sr_empty_netcdf_domain.nc')


# Decummulate Precipitation
# Thanks Charlie Becker!
def calc_precip(cum_precip, bucket_precip):
    total_precip = cum_precip + bucket_precip * 100.0
    PRCP = np.zeros(total_precip.shape)

    for i in np.arange(1, PRCP.shape[0]):
        PRCP[i, :, :] = total_precip[i, :, :].values - total_precip[i - 1, :, :].values

    return PRCP


def subset_variables(ds):
    # rename variables to what parflow clm expects
    ds = ds.rename(
        {'T2': 'Temp', 'Q2': 'SPFH', 'PSFC': 'Press', 'U10': 'UGRD', 'V10': 'VGRD', 'SWDOWN': 'DSWR', 'GLW': 'DLWR'})

    ds['APCP'] = ds['RAINNC']
    ds['APCP'].values = calc_precip(ds['RAINNC'], ds['I_RAINNC'])

    # drop columns not needed any longer
    ds = ds.drop(['RAINNC', 'I_RAINNC'])

    # set attributes of new column
    ds['APCP'].attrs = [('description', 'PRECIPITATION RATE'), ('units', 'mm/s'), ('MemoryOrder', 'XY')]

    return ds


def get_coords_from_lat_lon(lat_lon_file, x_size, y_size):

    f = open(lat_lon_file)
    latlon = f.read().rstrip('\n').split('\n')
    grid_dimensions = (x_size, y_size)
    coords = {}
    lats = []
    lons = []
    start_index = 0
    end_index = grid_dimensions[0]
    for i in range(0, grid_dimensions[1]):
        row_lats = np.zeros(grid_dimensions[0])
        row_lons = np.zeros(grid_dimensions[0])
        j = 0
        for point in latlon[start_index:end_index]:
            lat, lon = point.split(" ")
            row_lats[j] = lat
            row_lons[j] = lon
            j += 1
        if len(row_lons) > 0:
            lons.append(row_lons)
        if len(row_lats) > 0:
            lats.append(row_lats)
        start_index = end_index
        end_index = start_index + grid_dimensions[0]
    coords.update({'lats': lats})
    coords.update({'lons': lons})

    return coords


def make_dest_grid(coords, start_date, days):
    x_len = len(coords.get('lons')[0])
    y_len = len(coords.get('lats'))
    data = np.zeros((y_len, x_len))
    print(data.shape)
    # da_lons = xr.DataArray(coords.get('lons'), dims=['lon'])
    # da_lats = xr.DataArray(coords.get('lats'), dims=['lat'])
    # print(da_lons,da_lats)
    ds = xr.Dataset({'data': (['y', 'x'], data)},
                    coords={'lon': (['y', 'x'], coords.get('lons')), 'lat': (['y', 'x'], coords.get('lats')),
                            'time': pd.date_range(start_date, periods=days)})

    return ds


def build_regridder(original_dataset, destination_grid):
    ds_out = original_dataset.rename({'XLAT': 'lat', 'XLONG': 'lon'})
    # unsure why we're doing this exactly
    ds_out = ds_out.swap_dims({'Time': 'XTIME'})
    ds_out = ds_out.rename({'XTIME': 'Time'})
    # ds_out = ds_out.rename({'south_north' : 'lat', 'west_east':'lon'})
    # print(ds['lat'][0,:,:])

    # what's going on here?#
    # here we remove the time dim?
    ds_out['lat'] = ds_out['lat'][0, :, :]
    ds_out['lon'] = ds_out['lon'][0, :, :]

    # create the regridder
    regrid = xe.Regridder(ds_out, destination_grid, 'bilinear', reuse_weights=True)
    return ds_out, regrid


def regrid_data(out_grid, regridder):
    varlist = [varname for varname in out_grid.keys() if varname != 'Times']
    #print(varlist)
    new_ds = xr.Dataset(data_vars=None, attrs=out_grid.attrs)

    for var in varlist:
        # print(f'starting {var}')
        var_regrid = dask.delayed(regridder(out_grid[var]))
        new_ds[var] = dask.delayed(['Time', 'south_north', 'west_east'], dask.array.zeros_like(var_regrid))
        new_ds[var] = dask.delayed(var_regrid)
    new_ds.attrs = {'TITLE': 'REGRIDDED AND SUBSET OUTPUT FROM WRF V3.8.1 MODEL'}
    #new_ds.to_netcdf(os.path.join(out_dir, regrid_filename), mode='w')
    #print('done')
    return new_ds


def write_pfb_output(forcings_data, num_days):
    varnames = ['APCP', 'DLWR', 'DSWR', 'Press', 'SPFH', 'Temp', 'UGRD', 'VGRD']

    for var in varnames:
        start = 0 # start hour (inclusive)
        stop = 24 # end hour (exclusive)
        # start hour and stop hour for a day
        # range is number of days contained in forcings file
        for bulk_collect_times in range(0, num_days):
            dask.delayed(write_pfb.pfb_write(np.transpose(forcings_data[var].values[start:stop, :, :], (2, 1, 0)),
                                f'WRF.{var}.{start}_to_{stop}.pfb', float(0.0), float(0.0), float(0.0),
                                float(1000.0), float(1000.0), float(20.0)))
            start = stop
            stop = stop + 24 # size of day in hours

        dask.compute()



days_to_load = 4
# load the dataset
ds_orig = xr.open_mfdataset(input_files[:days_to_load], drop_variables=var_list_parflow, combine='nested', concat_dim='Time')

# subset the list of variables
ds_subset = subset_variables(ds_orig)

# get the coordinates into a dictionary
#local /home/arezaii/projects/parflow/snake_river_shape_domain/input_files
#R2 /scratch/arezaii/snake_river_shape_domain/input_files
filepath = '/home/arezaii/projects/parflow/snake_river_shape_domain/input_files/snake_river.latlon.txt'

# 704 = number of x values
# 736 = number of y values
coordinates = get_coords_from_lat_lon(filepath, 704, 736)

# create the destination grid with lat/lon values
dest_grid = make_dest_grid(coordinates, '2014-10-01', days_to_load)


out_grid, regrid  = build_regridder(ds_subset, dest_grid)

regridded_data = regrid_data(out_grid, regrid)

write_pfb_output(regridded_data, days_to_load)
