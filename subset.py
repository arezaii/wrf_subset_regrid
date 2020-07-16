import sys
import xarray as xr
import numpy as np
import pandas as pd
from varlist import var_list_parflow, var_list_wh
import glob
import os
import xesmf as xe
import dask
import write_pfb
import argparse
import datetime
import scipy
from tqdm import tqdm

descr = "subset and regrid (WRF) forcings data for (ParFlow) hydrologic model"


def is_valid_path(parser, arg):
    if not os.path.isdir(arg):
        parser.error("The path %s does not exist!" % arg)
    else:
        return arg  # return the arg


def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return open file handle


def parse_args(args):

    parser = argparse.ArgumentParser(descr)

    parser.add_argument("--input_path", "-i", dest="input_file_path", required=True,
                        help="path to the input files",
                        type=lambda x: is_valid_path(parser, x))

    parser.add_argument("--start_date", "-s", dest="start_date", required=True,
                        type=lambda x: datetime.datetime.strptime(x, '%m-%d-%Y'),
                        help="the starting date/time to subset from (inclusive)", )

    parser.add_argument("--end_date", "-e", dest="end_date", required=True,
                        type=lambda x: datetime.datetime.strptime(x, '%m-%d-%Y'),
                        help="the ending date/time to subset from (exclusive)")

    parser.add_argument("--output_dir", "-o", dest="out_dir", required=True,
                        help="the directory to write output to",
                        type=lambda x: is_valid_path(parser, x))

    parser.add_argument("--lat_lon_file", "-l", dest="lat_lon_file", required=True,
                        help="the list of lat/lon for the ParFlow grid",
                        type=lambda x: is_valid_file(parser, x))

    parser.add_argument("--dest_grid_nx", "-x", dest="nx", required=True,
                        help="the number of x cells in the destination grid", type=int)

    parser.add_argument("--dest_grid_ny", "-y", dest="ny", required=True,
                        help="the number of y cells in the destination grid", type=int)

    parser.add_argument("--day_number_start", "-d", dest="day_number", required=False,
                        default=0, type=int,
                        help="the counter value for the start day")

    parser.add_argument("--number_of_days", "-n", dest="num_days", required=False,
                        default=0, type=int,
                        help="the number of days in the timespan to subset")

    return parser.parse_args(args)


# TODO replace with arguments
# Arguments To Consider:
# INPUT FILE TYPE (WRF, NetCDF)
# WRF Specific stuff (list of vars to subset)
# OUTPUT_FILE_TYPE (ParFlow, PFB) (Hourly Forcings)
# TIMESTEPS_PER_FILE
# ParFlow Specific Operations (convert lat/lon file to grid)
# write output in PFB format
# hourly forcings data
# 1D or 3D output


# input_files = sorted(glob.glob(f'/scratch/arezaii/wrf_out/wy_{water_year}/d01/wrfout_d01_*'))


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
    f = lat_lon_file
    latlon = f.read().rstrip('\n').split('\n')
    f.close()
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


def make_dest_grid(coords, time_index):
    x_len = len(coords.get('lons')[0])
    y_len = len(coords.get('lats'))
    data = np.zeros((y_len, x_len))
    # print(data.shape)
    # da_lons = xr.DataArray(coords.get('lons'), dims=['lon'])
    # da_lats = xr.DataArray(coords.get('lats'), dims=['lat'])
    # print(da_lons,da_lats)
    ds = xr.Dataset({'data': (['y', 'x'], data)},
                    coords={'lon': (['y', 'x'], coords.get('lons')), 'lat': (['y', 'x'], coords.get('lats')),
                            'time': time_index})

    return ds

# Courtesy of https://github.com/JiaweiZhuang/xESMF/issues/15
def add_matrix_NaNs(regridder):
    X = regridder.weights
    M = scipy.sparse.csr_matrix(X)
    num_nonzeros = np.diff(M.indptr)
    M[num_nonzeros == 0, 0] = np.NaN
    regridder.weights = scipy.sparse.coo_matrix(M)
    return regridder

def create_out_ds(original_dataset):
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
    return ds_out

def build_regridder(ds_out, destination_grid):
    # create the regridder
    regrid = xe.Regridder(ds_out, destination_grid, 'bilinear', reuse_weights=True)
    regrid = add_matrix_NaNs(regrid)
    return regrid


def regrid_data(out_grid, regridder):
    varlist = [varname for varname in out_grid.keys() if varname != 'Times']
    # print(varlist)
    new_ds = xr.Dataset(data_vars=None, attrs=out_grid.attrs)
    for var in tqdm(varlist):
        # print(f'starting {var}')
        var_regrid = regridder(out_grid[var])
        new_ds[var] = ['Time', 'south_north', 'west_east'], dask.array.zeros_like(var_regrid)
        new_ds[var] = var_regrid
    new_ds.attrs = {'TITLE': 'REGRIDDED AND SUBSET OUTPUT FROM WRF V3.8.1 MODEL'}
    # new_ds.to_netcdf(os.path.join(out_dir, regrid_filename), mode='w')
    # print('done')
    return new_ds


def write_pfb_output(forcings_data, num_days, out_dir, start_day_num=0):
    varnames = ['APCP', 'DLWR', 'DSWR', 'Press', 'SPFH', 'Temp', 'UGRD', 'VGRD']

    hours_in_file = 24

    for var in tqdm(varnames):
        file_time_start = start_day_num * hours_in_file
        file_time_stop = file_time_start + hours_in_file

        start = 0  # start hour (inclusive)
        stop = hours_in_file  # end hour (exclusive)

        # start hour and stop hour for a day
        # range is number of days contained in forcings file
        for bulk_collect_times in range(0, num_days):
            write_pfb.pfb_write(np.transpose(np.nan_to_num(forcings_data[var].values[start:stop, :, :], nan=-9999.0),
                                             (2, 1, 0)),
                                os.path.join(out_dir, f'WRF.{var}.{file_time_start:06d}_to_{file_time_stop:06d}.pfb'),
                                float(0.0), float(0.0), float(0.0), float(1000.0), float(1000.0), float(1000.0))
            start = stop
            stop = stop + hours_in_file  # size of day in hours

            file_time_start = file_time_stop
            file_time_stop = file_time_stop + hours_in_file


def main():
    # parse the command line arguments
    args = parse_args(sys.argv[1:])

    # calculate number of input days to process
    if args.num_days == 0:
        days_to_load = (args.end_date - args.start_date).days - args.day_number
    else:
        days_to_load = args.num_days

    # alert the user about the job details
    print('Begin processing job:')
    for arg in vars(args):
        print(' {} {}'.format(arg, getattr(args, arg) or ''))
    print(f'Days of data to process: {days_to_load}')

    # generate the list of input files
    input_files = sorted(glob.glob(os.path.join(args.input_file_path, 'wrfout_d01_*')))

    # load the input files
    print('Begin opening the dataset...')
    input_files = input_files[args.day_number:days_to_load+args.day_number]
    ds_orig = xr.open_mfdataset(input_files,
                                drop_variables=var_list_parflow,
                                combine='nested',
                                concat_dim='Time')

    # subset the list of variables
    ds_subset = subset_variables(ds_orig)

    # R2 /scratch/arezaii/snake_river_shape_domain/input_files
    # filepath = '/home/arezaii/projects/parflow/snake_river_shape_domain/input_files/snake_river.latlon.txt'

    print('Begin reading destination coordinates...')
    coordinates = get_coords_from_lat_lon(args.lat_lon_file, args.nx, args.ny)

    # create the destination grid with lat/lon values
    dest_grid = make_dest_grid(coordinates, pd.date_range(args.start_date, periods=days_to_load))

    out_grid = create_out_ds(ds_subset)
    regrid = build_regridder(out_grid, dest_grid)

    print('Begin regridding data...')
    regridded_data = regrid_data(out_grid, regrid)

    print('Begin writing output files...')
    write_pfb_output(regridded_data, days_to_load, args.out_dir, args.day_number)
    print('Process complete!')


if __name__ == '__main__':
    main()
