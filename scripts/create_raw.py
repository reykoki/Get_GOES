import pyproj
import sys
from pyresample import create_area_def
import geopandas
import pandas as pd
from satpy import Scene
import os
import random
import glob
import skimage
from datetime import datetime
import numpy as np
import time
import pytz
import shutil
import wget
from datetime import timedelta

data_dir = './data/'

def reshape(A, idx, size=256):
    #print('before reshape: ', np.sum(A))
    d = int(size/2)
    A =A[idx[0]-d:idx[0]+d, idx[1]-d:idx[1]+d]
    #print('after reshape: ', np.sum(A))
    return A

def get_norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def save_data(data, idx, fn_data, size=256):
    data = reshape(data, idx, size)
    skimage.io.imsave(fn_data, data)

def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def save_data(data, band, fn_head):
    tif_fn_data = data_dir + 'data/{}_{}.tif'.format(band, fn_head)
    skimage.io.imsave(tif_fn_data, data)

def save_coords(lat, lon, fn_head):
    tif_fn_coords = data_dir + 'coords/{}.tif'.format(fn_head)
    coords_layers = np.dstack([lat, lon])
    skimage.io.imsave(tif_fn_coords, coords_layers)

def get_extent(lat, lon):
    lcc_str = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
    lcc_proj = pyproj.Proj(lcc_str)
    center = lcc_proj(lon,lat)
    x0 = center[0] - 1.28e5
    y0 = center[1] - 1.28e5
    x1 = center[0] + 1.28e5
    y1 = center[1] + 1.28e5
    return [x0, y0, x1, y1]


def get_bands_from_fns(fns):
    bands = []
    for fn in fns:
        band = fn.split('_')[1][-3:]
        bands.append(band)
    return bands

def get_scn(fns, bands, extent):
    scn = Scene(reader='abi_l1b', filenames=fns)
    scn.load(bands, generate=False)
    my_area = create_area_def(area_id='lccCONUS',
                              description='Lambert conformal conic for the contiguous US',
                              projection="+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs",
                              resolution=1000,
                              #full us
                              #area_extent=[-2.4e6, -1.5e6, 2.3e6, 1.4e6])
                              #western us
                              #area_extent=[-2.4e6, -1.5e6, 3.5e5, 1.4e6])
                              area_extent=extent)

    new_scn = scn.resample(my_area)
    return scn, new_scn

def get_fn_head(band, sat_fns, lat, lon):
    matching_band_fn = [s for s in sat_fns if band in s]
    fn_head = matching_band_fn.split(band).split('_c')+'_'+lat+'_'+lon

# remove large satellite files and the tif files created during corrections
def remove_goes(fn_head):
    print("REMOVING GOES FILES")
    for fn in glob.glob(data_dir + 'goes/*{}*'.format(fn_head)):
        os.remove(fn)


def main(goes_fns, lat, lon):
    fn_head = create_data(goes_fns, lat, lon)


#if __name__ == '__main__':
#E    goes_fns = sys.argv[1]
 #   main(input_dt)

if __name__ == '__main__':
    input_dt = '2023/09/24 11:00'
    lon = '-105.27'
    lat = '40.0'
    input_start = '20232672101174'
    sat_num = '16'
    goes_fns = glob.glob('./data/goes/*_G{}_*s{}*.nc'.format(sat_num, input_start))
    if goes_fns:
        main(goes_fns, lat, lon)

    #goes_fns = ['./data/goes/OR_ABI-L1b-RadC-M6C01_G16_s20232672101174_e20232672103547_c20232672103581.nc', './data/goes/OR_ABI-L1b-RadC-M6C02_G16_s20232672101174_e20232672103547_c20232672103575.nc', './data/goes/OR_ABI-L1b-RadC-M6C03_G16_s20232672101174_e20232672103547_c20232672103595.nc']
