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


def find_closest_pt(pt_x, pt_y, x, y):
    x_diff = np.abs(x - pt_x)
    y_diff = np.abs(y - pt_y)
    x_diff2 = x_diff**2
    y_diff2 = y_diff**2
    sum_diff = x_diff2 + y_diff2
    dist = sum_diff**(1/2)
    idx = np.unravel_index(dist.argmin(), dist.shape)
    #if distance is less than 2km away
    if np.min(dist) < 2000:
        return idx
    else:
        print("not close enough")
        return None

def get_centroid(center, x, y):
    pt_x = center[0]
    pt_y = center[1]
    idx = find_closest_pt(pt_x, pt_y, x, y)
    return idx

def plot_coords(lat, lon, idx, tif_fn):
    lat_coords = reshape(lat, idx)
    lon_coords = reshape(lon, idx)
    coords_layers = np.dstack([lat_coords, lon_coords])
    skimage.io.imsave(tif_fn, coords_layers)
    #print(coords_layers)


def get_extent(center):
    print(center)
    x0 = center[0] - 2.5e5
    y0 = center[1] - 2.5e5
    x1 = center[0] + 2.5e5
    y1 = center[1] + 2.5e5
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


def create_data(sat_fns, lat, lon, remove_goes_files=False):
    fn_head = 'G' + sat_fns[0].split('_G')[-1].split('_c')[0]+'_'+lat+'_'+lon

    lcc_str = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
    lcc_proj = pyproj.Proj(lcc_str)
    center = lcc_proj(lon,lat)
    extent = get_extent(center)
    bands = get_bands_from_fns(sat_fns)
    old_scn, scn = get_scn(sat_fns, bands, extent)

    lons, lats = scn[bands[0]].attrs['area'].get_lonlats()
    x = scn[bands[0]].coords['x']
    y = scn[bands[0]].coords['y']
    xx = np.tile(x, (len(y),1))
    yy = np.tile(y, (len(x),1)).T
    cent_idx = get_centroid(center, xx, yy)

    for band in bands:
        tif_band_fn_data = data_dir + 'data/{}_{}.tif'.format(band, fn_head)
        save_data(scn[band].data, cent_idx, tif_band_fn_data)

    tif_fn_coords = data_dir + 'coords/{}.tif'.format(fn_head)
    plot_coords(lats, lons, cent_idx, tif_fn_coords)
    if remove_goes_files:
        remove_goes(fn_head)
    #remove_tif(fn_head)
    return fn_head


def remove_tif(fn_head):
    s = fn_head.split('s')[1][:13]
    dt = pytz.utc.localize(datetime.strptime(s, '%Y%j%H%M%S'))
    tif_fn = glob.glob('cimss_true_color_sunz_rayleigh_{}{}{}_{}{}{}.tif'.format(dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'), dt.strftime('%H'), dt.strftime('%M'), dt.strftime('%S')))
    if tif_fn:
        os.remove(tif_fn[0])


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
