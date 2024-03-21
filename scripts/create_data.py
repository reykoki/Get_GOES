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

def save_data(R, G, B, idx, fn_data, size=256):
    R = reshape(R, idx, size)
    G = reshape(G, idx, size)
    B = reshape(B, idx, size)
    layers = np.dstack([R, G, B])
    total = np.sum(R) + np.sum(G) + np.sum(B)
    skimage.io.imsave(fn_data, layers)
    return True

def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

def get_rand_center(idx, img_shape, size=256):
    d = int(size/4)
    x_o = random.randint(idx[0]-d, idx[0]+d)
    y_o = random.randint(idx[1]-d, idx[1]+d)
    return (x_o, y_o)

def find_closest_pt(pt_x, pt_y, x, y):
    x_diff = np.abs(x - pt_x)
    y_diff = np.abs(y - pt_y)
    x_diff2 = x_diff**2
    y_diff2 = y_diff**2
    sum_diff = x_diff2 + y_diff2
    dist = sum_diff**(1/2)
    idx = np.unravel_index(dist.argmin(), dist.shape)
    #if distance is less than 1km away
    if np.min(dist) < 1000:
        return idx
    else:
        print("not close enough")
        return None

def get_centroid(center, x, y, img_shape):
    pt_x = center[0]
    pt_y = center[1]
    idx = find_closest_pt(pt_x, pt_y, x, y)
    if idx:
        rand_idx = get_rand_center(idx, img_shape, size=256)
        return idx, rand_idx
    else:
        return None, None
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

def get_scn(fns, extent):
    print(fns)
    scn = Scene(reader='abi_l1b', filenames=fns)

    scn.load(['cimss_true_color_sunz_rayleigh'], generate=False)
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

def get_get_scn(sat_fns, extent, sleep_time=0):
    time.sleep(sleep_time)
    old_scn, tmp_scn = get_scn(sat_fns, extent)
    return old_scn, tmp_scn

def create_data(sat_fns, lat, lon, remove_goes_files=False):
    fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]

    lcc_str = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
    lcc_proj = pyproj.Proj(lcc_str)
    center = lcc_proj(lon,lat)
    extent = get_extent(center)
    old_scn, scn = get_get_scn(sat_fns, extent)

    x = scn['cimss_true_color_sunz_rayleigh'].coords['x']
    y = scn['cimss_true_color_sunz_rayleigh'].coords['y']
    lon, lat = scn['cimss_true_color_sunz_rayleigh'].attrs['area'].get_lonlats()

    corr_data = scn.save_dataset('cimss_true_color_sunz_rayleigh', compute=False)
    img_shape = scn['cimss_true_color_sunz_rayleigh'].shape

    R = corr_data[0][0]
    G = corr_data[0][1]
    B = corr_data[0][2]

    R = get_norm(R)
    G = get_norm(G)
    B = get_norm(B)

    xx = np.tile(x, (len(y),1))
    yy = np.tile(y, (len(x),1)).T

    cent, idx = get_centroid(center, xx, yy, img_shape)

    if cent:
        tif_fn_data = data_dir + 'data/{}.tif'.format(fn_head)
        tif_fn_coords = data_dir + 'coords/{}.tif'.format(fn_head)
        data_saved = save_data(R, G, B, idx, tif_fn_data)
        print(data_saved)
        if data_saved:
            plot_coords(lat, lon, idx, tif_fn_coords)
            if remove_goes_files:
                remove_goes(fn_head)
            remove_tif(fn_head)
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
    goes_fns = ['./data/goes/OR_ABI-L1b-RadC-M6C01_G16_s20232672101174_e20232672103547_c20232672103581.nc', './data/goes/OR_ABI-L1b-RadC-M6C02_G16_s20232672101174_e20232672103547_c20232672103575.nc', './data/goes/OR_ABI-L1b-RadC-M6C03_G16_s20232672101174_e20232672103547_c20232672103595.nc']
    lon = '-105.27'
    lat = '40.0'
    main(goes_fns, lat, lon)

