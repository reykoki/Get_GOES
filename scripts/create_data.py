import pyproj
import cartopy.crs as ccrs
import sys
from pyresample import create_area_def
from satpy import Scene
import os
import glob
import skimage
from datetime import datetime
import numpy as np
import time
import pytz
from datetime import timedelta

data_dir = './data/'

def save_data(data, band, fn_head):
    tif_fn_data = data_dir + 'data/{}_{}.tif'.format(band, fn_head)
    skimage.io.imsave(tif_fn_data, data)

def save_coords(lat, lon, fn_head):
    tif_fn_coords = data_dir + 'coords/{}.tif'.format(fn_head)
    coords_layers = np.dstack([lat, lon])
    skimage.io.imsave(tif_fn_coords, coords_layers)

def get_extent(lat, lon, res, img_size=256):
    lcc_str = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
    lcc_proj = pyproj.Proj(lcc_str)
    center = lcc_proj(lon,lat)
    dist = int(img_size/2*res)
    x0 = center[0] - dist
    y0 = center[1] - dist
    x1 = center[0] + dist
    y1 = center[1] + dist
    return [x0, y0, x1, y1]

def get_scn(fns, extent, to_load, res):
    scn = Scene(reader='abi_l1b', filenames=fns)
    scn.load(to_load, generate=False)
    projection = ccrs.LambertConformal(central_longitude=262.5,
                                   central_latitude=38.5,
                                   standard_parallels=(38.5, 38.5),
                                    globe=ccrs.Globe(semimajor_axis=6371229,
                                                     semiminor_axis=6371229))

    my_area = create_area_def(area_id='lccCONUS',
                              description='Lambert conformal conic for the contiguous US',
                              #projection="+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs",
                              projection=projection,
                              resolution=res,
                              area_extent=extent)

    new_scn = scn.resample(my_area)
    return new_scn

def save_composite(composite, scn, fn_head):
    corr_data = scn.save_dataset(composite, compute=False)
    R = corr_data[0][0]
    G = corr_data[0][1]
    B = corr_data[0][2]
    RGB = np.dstack([R, G, B])
    data_saved = save_data(RGB, composite, fn_head)

def create_composite(sat_fns, lat, lon, composites = ['cimss_true_color_sunz_rayleigh', 'airmass'], remove_goes_files=False, res=3000):
    fn_head = 'G' + sat_fns[0].split('_G')[-1].split('_c')[0]+'_'+lat+'_'+lon
    extent = get_extent(lat, lon, res)
    scn = get_scn(sat_fns, extent, composites, res)
    lon, lat = scn[composites[0]].attrs['area'].get_lonlats()
    for composite in composites:
        save_composite(composite, scn, fn_head)
    save_coords(lat, lon, fn_head)
    if remove_goes_files:
        remove_goes(fn_head)
    remove_tif(fn_head)
    return fn_head, scn

def get_bands_from_fns(fns):
    bands = []
    for fn in fns:
        band = fn.split('_')[1][-3:]
        bands.append(band)
    return bands

def create_raw(sat_fns, lat, lon, remove_goes_files=False, res=3000):
    fn_head = 'G' + sat_fns[0].split('_G')[-1].split('_c')[0]+'_'+lat+'_'+lon
    extent = get_extent(lat, lon, res)
    bands = get_bands_from_fns(sat_fns)
    print(bands.sort())
    print(bands)
    x = input('stop')
    scn = get_scn(sat_fns, extent, bands, res)
    lons, lats = scn[bands[0]].attrs['area'].get_lonlats()
    for band in bands:
        save_data(scn[band].data, band, fn_head)
    save_coords(lats, lons, fn_head)
    if remove_goes_files:
        remove_goes(fn_head)
    return fn_head, scn

# remove the tif files generated during composition
def remove_tif(fn_head):
    s = fn_head.split('s')[1][:13]
    dt = pytz.utc.localize(datetime.strptime(s, '%Y%j%H%M%S'))
    tif_fns = glob.glob('*_{}{}{}_{}{}{}.tif'.format(dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'), dt.strftime('%H'), dt.strftime('%M'), dt.strftime('%S')))
    if tif_fns:
        for fn in tif_fns:
            os.remove(fn)

# remove large satellite files
def remove_goes(fn_head):
    print("REMOVING GOES FILES")
    for fn in glob.glob(data_dir + 'goes/*{}*'.format(fn_head)):
        os.remove(fn)

def main(goes_fns, lat, lon):
    #fn_head, scn = create_composite(goes_fns, lat, lon)
    fn_head, scn = create_raw(goes_fns, lat, lon)

if __name__ == '__main__':
    input_dt = sys.argv[1]
    #input_dt = '2023/09/24 11:00'
    input_start = '20232672101174'
    sat_num = '16'
    goes_fns = glob.glob('./data/goes/*_G{}_*s{}*.nc'.format(sat_num, input_start))
    lon = '-105.27'
    lat = '40.0'
    main(goes_fns, lat, lon)
