import cartopy.crs as ccrs
import sys
import pytz
from pyresample import create_area_def
import geopandas
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from satpy import Scene
from helper_functions import *

data_dir = './data/'

def get_RGB(scn, composite):
    data = scn.save_dataset(composite[0], compute=False)
    R = data[0][0]
    G = data[0][1]
    B = data[0][2]
    # reorder before computing for plotting
    RGB = np.dstack([R, G, B])
    RGB = RGB.compute()
    return RGB

def get_scn(fns, to_load, extent, res, proj, reader='abi_l1b'):
    scn = Scene(reader=reader, filenames=fns)
    scn.load(to_load, generate=False)
    my_area = create_area_def(area_id='my_area',
                              projection=proj,
                              resolution=res,
                              area_extent=extent
                              )
    new_scn = scn.resample(my_area)
    return new_scn

def plot_CONUS(data, crs):
    states = get_states(crs)
    fig = plt.figure(figsize=(15, 12))
    ax = plt.axes(projection=crs)
    plt.imshow(data, transform=crs, extent=crs.bounds, origin='upper')
    states.boundary.plot(ax=ax, edgecolor='white', linewidth=.25)
    plt.axis('off')
    fig.tight_layout()
    #plt.savefig('conus.png')
    plt.show()

def get_RGB_CONUS_scn(input_dt):
    dt = get_dt(input_dt)
    fns = get_fns_from_dt(dt)
    composite = ['cimss_true_color_sunz_rayleigh']
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5, central_latitude=38.5, standard_parallels=(38.5, 38.5), globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229))
    extent=[-2.4e6, -1.5e6, 2.3e6, 1.4e6] # CONUS
    res = 5000 # 5km resolution
    scn = get_scn(fns, composite, extent, res, lcc_proj)
    RGB = get_RGB(scn, composite)
    crs = scn[composite[0]].attrs['area'].to_cartopy_crs()
    plot_CONUS(RGB, crs)
    return scn

def main(input_dt):
    get_RGB_CONUS_scn(input_dt)

if __name__ == '__main__':
    input_dt = sys.argv[1]
    main(input_dt)
