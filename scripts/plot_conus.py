import pandas as pd
import sys
import pytz
from pyresample import create_area_def
import geopandas
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from satpy import Scene

data_dir = './data/'


def get_dt_str(dt):
    hr = dt.hour
    hr = str(hr).zfill(2)
    tt = dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)
    yr = dt.year
    return hr, dn, yr

def get_dt(input_dt):
    fmt = '%Y/%m/%d %H:%M'
    dt = datetime.strptime(input_dt, fmt)
    dt = pytz.utc.localize(dt)
    return dt

def get_fns(dt):
    hr, dn, yr = get_dt_str(dt)
    goes_dir = data_dir + 'goes/'
    fns = glob(goes_dir + '*C0[123]*_s{}{}{}*'.format(yr,dn,hr))
    print(fns)
    return fns

def get_rgb(corr_data):
    # corrected is in RGB order
    R = corr_data[0][0]
    G = corr_data[0][1]
    B = corr_data[0][2]
    RGB = np.dstack([R, G, B])
    return RGB


def plot_data(input_dt):
    dt = get_dt(input_dt)
    fns = get_fns(dt)

    scn = Scene(reader='abi_l1b', filenames=fns)

    composite = 'cimss_true_color_sunz_rayleigh'
    scn.load([composite])

    my_area = create_area_def(area_id='lccCONUS',
                              description='Lambert conformal conic for the contiguous US',
                              projection="+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs",
                              resolution=1000,
                              #entire US
                              area_extent=[-2.4e6, -1.5e6, 2.3e6, 1.4e6]
                              )

    new_scn = scn.resample(my_area, cache_dir='cache_dir')
    lcc_proj = new_scn[composite].attrs['area'].to_cartopy_crs()
    geos_proj = scn['C01'].attrs['area'].to_cartopy_crs()

    corr_data = new_scn.save_dataset('cimss_true_color_sunz_rayleigh', compute=False)
    RGB = get_rgb(corr_data)

    scan_start = new_scn[composite].attrs['start_time']
    scan_end = new_scn[composite].attrs['end_time']
    print('start of satellite scan: ', scan_start)
    print('end of satellite scan: ', scan_end)

    state_shape = './data/shape_files/cb_2018_us_state_500k.shp'

    states = geopandas.read_file(state_shape)
    states = states.to_crs(geos_proj)

    #t_0 = int("{:%H%M}".format(scan_start))
    #t_f = int("{:%H%M}".format(scan_end))
    t_0 = pytz.utc.localize(scan_start)
    t_f = pytz.utc.localize(scan_end)
    yr_day = int("{:%Y%j}".format(scan_end))
    geos_x = scn['C01'].coords['x']
    geos_y = scn['C01'].coords['y']

    states = states.to_crs(lcc_proj)

    x = new_scn[composite].coords['x']
    y = new_scn[composite].coords['y']

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1, projection=lcc_proj)

    ax.imshow(RGB, origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    states.boundary.plot(ax=ax, edgecolor='white', linewidth=.25)

    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig('conus.png')

    plt.show()

def main(input_dt):
    plot_data(input_dt)

if __name__ == '__main__':
    input_dt = sys.argv[1]
    main(input_dt)
