import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import skimage
from datetime import datetime

def get_lat_lon(coords_fn):
    lat_lon = skimage.io.imread(coords_fn, plugin='tifffile')
    lat = lat_lon[:,:,0]
    lon = lat_lon[:,:,1]
    return lat[::63,0], lon[-1,::63]

def get_data(fn, data_loc="./sample_data/"):
    data_fn = glob(data_loc + "data/" + fn)[0]
    truth_fn = glob(data_loc + "truth/" + fn)[0]
    RGB = skimage.io.imread(data_fn, plugin='tifffile')
    truths = skimage.io.imread(truth_fn, plugin='tifffile')
    lat, lon = get_lat_lon(fn,data_loc)
    return RGB, truths, lat, lon

def get_datetime_from_fn(fn):
    start = fn.split('_s')[-1].split('_e')[0]
    start_dt = datetime.strptime(start, '%Y%j%H%M%S')
    start_readable = start_dt.strftime('%Y/%m/%d %H:%M UTC')
    return start_readable

def get_mesh(num_pixels):
    x = np.linspace(0,num_pixels-1,num_pixels)
    y = np.linspace(0,num_pixels-1,num_pixels)
    X, Y = np.meshgrid(x,y)
    return X,Y


def plot_band(fn, data_loc="./data/"):
    data_fn = data_loc + "data/" + fn
    coords_fn = data_loc + "coords/" + 'G' + data_fns[0].split('_G')[-1]
    band = fn.split('_')[0]
    band_data = skimage.io.imread(data_fn, plugin='tifffile')
    lat, lon = get_lat_lon(coords_fn)
    plt.figure(figsize=(8, 6),dpi=100)
    plt.imshow(band_data, cmap='Greys_r')
    plt.yticks(np.linspace(0,255,5), np.round(lat,2), fontsize=12)
    plt.ylabel('latitude (degrees)', fontsize=16)
    plt.xticks(np.linspace(0,255,5), np.round(lon,2), fontsize=12)
    plt.xlabel('longitude (degrees)', fontsize=16)
    plt.title(band,fontsize=24)
    plt.tight_layout(pad=0)
    plt.show()


def main(data_fns):
    for data_fn in data_fns:
        plot_band(data_fn)


#if __name__ == '__main__':
    #data_fns = sys.argv[1]
    #main(data_fns)

if __name__ == '__main__':
    data_fns = ['C01_G16_s20232672101174_e20232672103547_40.0_-105.27.tif',  'C03_G16_s20232672101174_e20232672103547_40.0_-105.27.tif', 'C02_G16_s20232672101174_e20232672103547_40.0_-105.27.tif']
    main(data_fns)


