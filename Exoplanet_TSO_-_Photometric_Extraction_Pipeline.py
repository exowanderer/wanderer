import numpy as np
# from numpy import zeros, nanmedian as median, nanmean as mean, nan
from tqdm import tqdm
from time import time, localtime
from sys import exit
from statsmodels.nonparametric import kde
from statsmodels.robust import scale
from socket import gethostname
from sklearn.preprocessing import StandardScaler
# import joblib
from scipy import stats
from scipy.special import erf
# from numpy import concatenate, pi, sqrt, ones, diag, inf, isnan, isfinite, array, nanmax
# from numpy import sort, linspace, indices, median, mean, std, empty, transpose, ceil
from photutils import CircularAperture, CircularAnnulus, aperture_photometry, findstars
import pandas as pd
# from pandas import DataFrame, read_csv, read_pickle, scatter_matrix
import os
# from os import environ, path, mkdir
# from numpy import int32 as npint, round as npround, nansum as sum, nanstd as std
# from numpy import min as npmin, max as npmax, zeros, arange, sum, float, isnan, hstack
from multiprocessing import cpu_count, Pool
from functools import partial
from glob import glob
from datetime import datetime
from astropy.modeling import models, fitting
from astropy.io import fits
from argparse import ArgumentParser

from wanderer import wanderer

ap = ArgumentParser()
ap.add_argument('-pn', '--planet_name', required=True, type=str,
                help='Directory Name for the Planet (i.e. GJ1214).')
ap.add_argument('-c', '--channel', required=True, type=str,
                help='Channel number string (i.e. ch1 or ch2).')
ap.add_argument('-ad', '--aor_dir', required=True, type=str,
                help='AOR director (i.e. r59217921).')
ap.add_argument('-sd', '--save_sub_dir', type=str, default='ExtracedData',
                help='Subdirectory inside Planet_Directory to store extracted outputs.')
ap.add_argument('-pd', '--planets_dir', type=str,
                default='/Research/Planets/', help='Location of planet directory name from $HOME.')
ap.add_argument('-ds', '--data_sub_dir', type=str, default='/data/raw/',
                help='Sub directory structure from $HOME/Planet_Name/THIS/aor_dir/..')
ap.add_argument('-dt', '--data_tail_dir', required=False,
                type=str, default='/big/', help='String inside AOR DIR.')
ap.add_argument('-ff', '--fits_format', type=str,
                default='bcd', help='Format of the fits files (i.e. bcd).')
ap.add_argument('-uf', '--unc_format', type=str,
                default='bunc', help='Format of the photometric noise files (i.e. bcd).')
ap.add_argument('-m', '--method', type=str, default='median',
                help='method for photmetric extraction (i.e. median).')
ap.add_argument('-t', '--telescope', type=str,
                default='Spitzer', help='Telescope: [Spitzer, Hubble, JWST].')
ap.add_argument('-ou', '--outputUnits', type=str, default='electrons',
                help='Units for the extracted photometry [electrons, muJ_per_Pixel, etc].')
ap.add_argument('-d', '--data_dir', type=str, default='',
                help='Set location of all `bcd` and `bunc` files: bypass previous setup.')
ap.add_argument('-v', '--verbose', type=bool,
                default=False, help='Print out normally irrelevent things.')

args = vars(ap.parse_args())

planetName = args['planet_name']
channel = args['channel']
aor_dir = args['aor_dir']
planetDirectory = args['planets_dir']
save_sub_dir = args['save_sub_dir']
data_sub_dir = args['data_sub_dir']
data_tail_dir = args['data_tail_dir']
fits_format = args['fits_format']
unc_format = args['unc_format']
method = args['method']
telescope = args['telescope']
outputUnits = args['outputUnits']
data_dir = args['data_dir']
verbose = args['verbose']

# from astroML.plotting          import hist
# from image_registration        import cross_correlation_shifts
# from matplotlib.ticker         import MaxNLocator
# from matplotlib                import style
# from least_asymmetry.asym      import actr, moments, fitgaussian
# from pylab                     import gcf, ion, figure, plot, imshow, scatter, legend, rcParams
# from seaborn                   import *


startFull = time()

print(
    '\n\n**Initializing Master Class for '
    'Exoplanet Time Series Observation Photometry**\n\n'
)


def clipOutlier2D(arr2D, n_sig=10):
    arr2D = arr2D.copy()
    medArr2D = median(arr2D, axis=0)
    sclArr2D = np.sqrt(((scale.mad(arr2D)**2.).sum()))
    outliers = abs(arr2D - medArr2D) > n_sig*sclArr2D
    inliers = abs(arr2D - medArr2D) <= n_sig*sclArr2D
    arr2D[outliers] = median(arr2D[inliers], axis=0)
    return arr2D

# As an example, Spitzer data is expected to be store in the directory structure:
#
# `$HOME/PLANET_DIRECTORY/PLANETNAME/data/raw/AORDIR/CHANNEL/bcd/`
#
# EXAMPLE:
#
# 1. On a Linux machine
# 2. With user `tempuser`,
# 3. And all Spitzer data is store in `Research/Planets`
# 4. The planet named `Happy-5b`
# 5. Observed during AOR r11235813
# 6. In CH2 (4.5 microns)
#
# The `loadfitsdir` should read as:
#   `/home/tempuser/Research/Planets/HAPPY5/data/raw/r11235813/ch2/bcd/`

# channel = 'ch2/'


dataSub = fits_format+'/'

if data_dir is '':
    data_dir = environ['HOME'] + planetDirectory + \
        planetName + data_sub_dir + channel + data_tail_dir
print('Current Data Dir: {}'.format(data_dir))

fileExt = '*{}.fits'.format(fits_format)
uncsExt = '*{}.fits'.format(unc_format)

loadfitsdir = data_dir + aor_dir + '/' + channel + '/' + dataSub

print('Directory to load fits files from: {}'.format(loadfitsdir))

nCores = cpu_count()
print('Found {} cores to process'.format(nCores))

fitsFilenames = glob(loadfitsdir + fileExt)
uncsFilenames = glob(loadfitsdir + uncsExt)

print('Found {} {}.fits files'.format(len(fitsFilenames), fits_format))
print('Found {} unc.fits files'.format(len(uncsFilenames)))

if len(fitsFilenames) == 0:
    raise ValueError('There are NO `{}.fits` files in the directory {}'.format(
        fits_format, loadfitsdir))
if len(uncsFilenames) == 0:
    raise ValueError('There are NO `{}.fits` files in the directory {}'.format(
        unc_format, loadfitsdir))

do_db_scan = False  # len(fitsFilenames*64) < 6e4
if do_db_scan:
    pass
else:
    print('There are too many images for a DB-Scan; i.e. >1e5 images')

header_test = fits.getheader(fitsFilenames[0])
print('\n\nAORLABEL:\t{}\nNum Fits Files:\t{}\nNum Unc Files:\t{}\n\n'.format(
    header_test['AORLABEL'], len(fitsFilenames), len(uncsFilenames)))

if verbose:
    print(fitsFilenames)
if verbose:
    print(uncsFilenames)

# Necessary Constants Spitzer
ppm = 1e6
y, x = 0, 1

yguess, xguess = 15., 15.   # Specific to Spitzer circa 2010 and beyond
# Specific to Spitzer Basic Calibrated Data
filetype = '{}.fits'.format(fits_format)

print('Initialize an instance of `wanderer` as `example_wanderer_median`\n')
example_wanderer_median = wanderer(
    fitsFileDir=loadfitsdir,
    filetype=filetype,
    telescope=telescope,
    yguess=yguess,
    xguess=xguess,
    method=method,
    nCores=nCores
)

example_wanderer_median.AOR = aor_dir
example_wanderer_median.planetName = planetName
example_wanderer_median.channel = channel

print('Load Data From Fits Files in ' + loadfitsdir + '\n')
example_wanderer_median.spitzer_load_fits_file(outputUnits=outputUnits)

print('**Double check for NaNs**')
is_nan_ = np.isnan(example_wanderer_median.imageCube)
med_image_cube = np.nanmedian(example_wanderer_median.imageCube)
example_wanderer_median.imageCube[is_nan_] = med_image_cube

print('**Identifier Strong Outliers**')
print('Find, flag, and NaN the "Bad Pixels" Outliers' + '\n')
example_wanderer_median.find_bad_pixels()

print(
    'Fit for All Centers: Flux Weighted, Gaussian Fitting, '
    'Gaussian Moments, Least Asymmetry\n'
)

# example_wanderer_median.fit_gaussian_centering()
example_wanderer_median.fit_flux_weighted_centering()
# example_wanderer_median.fit_least_asymmetry_centering()
# example_wanderer_median.fit_all_centering() # calling this calls least_asymmetry, which does not work :(

start = time()
example_wanderer_median.mp_lmfit_gaussian_centering(
    subArraySize=6,
    recheckMethod=None,
    median_crop=False
)

print(
    f'Operation took {time()-start} seconds with {nCores} cores')

if do_db_scan:
    print('DBScanning Gaussian Fit Centers')
    from sklearn.cluster import DBSCAN

    dbs = DBSCAN(n_jobs=-1, eps=0.2, leaf_size=10)
    dbsPred = dbs.fit_predict(example_wanderer_median.centering_GaussianFit)

    dbs_options = [k for k in range(-1, 100) if (dbsPred == k).sum()]
else:
    dbsPred = None
    dbs_options = []

# n_pix = 3
# stillOutliers = np.where(
#   abs(
#       example_wanderer_median.centering_GaussianFit - medGaussCenters
#   ) > 4*sclGaussCenterAvg
# )[0]
# print('There are {} outliers remaining'.format(len(stillOutliers)))

if do_db_scan:
    dbsClean = 0
    dbsKeep = (dbsPred == dbsClean)

# nCores = example_wanderer_median.nCores
start = time()
example_wanderer_median.mp_measure_background_circle_masked()
print('CircleBG took {} seconds with {} cores'.format(time() - start, nCores))

start = time()
example_wanderer_median.mp_measure_background_annular_mask()
print('AnnularBG took {} seconds with {} cores'.format(time() - start, nCores))

start = time()
example_wanderer_median.mp_measure_background_KDE_Mode()
print('KDEUnivBG took {} seconds with {} cores'.format(time() - start, nCores))

start = time()
example_wanderer_median.mp_measure_background_median_masked()
print('MedianBG took {} seconds with {} cores'.format(time() - start, nCores))

example_wanderer_median.measure_effective_width()
print(
    example_wanderer_median.effective_widths.mean(),
    np.sqrt(example_wanderer_median.effective_widths).mean()
)

print('Pipeline took {} seconds thus far'.format(time() - startFull))

print('Iterating over Background Techniques, Centering Techniques, Aperture Radii' + '\n')
# , 'Gaussian_Mom', 'FluxWeighted']#, 'LeastAsymmetry']
centering_choices = ['Gaussian_Fit']
# example_wanderer_median.background_df.columns
background_choices = ['AnnularMask']
staticRads = np.arange(1, 6, 0.5)  # [1.0 ]# aperRads = np.arange(1, 6,0.5)
varRads = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50]  # [None]#

med_quad_widths = np.median(example_wanderer_median.quadrature_widths)
vrad_dist = example_wanderer_median.quadrature_widths - med_quad_widths

vrad_dist = clipOutlier2D(vrad_dist, n_sig=5)

for staticRad in tqdm(staticRads, total=len(staticRads), desc='Static'):
    for varRad in tqdm(varRads, total=len(varRads), desc='Variable'):
        startMPFlux = time()
        example_wanderer_median.mp_compute_flux_over_time_varRad(
            staticRad,
            varRad,
            centering_choices[0],
            background_choices[0],
            useTheForce=True
        )

print('**Create Beta Variable Radius**')
# Gaussian_Fit_AnnularMask_rad_betaRad_0.0_0.0
example_wanderer_median.mp_compute_flux_over_time_betaRad()

print('Entire Pipeline took {} seconds'.format(time() - startFull))

if do_db_scan:
    print('DB_Scanning All Flux Vectors')
    example_wanderer_median.mp_DBScan_Flux_All()

print('Creating master Inliers Array')
# inlier_master = example_wanderer_median.inliers_Phots.values()
# inlier_master = array(list(inlier_master)).mean(axis=0) == 1.0

print('Extracting PLD Components')
example_wanderer_median.extract_PLD_components()

if do_db_scan:
    print('Running DBScan on the PLD Components')
    example_wanderer_median.mp_DBScan_PLD_All()

print('Saving `example_wanderer_median` to a set of pickles for various Image Cubes and the Storage Dictionary')

savefiledir_parts = [
    os.environ['HOME'] + planetDirectory,
    planetName+'/',
    save_sub_dir + '/',
    channel + '/',
    aor_dir + '/'
]

savefiledir = ''
for sfpart in savefiledir_parts:
    savefiledir = savefiledir + sfpart
    if not os.path.exists(savefiledir):
        os.mkdir(savefiledir)

# savefiledir         = environ['HOME']+planetDirectory+planetName+'/' + save_sub_dir + '/' + channel + '/' + aor_dir + '/'

saveFileNameHeader = planetName+'_' + aor_dir + '_Median'
saveFileType = '.joblib.save'

path_to_files = os.path.join(
    os.environ['HOME'],
    planetDirectory,
    planetName,
    save_sub_dir
)
if not os.path.exists(path_to_files):
    os.mkdir(path_to_files)

if not os.path.exists(savefiledir):
    print(f'Creating {savefiledir}')
    os.mkdir(savefiledir)

save_path = os.path.join(savefiledir, saveFileNameHeader, saveFileType)
print()
print(f'Saving to {save_path}')
print()

example_wanderer_median.save_data_to_save_files(
    savefiledir=savefiledir,
    saveFileNameHeader=saveFileNameHeader,
    saveFileType=saveFileType
)

print('Entire Pipeline took {time() - startFull} seconds')
