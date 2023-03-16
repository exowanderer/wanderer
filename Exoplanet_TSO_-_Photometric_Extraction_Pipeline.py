import numpy as np
import os
import sys

from astropy.io import fits
from glob import glob
from multiprocessing import cpu_count
from time import time
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from statsmodels.robust import scale

# TODO: make this more direct
from wanderer.wanderer import Wanderer
from wanderer.utils import command_line_inputs, clipOutlier2D

clargs = command_line_inputs(check_defaults=True)

planet_name = clargs.planet_name
channel = clargs.channel
aor_dir = clargs.aor_dir
planets_dir = clargs.planets_dir
save_sub_dir = clargs.save_sub_dir
data_sub_dir = clargs.data_sub_dir
data_tail_dir = clargs.data_tail_dir
fits_format = clargs.fits_format
unc_format = clargs.unc_format
method = clargs.method
telescope = clargs.telescope
output_units = clargs.output_units
data_dir = clargs.data_dir
num_cores = clargs.num_cores
verbose = clargs.verbose

startFull = time()
savefiledir_parts = [
    planets_dir,
    # planet_name,
    save_sub_dir,
    channel,
    aor_dir
]

print('Creating save file directory')
savefiledir = ''
for sfpart in savefiledir_parts:
    savefiledir = os.path.join(savefiledir, sfpart)
    if not os.path.exists(savefiledir):
        os.mkdir(savefiledir)

print(
    '\n\n**Initializing Master Class for '
    'Exoplanet Time Series Observation Photometry**\n\n'
)


# As an example, Spitzer data is expected to be store in the directory structure:
#
# `PLANET_DIRECTORY/data/raw/AORDIR/CHANNEL/bcd/`
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
#   `./Research/Planets/HAPPY5/data/raw/r11235813/ch2/bcd/`


dataSub = f'{fits_format}/'

if data_dir == '':
    data_dir = os.path.join(
        planets_dir,
        # planet_name,
        data_sub_dir,
        channel,
        data_tail_dir
    )

print(f'Current Data Dir: {data_dir}')

fileExt = f'*{fits_format}.fits'
uncsExt = f'*{unc_format}.fits'

loadfitsdir = data_dir + aor_dir + '/' + channel + '/' + dataSub

print(f'Directory to load fits files from: {loadfitsdir}')

print(f'Found {num_cores} cores to process')

fitsFilenames = glob(loadfitsdir + fileExt)
uncsFilenames = glob(loadfitsdir + uncsExt)

n_fitsfiles = len(fitsFilenames)
n_uncfiles = len(uncsFilenames)
print(f'Found {n_fitsfiles} {fits_format}.fits files')
print(f'Found {n_uncfiles} unc.fits files')


if len(fitsFilenames) == 0:
    raise ValueError(
        f'There are NO `{fits_format}.fits` files '
        f'in the directory {loadfitsdir}'
    )
if len(uncsFilenames) == 0:
    raise ValueError(
        f'There are NO `{unc_format}.fits` files '
        f'in the directory {loadfitsdir}'
    )


do_db_scan = False  # len(fitsFilenames*64) < 6e4
if not do_db_scan:
    print('There are too many images for a DB-Scan; i.e. >1e5 images')

header_test = fits.getheader(fitsFilenames[0])
print(
    f'\n\nAORLABEL:\t{header_test["AORLABEL"]}'+'\n'
    f'Num Fits Files:\t{len(fitsFilenames)}'+'\n'
    f'Num Unc Files:\t{len(uncsFilenames)}\n\n'
)

if verbose:
    print(fitsFilenames)
if verbose:
    print(uncsFilenames)

# Necessary Constants Spitzer
ppm = 1e6
y, x = 0, 1

yguess, xguess = 15., 15.   # Specific to Spitzer circa 2010 and beyond
# Specific to Spitzer Basic Calibrated Data
filetype = f'{fits_format}.fits'

print('Initialize an instance of `Wanderer` as `planetname_wanderer_median`\n')
planetname_wanderer_median = Wanderer(
    fitsFileDir=loadfitsdir,
    filetype=filetype,
    telescope=telescope,
    yguess=yguess,
    xguess=xguess,
    method=method,
    num_cores=num_cores
)

planetname_wanderer_median.AOR = aor_dir
planetname_wanderer_median.planet_name = planet_name
planetname_wanderer_median.channel = channel

print(f'Load Data From Fits Files in {loadfitsdir}')
planetname_wanderer_median.spitzer_load_fits_file(output_units=output_units)

print('**Double check for NaNs**')
is_nan_ = np.isnan(planetname_wanderer_median.imageCube)
med_image_cube = np.nanmedian(planetname_wanderer_median.imageCube)
planetname_wanderer_median.imageCube[is_nan_] = med_image_cube

print('**Identifier Strong Outliers**')
print('Find, flag, and NaN the "Bad Pixels" Outliers')
planetname_wanderer_median.find_bad_pixels()

print(
    'Fit for All Centers: Flux Weighted, Gaussian Fitting, '
    'Gaussian Moments, Least Asymmetry\n'
)

# planetname_wanderer_median.fit_gaussian_centering()
planetname_wanderer_median.fit_flux_weighted_centering()
# planetname_wanderer_median.fit_least_asymmetry_centering()
# planetname_wanderer_median.fit_all_centering() # calling this calls least_asymmetry, which does not work :(

start = time()
planetname_wanderer_median.mp_lmfit_gaussian_centering(
    subArraySize=6,
    recheckMethod=None,
    median_crop=False
)

print(f'Operation took {time()-start} seconds with {num_cores} cores')

if do_db_scan:
    print('DBScanning Gaussian Fit Centers')

    dbs = DBSCAN(n_jobs=-1, eps=0.2, leaf_size=10)
    dbsPred = dbs.fit_predict(planetname_wanderer_median.centering_GaussianFit)

    dbs_options = [k for k in range(-1, 100) if (dbsPred == k).sum()]
else:
    dbsPred = None
    dbs_options = []

# n_pix = 3
# stillOutliers = np.where(
#   abs(
#       planetname_wanderer_median.centering_GaussianFit - medGaussCenters
#   ) > 4*sclGaussCenterAvg
# )[0]
# print(f'There are {len(stillOutliers)} outliers remaining')

if do_db_scan:
    dbsClean = 0
    dbsKeep = (dbsPred == dbsClean)

# num_cores = planetname_wanderer_median.num_cores
start = time()
planetname_wanderer_median.mp_measure_background_circle_masked()
print(f'CircleBG took {time() - start} seconds with {num_cores} cores')

start = time()
planetname_wanderer_median.mp_measure_background_annular_mask()
print(f'AnnularBG took {time() - start} seconds with {num_cores} cores')

start = time()
planetname_wanderer_median.mp_measure_background_KDE_Mode()
print(f'KDEUnivBG took {time() - start} seconds with {num_cores} cores')

start = time()
planetname_wanderer_median.mp_measure_background_median_masked()
print(f'MedianBG took {time() - start} seconds with {num_cores} cores')

planetname_wanderer_median.measure_effective_width()
print(
    f"{planetname_wanderer_median.effective_widths.mean()}",
    f"{np.sqrt(planetname_wanderer_median.effective_widths).mean()}"
)

print(f'Pipeline took {time() - startFull} seconds thus far')

print(
    'Iterating over Background Techniques, Centering Techniques, '
    'Aperture Radii' + '\n'
)
# , 'Gaussian_Mom', 'FluxWeighted']#, 'LeastAsymmetry']
centering_choices = ['Gaussian_Fit']

# planetname_wanderer_median.background_df.columns
background_choices = ['AnnularMask']
staticRads = np.arange(1, 6, 0.5)  # [1.0 ]  # aperRads = np.arange(1, 6,0.5)
varRads = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50]  # [None]#

med_quad_widths = np.nanmedian(planetname_wanderer_median.quadrature_widths)
vrad_dist = planetname_wanderer_median.quadrature_widths - med_quad_widths

vrad_dist = clipOutlier2D(vrad_dist, n_sig=5)

for staticRad in tqdm(staticRads, total=len(staticRads), desc='Static'):
    for varRad in tqdm(varRads, total=len(varRads), desc='Variable'):
        startMPFlux = time()
        planetname_wanderer_median.mp_compute_flux_over_time_varRad(
            staticRad,
            varRad,
            centering_choices[0],
            background_choices[0],
            useTheForce=True
        )

print('**Create Beta Variable Radius**')
# Gaussian_Fit_AnnularMask_rad_betaRad_0.0_0.0
planetname_wanderer_median.mp_compute_flux_over_time_betaRad()

print(f'Entire Pipeline took {time() - startFull} seconds')

if do_db_scan:
    print('DB_Scanning All Flux Vectors')
    planetname_wanderer_median.mp_DBScan_Flux_All()

print('Creating master Inliers Array')
# inlier_master = planetname_wanderer_median.inliers_Phots.values()
# inlier_master = array(list(inlier_master)).mean(axis=0) == 1.0

print('Extracting PLD Components')
planetname_wanderer_median.extract_PLD_components()

if do_db_scan:
    print('Running DBScan on the PLD Components')
    planetname_wanderer_median.mp_DBScan_PLD_All()

print(
    'Saving `planetname_wanderer_median` to a set of pickles for various '
    'Image Cubes and the Storage Dictionary'
)

save_name_header = f'{planet_name}_{aor_dir}_median'
save_file_type = '.joblib.save'

path_to_files = os.path.join(
    planets_dir,
    # planet_name,
    save_sub_dir
)
if not os.path.exists(path_to_files):
    os.mkdir(path_to_files)

if not os.path.exists(savefiledir):
    print(f'Creating {savefiledir}')
    os.mkdir(savefiledir)

save_path = os.path.join(
    savefiledir,
    f'{save_name_header}_STRUCTURE_{save_file_type}'
)

print()
print(f'Saving to {save_path}')
print()

planetname_wanderer_median.save_data_to_save_files(
    savefiledir=savefiledir,
    save_name_header=save_name_header,
    save_file_type=save_file_type
)

print('Entire Pipeline took {time() - startFull} seconds')
