# ExoplanetTSO
# Exoplanet Photometric Analysis for Time Series Observations
---

Many ground and space telescopes missions contain the capacity to perform photometric time series observations.  
In order to process that data, it is important to identify the host star, accurately estimate it's PSF (point spread function) center -- for all images in the time series.

**METHOD**

1. Develop a single routine that inputs 
    1. Directory name with fits files to generate the image array from both Spitzer and JWST-NIRCam
    2. Type of files using the file extension
    3. The expected location of the star (center of frame is default)
    4. Subframe size (for better estimation during center fitting)
    5. List of aperture radii (or a float for a single aperture radii)
    6. Method of measuring the backgrounds and/or clipping algorithms (mean or median)
2. This routine will load all files that match the specific extension given in (1.2) above
3. For each fits file:
    1. Load the data.
    2. Compute the time element (varies between Spiter and JWST) 
    3. Subtract the background (store background level)
        1. use mean/median full frame, mean/median outside a circle, mean/median from annulus, mean/median from median mask
    4. Isolate the star into a subframe
    5. Fit the center to this subframe, starting at guessed center:
        1. use Gaussian Moments, Gaussian Fits, Flux Weighted, Least Asymmetry centering methods
    6. Perform aperture photometry with each radius, looped over with both static and variable aperture radii
    7. Store aperture photometry vs aperture radii

This routine ensures that the user can manipulate the inputs as needed. Users can either send a single fits array, a set of fits array, a single string with the location of the fits files.

The result will be an instance of the `wanderer` class, containing (labeled as instance.values) the 
- 'sky background'
- 'gaussian center'
- 'gaussian width'
- 'gaussian ampitude'
- 'aperture photometry dataframe'
    - the keys to the aperture photometry Dataframe will be a string corresponding to CENTERING_BACKGROUND_STATIC_VARIABLE
- 'time' (in in MJD)

===
Jupyter Notebook Re-written in Markdown
===
Load All Necessary Libraries and Functions
---

```python
# Matplotlib for Plotting
%matplotlib inline
from astroML.plotting          import hist
from astropy.io                import fits
from astropy.modeling          import models, fitting
from datetime                  import datetime
from image_registration        import cross_correlation_shifts
from glob                      import glob
from matplotlib.ticker         import MaxNLocator
from matplotlib                import style
from os                        import listdir
from pandas                    import DataFrame, read_csv, read_pickle, scatter_matrix
from photutils                 import CircularAperture, CircularAnnulus, aperture_photometry, findstars
from least_asymmetry           import actr, moments, fitgaussian
from pylab                     import ion, gcf, sort, linspace, indices, median, mean, std, empty, figure, transpose, ceil
from pylab                     import concatenate, pi, sqrt, ones, diag, inf, rcParams, isnan, isfinite, array, nanmax
from numpy                     import min as npmin, max as npmax, zeros, arange, sum, float, isnan, hstack
from numpy                     import int32 as npint, round as npround, nansum as sum, nanstd as std
from seaborn                   import *
from scipy.special             import erf
from scipy                     import stats
from sklearn.externals         import joblib
from socket                    import gethostname
from statsmodels.robust        import scale
from statsmodels.nonparametric import kde
from sys                       import exit
from time                      import time, localtime

from numpy                     import zeros, nanmedian as median, nanmean as mean, nan
from sys                       import exit
from sklearn.externals         import joblib
from least_asymmetry           import actr

import numpy as np

style.use('fivethirtyeight')
```

This is an example input for the requests below. The directory contains a collection of fits files within it

'/path/to/fits/files/'

---

This function is a wrapper for `julian_date` in the `jd.py` package (soon to be converted to `julian_date.py` package.
It's utility is in taking in the time stamps from the headers and converting them to the julian date; to be saved in the 'master' data frame below.

**Load Saved Data from the Wanderer Class**
```python
method    = 'median'
midFrame  = frameSize//2 # We need to know frameSize ahead of time
yguess    = midFrame     # We assume that the stellar PSF is in the center of the frame
xguess    = midFrame     # We assume that the stellar PSF is in the center of the frame

dataDir     = '/path/to/fits/files/main/directory/'
fitsFileDir = 'path/to/fits/subdirectories/'

loadfitsdir = dataDir + fitsFileDir

example_wanderer_median = wanderer(fitsFileDir=loadfitsdir, filetype=filetype, 
                                            yguess=yguess, xguess=xguess, method=method)

print('Loading a saved instance of `wanderer` as `example_wanderer_median`\n')
example_wanderer_median.load_data_from_save_files(savefiledir='./SaveFiles/', saveFileNameHeader='Example_Wanderer_Median_', saveFileType='.pickle.save')
```

**Start from Scratch with the Wanderer Class**
```python
method    = 'median'
midFrame  = frameSize//2 # We need to know frameSize ahead of time
yguess    = midFrame     # We assume that the stellar PSF is in the center of the frame
xguess    = midFrame     # We assume that the stellar PSF is in the center of the frame

dataDir     = '/path/to/fits/files/main/directory/'
fitsFileDir = 'path/to/fits/subdirectories/'

loadfitsdir = dataDir + fitsFileDir

print('Initializing an instance of `wanderer` as `example_wanderer_median`\n')
example_wanderer_median = wanderer(fitsFileDir=loadfitsdir, filetype=filetype, 
                                            yguess=yguess, xguess=xguess, method=method)

print('Load Data From Fits Files in ' + loadfitsdir + '\n')
example_wanderer_median.load_data_from_fits_files()

print('Skipping Load Data From Save Files in ' + loadfitsdir + '\n')
# example_wanderer_median.load_data_from_save_files()

print('Find, flag, and NaN the "Bad Pixels" Outliers' + '\n')
example_wanderer_median.find_bad_pixels()

print('Fit for All Centers: Flux Weighted, Gaussian Fitting, Gaussian Moments, Least Asymmetry' + '\n')
# example_wanderer_median.fit_gaussian_fitting_centering()
# example_wanderer_median.fit_flux_weighted_centering()
# example_wanderer_median.fit_least_asymmetry_centering()
example_wanderer_median.fit_all_centering()

print('Measure Background Estimates with All Methods: Circle Masked, Annular Masked, KDE Mode, Median Masked' + '\n')
# example_wanderer_median.measure_background_circle_masked()
# example_wanderer_median.measure_background_annular_mask()
# example_wanderer_median.measure_background_KDE_Mode()
# example_wanderer_median.measure_background_median_masked()
example_wanderer_median.measure_all_background()

print('Iterating over Background Techniques, Centering Techniques, Aperture Radii' + '\n')
background_choices = example_wanderer_median.background_df.columns
centering_choices  = ['Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', 'LeastAsymmetry']
aperRads           = np.arange(1, 100.5,0.5)

start = time()
for bgNow in background_choices:
    for ctrNow in centering_choices:
        for aperRad in aperRads:
            print('Working on Background ' + bgNow + ' with Centering ' + ctrNow + ' and AperRad ' + str(aperRad), end=" ")
            example_wanderer_median.compute_flux_over_time(aperRad=aperRad, centering=ctrNow, background=bgNow)
            flux_key_now  = ctrNow + '_' + bgNow+'_' + 'rad' + '_' + str(aperRad)
            print(std(example_wanderer_median.flux_TSO_df[flux_key_now] / median(example_wanderer_median.flux_TSO_df[flux_key_now]))*ppm)

print('Operation took: ', time()-start)

print('Saving `example_wanderer_median` to a set of pickles for various Image Cubes and the Storage Dictionary')
example_wanderer_median.save_data_to_save_files(savefiledir='./SaveFiles/', saveFileNameHeader='Example_Wanderer_Median_', saveFileType='.pickle.save')

```
