
# coding: utf-8

# # JWST - NIRCam - Level 3 TSO Pipeline
# This pipeline was commissioned to take input from the Level 2 pipeline -- having been processed through the NIRCam `ncdhas` pipeline -- and now being further processed through from a stack of images into a time series
# 
# **NEW METHOD**
# 
# 1. Develop a single routine that inputs 
#     1. String (fits file name) or array (loaded fits file)
#     2. The expected location of the star (center of frame is default)
#     3. Subframe size (for better center fitting)
#     4. List of aperture radii (or a float for a single aperture radii)
# 2. This routine will load a single fits file or list of fits files (one at a time; recursive?)
# 3. For each single, or recursively for a list of fits files, 
#     1. load the data.
#     2. Computer the time element
#     3. subtract the background (store background level)
#     4. isolate the star into a subframe
#     5. Cross-correlate a Gaussian (or JWST psf) with the image to find predicted center (store CC center)
#     6. Gaussian fit to subframe, starting at CC center (store GS center, width, amplitude)
#     7. Perform apeture photometry with each radius given at the beginning (store aperture radii as a function of radius)
# 
# This routine ensures that the user can manipulate the inputs as needed. Users can either send a single fits array, a set of fits array, a single string with the location of a fits file, or a list of strings with the location of several fits files.
# 
# The result will be a 'DataFrame' of the same depth as the input structure, containing (labeled as keys) the 
# - 'sky background'
# - 'cross correlation center'
# - 'gaussian center'
# - 'gaussian width'
# - 'gaussian ampitude'
# - 'aperture photometry dictionary' or 'aperture photometry dataframe'
#     - the keys to the aperture photometry dictionary or data frame will be the float values of the aperture radii
# - 'time' (in days?)

# **OLD METHOD - for Posterity and External Comparison**
# 1.  Input data from file directory from user
# 2.  Access that file directory and grab all file names
#      -- possible include a data file 
# 3.  Sequentially open all fits file in that directory (or from the data file)
# 4.  During the opening process, store the data frame(s) necessary for production of time series
# 5.  Remove the original data from RAM (too much space)
# 6.  Subtract median background
# 7.  Cross-Correlated Gaussian with center of image
# 8.  Fit a Gaussian to center of image, starting from Cross-Correlation solution
# 9.  Integrate (using 'exact') the aperture photometry
# 10. Store aperture photometry, gaussian centers, cross-correlation centers, gaussian widths, gaussian heights

# Load All Necessary Libraries and Functions
# ---
# 
#     `pylab`      : combination of array manipulation and plotting functions
#     `matplotlib` : specialized plotting functions
#     `numpy`      : array more manipulation functions
#     `pandas`     : dataframe -- more advanced array / table -- functions
#     `photutils`  : astropy associated package for aperture photometry
#     `astroML`    : better histogram function for plotting
#     `astropy`    : `modeling` : access linear and gaussian functions with astropy formatting
#                    `fitting`  : access to astropy fitting routines
#     `jd`         : julian date from header info calculations
#     `julian_date`: julian data from header info calculations written by Ian Crossfield (IJC)
#     `datatime`   : assists `jd` with calculating header time
#     `os`         : Operating System level control for python
#     `glob`       : grab list of files in directory
#     `sklearn`    : `externals`: imports operating system (storage) level function (i.e. joblib)
#     `statsmodels`: `robust`   : robust statistical modeling packages; `scale.mad` == median average distance
#     `sys`        : python-os level functions (i.e. path)
#     `time`       : compute and convert current timestamps from python / os

# In[1]:

# Matplotlib
get_ipython().magic(u'matplotlib inline')
from pylab              import gcf, sort, linspace, indices, std, empty, concatenate, pi, sqrt, ones, diag, inf
from pylab              import rcParams, array, get_current_fig_manager, twinx, figure, subplots_adjust

from matplotlib.ticker  import MaxNLocator
from matplotlib         import style
from matplotlib         import pyplot as plt

# Numpy & Pandas
from numpy              import min, max, median, mean, zeros, empty
from numpy              import ones, where, arange, indices
from pandas             import DataFrame, read_csv, scatter_matrix

# Astropy
from photutils          import CircularAperture, aperture_photometry
from astroML.plotting   import hist
from astropy.modeling   import models, fitting
from astropy.io         import fits

# Time Stamps
import jd

# Built in Libraries
from datetime           import datetime
from os                 import listdir
from glob               import glob

from julian_date        import gd2jd
# Adam Ginsburg
from image_registration import cross_correlation_shifts

# Data Storage from Sci-kits
from sklearn.externals import joblib

from seaborn            import *

# from socket             import gethostname
from statsmodels.robust import scale
from sys                import exit, stdout
from time               import time

style.use('fivethirtyeight')


# This is an example input for the requests below. The directory contains JWST-NIRCam fits files within it
#     - only works on Jonathan Fraine's Laptop
#     - soon to 'upgrade' to working on surtr
# 
# '/Users/jonathan/Research/NIRCam/CV3/StabilityTest/fitsfilesonly/reduced_orig_flags/redfits/NRCN821CLRSUB1-6012172256_1_481_SE_2016-01-12T18h00m43.red/'
# 
# There is also a test file in the current working directory named `'fits_input_file.txt'`. It was creating using the bash 'script'
# 
# ```bash
# cd /Users/jonathan/Research/NIRCam/CV3/StabilityTest/fitsfilesonly/reduced_orig_flags/redfits/NRCN821CLRSUB1-6012172256_1_481_SE_2016-01-12T18h00m43.red/
# 
# ls > fits_input_file.txt
# ```
# 
# ---
# Responding to the inquiry with (including appostraphes) either 
# 
# `'fits_input_file.txt'` 
# 
# or 
# 
# `'/Users/jonathan/Research/NIRCam/CV3/StabilityTest/fitsfilesonly/reduced_orig_flags/redfits/NRCN821CLRSUB1-6012172256_1_481_SE_2016-01-12T18h00m43.red/'` 
# 
# is successful

# Request Directory with a Set of Fits Files OR a Text File with the Same List
# ---

# In[2]:

list_of_data_file_types = ['.txt', '.dat', '.csv']
nircam_data = DataFrame()
found       = False
DataDir     = input()


for filetype in list_of_data_file_types:
    if filetype in DataDir:
        nircam_data['fitsfilenames'] = read_csv(DataDir)
        found = True

if not found:
    nircam_data['fitsfilenames'] = glob(DataDir+'/*')


# Compute Julian Data from Header
# ---
# 
# This function is a wrapper for `julian_date` in the `jd.py` package (soon to be converted to `julian_date.py` package.
# It's utility is in taking in the time stamps from the headers and converting them to the julian date; to be saved in the 'master' data frame below.

# In[3]:

def get_julian_date_from_header(header):
    from jd import julian_date
    fitsDate    = header['DATE-OBS']
    startTimeStr= header['TIME-OBS']
    endTimeStr  = header['TIME-END']
    
    yy,mm,dd    = fitsDate.split('-')
    
    hh1,mn1,ss1 = array(startTimeStr.split(':')).astype(float)
    hh2,mn2,ss2 = array(endTimeStr.split(':')).astype(float)
    
    startDate   = julian_date(yy,mm,dd,hh1,mn1,ss1)
    endDate     = julian_date(yy,mm,dd,hh2,mn2,ss2)

    return startDate, endDate


# Load Data / Gaussian Fit / AperturePhot Image
# ---
# 
# This function is the **crux** of the entire algorithm. The operation takes in one fits file name and outputs its time stamp, aperture photometry, gaussian centering / widths / amplitude, cross-correlation centering, and background subtracted values.  The routine does the following:
# 
# 1. Input:
#     1. String (fits file name) or array (loaded fits file)
#     2. The expected location of the star (center of frame is default)
#     3. Subframe size (for better center fitting)
#     4. List of aperture radii (or a float for a single aperture radii)
# 2. Operation:
#     1. load the data.
#     2. Computer the time element
#     3. subtract the background (store background level)
#     4. isolate the star into a subframe
#     5. Cross-correlate a Gaussian (or JWST psf) with the image to find predicted center (store CC center)
#     6. Gaussian fit to subframe, starting at CC center (store GS center, width, amplitude)
#     7. Perform apeture photometry with each radius given at the beginning (store aperture radii as a function of radius)
# 3. Output
#     1. time stamp
#     2. aperture photometry
#     3. gaussian amplitude
#     4. gaussian centering
#     5. gaussian widths
#     6. cross-correlation centering
#     7. background subtracted values.
# 
# This routine ensures that the user can manipulate the inputs as needed. Users can either send a single fits array, a set of fits array, a single string with the location of a fits file, or a list of strings with the location of several fits files.
# 

# In[4]:

def load_fit_phot_time(fitsfile, guesscenter = None, subframesize = [10,10], aperrad = [5], 
                           nGroupsBig = 100, stddev0 = 2.0):
    y,x     = 0,1
    zero    = 0
    day2sec = 86400.
    k       = int(fitsfile.split('_I')[-1][:3])
    
    fitsname      = fitsfile.split('/')[-1]
    fitsfile      = fits.open(fitsfile)
    startJD,endJD = get_julian_date_from_header(fitsfile[0].header)
    timeSpan      = (endJD - startJD)*day2sec/nGroupsBig
    time          = startJD  + timeSpan*(k+0.5) / day2sec - 2450000.

#     print '\nNEED to control for multiframe arrays; maybe request only SLP\n'
    dataframe     = fitsfile[0].data[2] - fitsfile[0].data[0]
    skybg         = np.median(dataframe)
    
    imagecenter   = 0.5*array(dataframe.shape)
    if guesscenter == None:
        guesscenter = imagecenter
    
    subframe      = dataframe[guesscenter[y]-subframesize[y]:guesscenter[y]+subframesize[y],
                              guesscenter[y]-subframesize[x]:guesscenter[y]+subframesize[x]].copy()
    
    # ysize, xsize  = fitsfile[0].data.shape
    yinds0, xinds0= indices(dataframe.shape)
    yinds         = yinds0[guesscenter[y]-subframesize[y]:guesscenter[y]+subframesize[y],
                           guesscenter[y]-subframesize[x]:guesscenter[y]+subframesize[x]]
    xinds         = xinds0[guesscenter[y]-subframesize[y]:guesscenter[y]+subframesize[y],
                           guesscenter[y]-subframesize[x]:guesscenter[y]+subframesize[x]]
    
    fitter        = fitting.LevMarLSQFitter()
    plane         = models.Linear1D
    gauss0        = models.Gaussian2D(amplitude = fitsfile[0].data.max(), 
                                      x_mean    = guesscenter[x], 
                                      y_mean    = guesscenter[y],
                                      x_stddev  = stddev0       ,
                                      y_stddev  = stddev0       ,
                                      theta     = zero)
    
    CCCenter      = cross_correlation_shifts(gauss0(xinds, yinds), subframe) + imagecenter
    CCCenter      = CCCenter[::-1] # need in order to associate y = 1, x = 0
    
    gauss1        = fitter(gauss0, xinds, yinds, subframe - skybg)
    
    circCenter     = gauss1.parameters[1:3][::-1] - imagecenter + subframesize
    
    circaper       = CircularAperture(circCenter, aperrad[0])
    aperphot       = aperture_photometry(data=subframe - skybg, apertures=circaper)
    del fitsfile[0].data
    fitsfile.close()
    del fitsfile
    
    return fitsname, float(aperphot['aperture_sum']), time, gauss1.amplitude.value, gauss1.y_mean.value,             gauss1.x_mean.value, abs(gauss1.y_stddev.value), abs(gauss1.x_stddev.value),             CCCenter[1], CCCenter[0], skybg

#     return time, aperphot['aperture_sum'], gauss1, CCCenter, skybg


# Test output using the first fits file name in the list from above
# ---

# In[5]:

load_fit_phot_time(nircam_data['fitsfilenames'][0], guesscenter = None)#[160,160]


# Wrapper function to cycle through each fits file name in the list of fits files from user input
# ---
# 
# Takes in a list of fits file names, loops over them in the crux function above, stores each entry (output from crux) into a dataframe for later storage and processing.
# 
# Input:
#     1. List of fits file names to be loaded
#     2. Initial guess location of star
#     3. Subframe size to compute centering and photometry within
#     4. Aperature radius to compute photometry over
#     5. Predicted with of PSF (nyquist sampling = 2)
# Operation:
#     1. Loop over each file in the list of fits files
#     2. Send the fits file names to the crux function
#     3. Receive output list of aper phot, gauss centers/widths/amplitudes, cross-corr centers, sky background
#     4. Input the above computed values in the master data frame for stroage and later processing
# Outputs:
#     1. Master dataframe containing list of aper phot, gauss centers/widths/amplitudes, cross-corr centers, sky bg

# In[6]:

def loads_fits_phots_times(fitsfiles, guesscenter = None, subframesize = [10,10], aperrad = [5], stddev0 = 2.0):
    '''
    'sky background'
    'cross correlation center'
    'gaussian center'
    'gaussian width'
    'gaussian ampitude'
    'aperture photometry dictionary' or 'aperture photometry dataframe'
    the keys to the aperture photometry dictionary or data frame will be the float values of the aperture radii
    'time' (in days?)
    '''
    
    print 'Need to add multiple aperture raddii usage'
    columnNames = ['filename'           , 'aperture phot %.1f' %aperrad[0], 
                   'time'               , 'gaussian amplitude' , 
                   'gaussian y center'  , 'gaussian x center'  , 
                   'gaussian y width'   , 'gaussian x width'   , 
                   'cross corr y center', 'cross corr x center', 
                   'sky background']

    nircam_master_df = DataFrame(columns=columnNames)
    for fitsfile in fitsfiles:
        columnInputs = load_fit_phot_time(fitsfile, guesscenter  = guesscenter, 
                                                    subframesize = subframesize, 
                                                    aperrad      = aperrad, 
                                                    stddev0      = stddev0)
        #
        nircam_master_df.loc[len(nircam_master_df)] = columnInputs
    
    return nircam_master_df


# Create JWST-NIRCam Master DataFrame and Print Out Table Thereof
# ---

# In[7]:

nircam_master_df = loads_fits_phots_times(nircam_data['fitsfilenames'], guesscenter = None, 
                                          subframesize = [10,10], aperrad = [3], stddev0 = 2.0)
nircam_master_df


# Generate Scatter Matrix to Cross Compare All Values with Eachother
# ---

# In[8]:

scatter_matrix(nircam_master_df.drop('filename',1), diagonal='kde', figsize=(14,12));


# Plot All Values as Function of Time and Gaussian Centers
# ---

# In[9]:

def renorm(arr):
    if arr.dtype == 'float64':
        return arr - median(arr)
    else:
        return arr

nircam_master_df.apply(renorm, axis=0)

fig = figure(figsize=(14,12))
for k, key in enumerate(nircam_master_df.keys()):
    ax  = fig.add_subplot(len(nircam_master_df.keys()), 1, k+1)
    if not key in ['time', 'filename']:
        ax.plot(nircam_master_df['time'], nircam_master_df[key])
        if k == len(nircam_master_df.keys()) - 1:
            ax.set_xlabel('time')
        else:
            ax.set_xticklabels([])
        
        ax.set_ylabel(key.replace('gaussian', 'gauss').replace('background', 'bg').replace(' ', '\n'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

ax  = fig.add_subplot(len(nircam_master_df.keys()), 1, 1)
ax.plot(nircam_master_df['gaussian y center'], nircam_master_df['aperture phot 3.0'], 'o')
ax.set_ylabel('aperture phot 3.0'.replace(' ', '\n'))
ax.set_xlabel('gauss y center')
ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

ax  = fig.add_subplot(len(nircam_master_df.keys()), 1, 3)
ax.plot(nircam_master_df['gaussian x center'], nircam_master_df['aperture phot 3.0'], 'o')
ax.set_ylabel('aperture phot 3.0'.replace(' ', '\n'))
ax.set_xlabel('gauss x center')
ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

subplots_adjust( hspace=1 )
fig.canvas.draw()


# ---
# The Following is Strictly from My Python Routine 
# ===
# ---

# ```python
# import numpy as np
# from bokeh.plotting import figure, show, output_file
# 
# N = 10000
# 
# x = np.random.normal(0,np.pi, N)
# y = np.sin(x) + np.random.normal(0,0.2,N)
# 
# output_file('test_bokeh2.html', title='scatter 10k points')
# 
# p = figure(webgl=False)
# p.scatter(x,y,alpha=0.1)
# show(p)
# ```
