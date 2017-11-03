from astroML.plotting          import hist
from astropy.io                import fits
from astropy.modeling          import models, fitting
from datetime                  import datetime
from functools                 import partial
# from image_registration        import cross_correlation_shifts
from glob                      import glob
from matplotlib.ticker         import MaxNLocator
from matplotlib                import style, colors
from multiprocessing           import cpu_count, Pool
from os                        import listdir, path, mkdir, chdir
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
from time                      import time, localtime, sleep
from tqdm                      import tnrange, tqdm_notebook

from numpy                     import zeros, nanmedian as median, nanmean as mean, nan
from sys                       import exit
from sklearn.externals         import joblib
from least_asymmetry           import actr

import numpy as np

rcParams['image.interpolation'] = 'None'
rcParams['image.cmap']          = 'Blues_r'
rcParams['axes.grid']           = False

# dataDir = '/path/to/fits/files/main/directory/'
# fitsFileDir = 'path/to/fits/subdirectories/'
#
# fitsFilenames = glob(dataDir + fitsFileDir + '*slp.fits')
# fitsFilenames


# Load All of the Data
# ===

# In[ ]:

def get_julian_date_from_gregorian_date(*date):
    """gd2jd.py converts a UT Gregorian date to Julian date.
    
    Functions for JD <-> GD conversion, 
      courtesy of Ian Crossfield at 
      http://www.astro.ucla.edu/~ianc/python/_modules/date.html
    
    Downloaded from Marshall Perrin Github at
        https://github.com/mperrin/misc_astro/blob/master/idlastro_ports/gd2jd.py
    
    Usage: gd2jd.py (2009, 02, 25, 01, 59, 59)

    To get the current Julian date:
        import time
        gd2jd(time.gmtime())

    Hours, minutes and/or seconds can be omitted -- if so, they are
    assumed to be zero.

    Year and month are converted to type INT, but all others can be
    type FLOAT (standard practice would suggest only the final element
    of the date should be float)
    """
    verbose=False
    if verbose: print(date)

    date = list(date)
    
    if len(date)<3:
        print("You must enter a date of the form (2009, 02, 25)!")
        return -1
    elif len(date)==3:
        for ii in range(3): date.append(0)
    elif len(date)==4:
        for ii in range(2): date.append(0)
    elif len(date)==5:
        date.append(0)

    yyyy = int(date[0])
    mm = int(date[1])
    dd = float(date[2])
    hh = float(date[3])
    min = float(date[4])
    sec = float(date[5])

    UT=hh+min/60+sec/3600


    total_seconds=hh*3600+min*60+sec
    fracday=total_seconds/86400

    if (100*yyyy+mm-190002.5)>0:
        sig=1
    else:
        sig=-1

    JD = 367*yyyy - int(7*(yyyy+int((mm+9)/12))/4) + int(275*mm/9) + dd + 1721013.5 + UT/24 - 0.5*sig +0.5

    months=["January", "February", "March", "April", "May", "June", "July", "August", 
                "September", "October", "November", "December"]

    # Now calculate the fractional year. Do we have a leap year?
    daylist=[31,28,31,30,31,30,31,31,30,31,30,31]
    daylist2=[31,29,31,30,31,30,31,31,30,31,30,31]
    if (yyyy%4 != 0):
        days=daylist2
    elif (yyyy%400 == 0):
        days=daylist2
    elif (yyyy%100 == 0):
        days=daylist
    else:
        days=daylist2

    daysum=0
    for y in range(mm-1):
        daysum=daysum+days[y]
    daysum=daysum+dd-1+UT/24

    if days[1]==29:
        fracyear=yyyy+daysum/366
    else:
        fracyear=yyyy+daysum/365
    if verbose: 
        print(yyyy,mm,dd,hh,min,sec)
        print("UT="+UT)
        print("Fractional day: %f" % fracday)
        print("\n"+months[mm-1]+" %i, %i, %i:%i:%i UT = JD %f" % (dd, yyyy, hh, min, sec, JD), end= " ")
        print(" = " + fracyear+"\n")
    
    return JD


# In[ ]:

def get_julian_date_from_header(header):
    # These are specific to STScI standards -- may vary on the ground
    fitsDate    = header['DATE-OBS']
    startTimeStr= header['TIME-OBS']
    endTimeStr  = header['TIME-END']
    
    yyyy, mm , dd   = fitsDate.split('-')
    
    hh1 , mn1, ss1  = array(startTimeStr.split(':')).astype(float)
    hh2 , mn2, ss2  = array(endTimeStr.split(':')).astype(float)
    
    yyyy  = float(yyyy)
    mm    = float(mm)
    dd    = float(dd)
    
    hh1   = float(hh1)
    mn1   = float(mn1)
    ss1   = float(ss1)
    
    hh2   = float(hh2)
    mn2   = float(mn2)
    ss2   = float(ss2)
    
    startDate   = get_julian_date_from_gregorian_date(yyyy,mm,dd,hh1,mn1,ss1)
    endDate     = get_julian_date_from_gregorian_date(yyyy,mm,dd,hh2,mn2,ss2)

    return startDate, endDate


# In[ ]:

def flux_weighted_centroid(image, ypos, xpos, bSize = 7):
    '''
        Flux-weighted centroiding (Knutson et al. 2008)
        xpos and ypos are the rounded pixel positions of the star
    '''
    
    ypos, xpos, bsize = np.int32([ypos, xpos, bSize])
    ## extract a box around the star:
    #im = a[ypos-bSize:ypos+bSize, xpos-bSize:xpos+bSize].copy()
    subImage = image[ypos-bSize:ypos+bSize, xpos-bSize:xpos+bSize].transpose().copy()

    y,x = 0,1
    
    ydim = subImage.shape[y]
    xdim = subImage.shape[x]
    
    ## add up the flux along x and y
    xflux = zeros(xdim)
    xrng  = arange(xdim)
    
    yflux = zeros(ydim)
    yrng  = arange(ydim)
    
    for i in range(xdim):
        xflux[i] = sum(subImage[i,:])

    for j in range(ydim):
        yflux[j] = sum(subImage[:,j])

    ## get the flux weighted average position:
    ypeak = sum(yflux * yrng) / sum(yflux) + ypos - float(bSize)
    xpeak = sum(xflux * xrng) / sum(xflux) + xpos - float(bSize)

    return (ypeak, xpeak)


# In[ ]:

def fit_gauss(subFrameNow, xinds, yinds, initParams, print_compare=False):
    # initParams = (height, x, y, width_x, width_y, offset)
    fit_lvmq = fitting.LevMarLSQFitter()
    model0  = models.Gaussian2D(amplitude=initParams[0], x_mean=initParams[1], y_mean=initParams[2], 
                                x_stddev=initParams[3], y_stddev=initParams[4], theta=0.0) + \
              models.Const2D(amplitude=initParams[5])
    
    model1  = fit_lvmq(model0, xinds, yinds, subFrameNow)
    model1  = fit_lvmq(model1, xinds, yinds, subFrameNow)
    
    if print_compare:
        print(model1.amplitude_0 - initParams[0], end=" ")
        print(model1.x_mean_0    - initParams[1], end=" ")
        print(model1.y_mean_0    - initParams[2], end=" ")
        print(model1.x_stddev_0  - initParams[3], end=" ")
        print(model1.y_stddev_0  - initParams[4], end=" ")
        print(model1.amplitude_1 - initParams[5])
    
    return model1.parameters


# In[ ]:

class wanderer(object):
    def __init__(self, fitsFileDir = './', filetype = 'slp.fits', telescope = None,
                 yguess=None, xguess=None, npix=5, method='mean', nCores = None):
        
        print('\n\n** Not all who wander are lost **\n\n')
        
        self.method   = method
        
        self.y,self.x = 0,1
        
        self.day2sec  = 86400.
        if method == 'mean':
            self.metric  = mean
        elif method == 'median':
            self.metric  = median
        else:
            raise Exception("`method` must be from the list ['mean', 'median']")
        
        self.fitsFileDir  = fitsFileDir
        self.fitsFilenames = glob(self.fitsFileDir + '/*' + filetype)
        self.nSlopeFiles  = len(self.fitsFilenames)
        
        if telescope is None:
            raise Exception("Please specify `telescope` as either 'JWST' or 'Spitzer' or 'HST'")
        
        self.telescope    = telescope
        
        if self.telescope == 'Spitzer':
            self.permBadPixels = fits.open(self.fitsFileDir + '../cal/nov14_ch1_bcd_pmask_subarray.fits')
        
        if self.nSlopeFiles == 0:
            print('Pipeline found no Files in ' + self.fitsFileDir + ' of type /*' + filetype)
            exit(-1)
        
        self.centering_df   = DataFrame()
        self.background_df  = DataFrame()
        self.flux_TSO_df    = DataFrame()
        self.noise_TSO_df   = DataFrame()
        
        if yguess == None:
            self.yguess = self.imageCube.shape[self.y]//2
        else:
            self.yguess = yguess
        if xguess == None:
            self.xguess = self.imageCube.shape[self.x]//2
        else:
            self.xguess = xguess
        
        self.npix = npix
        
        self.nCores cpu_count()//2 if nCores is None else int(nCores)
        
        tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst = localtime()
        print('Completed Class Definition at ' +
              str(tm_year) + '-' + str(tm_mon) + '-' + str(tm_mday) + ' ' + \
              str(tm_hour) + 'h' + str(tm_min) + 'm' + str(tm_sec) + 's')
        
    def jwst_load_fits_file(self):
        testfits            = fits.open(self.fitsFilenames[0])[0]
        
        self.nFrames        = self.nSlopeFiles
        self.imageCube      = np.zeros((self.nFrames, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.noiseCube      = np.zeros((self.nFrames, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.timeCube       = np.zeros(self.nFrames)
        
        del testfits
        
        for kframe, fname in tqdm_notebook(enumerate(self.fitsFilenames), desc='JWST Load File', leave = False, total=self.nSlopeFiles):
            fitsNow = fits.open(fname)
            
            self.imageCube[kframe] = fitsNow[0].data[0]
            self.noiseCube[kframe] = fitsNow[0].data[1]

            # re-write these 4 lines into `get_julian_date_from_header`
            startJD,endJD          = get_julian_date_from_header(fitsNow[0].header)
            timeSpan               = (endJD - startJD)*day2sec / self.nFrames
            self.timeCube[kframe]  = startJD  + timeSpan*(kframe+0.5) / day2sec - 2450000.
            
            del fitsNow[0].data
            fitsNow.close()
            del fitsNow
    
    def spitzer_load_fits_file(self):
        
        nFramesPerFile      = 64
        testfits            = fits.open(self.fitsFilenames[0])[0]
        
        self.nFrames        = self.nSlopeFiles * nFramesPerFile
        self.imageCube      = np.zeros((self.nFrames, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.noiseCube      = np.zeros((self.nFrames, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.timeCube       = np.zeros((self.nFrames))
        
        del testfits
        
        for kfile, fname in tqdm_notebook(enumerate(self.fitsFilenames), desc='Spitzer Load File', leave = False, total=self.nSlopeFiles):
            fitsNow = fits.open(fname)
            
            for iframe in range(nFramesPerFile):
                self.timeCube[kfile * nFramesPerFile + iframe]  = fitsNow[0].header['BMJD_OBS'] \
                                + iframe * float(fitsNow[0].header['FRAMTIME'])
                
                self.imageCube[kfile * nFramesPerFile + iframe] = fitsNow[0].data[iframe]

                # Initial Guess of Photon Limited Noise
                self.noiseCube[kfile * nFramesPerFile + iframe] = sqrt(fitsNow[0].data[iframe])
            
            del fitsNow[0].data
            fitsNow.close()
            del fitsNow
    
    def hst_load_fits_file(fitsNow):
        raise Exception('NEED TO CODE THIS')
    
    def load_data_from_fits_files(self):
        
        if self.telescope == 'JWST':
            self.jwst_load_fits_file()
        
        if self.telescope == 'Spitzer':
            self.spitzer_load_fits_file()
        
        if self.telescope == 'HST':
            self.hst_load_fits_file()
    
    def load_data_from_save_files(self, savefiledir=None, saveFileNameHeader=None, saveFileType='.pickle.save'):
        
        if saveFileNameHeader is None:
            raise Exception('`saveFileNameHeader` should be the beginning of each save file name')
        
        if savefiledir is None:
            savefiledir = './'
        
        print('Loading from Master Files')
        self.centering_df   = read_pickle(savefiledir  + saveFileNameHeader + '_centering_dataframe'  + saveFileType)
        self.background_df  = read_pickle(savefiledir + saveFileNameHeader + '_background_dataframe' + saveFileType)
        self.flux_TSO_df    = read_pickle(savefiledir   + saveFileNameHeader + '_flux_TSO_dataframe'   + saveFileType)
        
        try:
            self.noise_TSO_df   = read_pickle(savefiledir   + saveFileNameHeader + '_noise_TSO_dataframe'   + saveFileType)
        except:
            self.noise_TSO_df   = None
        
        self.imageCube        = joblib.load(savefiledir  + saveFileNameHeader + '_image_cube_array' + saveFileType)
        self.noiseCube        = joblib.load(savefiledir  + saveFileNameHeader + '_noise_cube_array' + saveFileType)
        self.timeCube         = joblib.load(savefiledir  + saveFileNameHeader + '_time_cube_array'  + saveFileType)
        
        self.imageBadPixMasks = joblib.load(savefiledir  + saveFileNameHeader + '_image_bad_pix_cube_array' + saveFileType)
        
        self.save_dict        = joblib.load(savefiledir + saveFileNameHeader + '_save_dict' + saveFileType)
        
        print('Assigning Parts of `self.save_dict` to individual data structures')
        for key in self.save_dict.keys():
            exec("self." + key + " = self.save_dict['" + key + "']")
        
        # self.fitsFileDir              = self.save_dict['fitsFileDir']
        # self.fitsFilenames             = self.save_dict['fitsFilenames']
        
        # self.background_Annulus       = self.save_dict['background_Annulus']
        # self.background_CircleMask    = self.save_dict['background_CircleMask']
        # self.background_GaussMoment   = self.save_dict['background_GaussMoment']
        # self.background_GaussianFit   = self.save_dict['background_GaussianFit']
        # self.background_KDEUniv       = self.save_dict['background_KDEUniv']
        # self.background_MedianMask    = self.save_dict['background_MedianMask']
        # self.centering_FluxWeight     = self.save_dict['centering_FluxWeight']
        # self.centering_GaussianFit    = self.save_dict['centering_GaussianFit']
        # self.centering_GaussianMoment = self.save_dict['centering_GaussianMoment']
        # self.centering_LeastAsym      = self.save_dict['centering_LeastAsym']
        # self.fitsFileDir              = self.save_dict['fitsFileDir']
        # self.heights_GaussianFit      = self.save_dict['heights_GaussianFit']
        # self.heights_GaussianMoment   = self.save_dict['heights_GaussianMoment']
        # # self.imageCubeMAD             = self.save_dict['imageCubeMAD']
        # # self.imageCubeMedian          = self.save_dict['imageCubeMedian']
        # self.fitsFilenames             = self.save_dict['fitsFilenames']
        # self.widths_GaussianFit       = self.save_dict['widths_GaussianFit']
        # self.widths_GaussianMoment    = self.save_dict['widths_GaussianMoment']
    
    def save_data_to_save_files(self, savefiledir=None, saveFileNameHeader=None, saveFileType='.pickle.save'):
        
        if saveFileNameHeader is None:
            raise Exception('`saveFileNameHeader` should be the beginning of each save file name')
        
        if savefiledir is None:
            savefiledir = './'
        
        if not path.exists(savefiledir):
            mkdir(savefiledir)
        
        if not path.exists(savefiledir + 'TimeStamped'):
            mkdir(savefiledir + 'TimeStamped')
        
        date            = localtime()
        
        year            = date.tm_year
        month           = date.tm_mon
        day             = date.tm_mday
        
        hour            = date.tm_hour
        minute          = date.tm_min
        sec             = date.tm_sec
        
        date_string     = '_' + str(year) + '-' + str(month)  + '-' + str(day) + '_' + \
                                str(hour) + 'h' + str(minute) + 'm' + str(sec) + 's'
        
        saveFileTypeBak = date_string + saveFileType
        
        self.initiate_save_dict()
        
        #self.save_dict = self.__dict__
        
        print('Saving to Master File -- Overwriting Previous Master')
        self.centering_df.to_pickle(savefiledir  + saveFileNameHeader + '_centering_dataframe'  + saveFileType)
        self.background_df.to_pickle(savefiledir + saveFileNameHeader + '_background_dataframe' + saveFileType)
        self.flux_TSO_df.to_pickle(savefiledir   + saveFileNameHeader + '_flux_TSO_dataframe'   + saveFileType)
        
        joblib.dump(self.imageCube, savefiledir  + saveFileNameHeader + '_image_cube_array' + saveFileType)
        joblib.dump(self.noiseCube, savefiledir  + saveFileNameHeader + '_noise_cube_array' + saveFileType)
        joblib.dump(self.timeCube , savefiledir  + saveFileNameHeader + '_time_cube_array'  + saveFileType)
        
        joblib.dump(self.imageBadPixMasks, savefiledir  + saveFileNameHeader + '_image_bad_pix_cube_array' + saveFileType)
        
        joblib.dump(self.save_dict, savefiledir + saveFileNameHeader + '_save_dict' + saveFileType)
        
        print('Saving to New TimeStamped File -- These Tend to Pile Up!')
        self.centering_df.to_pickle(savefiledir + 'TimeStamped/' + saveFileNameHeader + '_centering_dataframe'  + saveFileTypeBak)
        self.background_df.to_pickle(savefiledir + 'TimeStamped/' + saveFileNameHeader + '_background_dataframe' + saveFileTypeBak)
        self.flux_TSO_df.to_pickle(savefiledir + 'TimeStamped/' + saveFileNameHeader + '_flux_TSO_dataframe'   + saveFileTypeBak)
        
        joblib.dump(self.imageCube, savefiledir + 'TimeStamped/' + saveFileNameHeader + '_image_cube_array' + saveFileTypeBak)
        joblib.dump(self.noiseCube, savefiledir + 'TimeStamped/' + saveFileNameHeader + '_noise_cube_array' + saveFileTypeBak)
        joblib.dump(self.timeCube , savefiledir + 'TimeStamped/' + saveFileNameHeader + '_time_cube_array'  + saveFileTypeBak)
        
        joblib.dump(self.imageBadPixMasks, savefiledir + 'TimeStamped/' + saveFileNameHeader + '_image_bad_pix_cube_array' + saveFileTypeBak)
        
        joblib.dump(self.save_dict, savefiledir + 'TimeStamped/' + saveFileNameHeader + '_save_dict' + saveFileTypeBak)
    
    def initiate_save_dict(self):
        
        self.save_dict  = {} # DataFrame() -- test if this works later
        # for key,val in self.__dict__.items():
        #     self.save_dict[key]  = val
        
        try:
            self.save_dict['fitsFileDir']                = self.fitsFileDir
        except:
            pass
        try:
            self.save_dict['fitsFilenames']              = self.fitsFilenames
        except:
            pass
        try:
            self.save_dict['background_Annulus']        = self.background_Annulus
        except:
            pass
        try:
            self.save_dict['background_CircleMask']     = self.background_CircleMask
        except:
            pass
        try:
            self.save_dict['background_GaussMoment']    = self.background_GaussMoment
        except:
            pass
        try:
            self.save_dict['background_GaussianFit']    = self.background_GaussianFit
        except:
            pass
        try:
            self.save_dict['background_KDEUniv']        = self.background_KDEUniv
        except:
            pass
        try:
            self.save_dict['background_MedianMask']     = self.background_MedianMask
        except:
            pass
        try:
            self.save_dict['centering_FluxWeight']      = self.centering_FluxWeight
        except:
            pass
        try:
            self.save_dict['centering_GaussianFit']     = self.centering_GaussianFit
        except:
            pass
        try:
            self.save_dict['centering_GaussianMoment']  = self.centering_GaussianMoment
        except:
            pass
        try:
            self.save_dict['centering_LeastAsym']       = self.centering_LeastAsym
        except:
            pass
        try:
            self.save_dict['effective_widths']          = self.effective_widths
        except:
            pass
        try:
            self.save_dict['fitsFileDir']               = self.fitsFileDir
        except:
            pass
        try:
            self.save_dict['fitsFilenames']             = self.fitsFilenames
        except:
            pass
        try:
            self.save_dict['heights_GaussianFit']       = self.heights_GaussianFit
        except:
            pass
        try:
            self.save_dict['heights_GaussianMoment']    = self.heights_GaussianMoment
        except:
            pass
        try:
            self.save_dict['method']                    = self.method
        except:
            pass
        try:
            self.save_dict['npix']                      = self.npix
        except:
            pass
        try:
            self.save_dict['Quadrature']                      = self.npix
        except:
            pass
        try:
            self.save_dict['yguess']                    = self.yguess
        except:
            pass
        try:
            self.save_dict['xguess']                    = self.xguess
        except:
            pass
        try:
            self.save_dict['widths_GaussianFit']        = self.widths_GaussianFit
        except:
            pass
        try:
            self.save_dict['widths_GaussianMoment']     = self.widths_GaussianMoment
        except:
            pass
    
    def copy_instance(self):
        
        temp = wanderer()
        temp.saveFileNameHeader = self.saveFileNameHeader
        temp.savefiledir = self.savefiledir
        
        temp.centering_df = self.centering_df
        temp.background_df = self.background_df
        temp.flux_TSO_df = self.flux_TSO_df
        temp.noise_TSO_df = self.noise_TSO_df
        
        temp.imageCube = self.imageCube
        temp.noiseCube = self.noiseCube
        temp.timeCube = self.timeCube
        
        temp.imageBadPixMasks = self.imageBadPixMasks = joblib.load(savefiledir  + saveFileNameHeader + '_image_bad_pix_cube_array' + saveFileType)
        
        print('Assigning Parts of `temp.save_dict` to from `self.save_dict`')
        temp.save_dict = self.save_dict
        for key in self.save_dict.keys():
            exec("temp." + key + " = temp.save_dict['" + key + "']")
        
        # temp.fitsFileDir              = temp.save_dict['fitsFileDir']
        # temp.fitsFilenames             = temp.save_dict['fitsFilenames']
        
        # temp.background_Annulus       = temp.save_dict['background_Annulus']
        # temp.background_CircleMask    = temp.save_dict['background_CircleMask']
        # temp.background_GaussMoment   = temp.save_dict['background_GaussMoment']
        # temp.background_GaussianFit   = temp.save_dict['background_GaussianFit']
        # temp.background_KDEUniv       = temp.save_dict['background_KDEUniv']
        # temp.background_MedianMask    = temp.save_dict['background_MedianMask']
        # temp.centering_FluxWeight     = temp.save_dict['centering_FluxWeight']
        # temp.centering_GaussianFit    = temp.save_dict['centering_GaussianFit']
        # temp.centering_GaussianMoment = temp.save_dict['centering_GaussianMoment']
        # temp.centering_LeastAsym      = temp.save_dict['centering_LeastAsym']
        # temp.fitsFileDir              = temp.save_dict['fitsFileDir']
        # temp.heights_GaussianFit      = temp.save_dict['heights_GaussianFit']
        # temp.heights_GaussianMoment   = temp.save_dict['heights_GaussianMoment']
        # # temp.imageCubeMAD             = temp.save_dict['imageCubeMAD']
        # # temp.imageCubeMedian          = temp.save_dict['imageCubeMedian']
        # temp.fitsFilenames             = temp.save_dict['fitsFilenames']
        # temp.widths_GaussianFit       = temp.save_dict['widths_GaussianFit']
        # temp.widths_GaussianMoment    = temp.save_dict['widths_GaussianMoment']
    
    def find_bad_pixels(self, nSig=5):
        # we chose 5 arbitrarily, but from experience
        self.imageCubeMedian  = median(self.imageCube,axis=0)
        self.imageCubeMAD     = scale.mad(self.imageCube,axis=0)
        
        self.imageBadPixMasks = abs(self.imageCube - self.imageCubeMedian) > nSig*self.imageCubeMAD
        
        print("There are " + str(sum(self.imageBadPixMasks)) + " 'Hot' Pixels")
        
        # self.imageCube[self.imageBadPixMasks] = nan
    
    def fit_gaussian_centering(self, method='la', initc='fw', subArray=False, print_compare=False):
        y,x = 0,1
        
        yinds0, xinds0 = indices(self.imageCube[0].shape)
        
        ylower = self.yguess - self.npix
        yupper = self.yguess + self.npix
        xlower = self.xguess - self.npix
        xupper = self.xguess + self.npix
        
        ylower, xlower, yupper, xupper = np.int32([ylower, xlower, yupper, xupper])
        
        yinds = yinds0[ylower:yupper, xlower:xupper]
        xinds = xinds0[ylower:yupper, xlower:xupper]
        
        self.centering_GaussianFit    = zeros((self.imageCube.shape[0], 2))
        self.centering_GaussianMoment = zeros((self.imageCube.shape[0], 2))
        self.widths_GaussianFit       = zeros((self.imageCube.shape[0], 2))
        self.widths_GaussianMoment    = zeros((self.imageCube.shape[0], 2))
        
        self.heights_GaussianFit      = zeros(self.imageCube.shape[0])
        self.heights_GaussianMoment   = zeros(self.imageCube.shape[0])
        # self.rotation_GaussianFit     = zeros(self.imageCube.shape[0])
        self.background_GaussMoment   = zeros(self.imageCube.shape[0])
        self.background_GaussianFit   = zeros(self.imageCube.shape[0])
        
        for kframe in tqdm_notebook(range(self.nFrames), desc='GaussFit', leave = False, total=self.nFrames):
            subFrameNow = self.imageCube[kframe][ylower:yupper, xlower:xupper]
            subFrameNow[isnan(subFrameNow)] = median(~isnan(subFrameNow))
            
            cmom    = np.array(moments(subFrameNow))  # H, Xc, Yc, Xs, Ys, O
            
            if method == 'ap':
                if initc == 'fw' and self.centering_FluxWeight.sum():
                    FWCNow    = self.centering_FluxWeight[kframe]
                    FWCNow[self.y] = FWCNow[self.y] - ylower
                    FWCNow[self.x] = FWCNow[self.x] - xlower
                    gaussI    = hstack([cmom[0], FWCNow, cmom[3:]])
                if initc == 'cm':
                    gaussI  = hstack([cmom[0], cmom[1], cmom[2], cmom[3:]])
                
                gaussP  = fit_gauss(subFrameNow, xinds, yinds, gaussI) # H, Xc, Yc, Xs, Ys, Th, O
            
            if method == 'la':
                gaussP  = fitgaussian(subFrameNow)#, xinds, yinds, np.copy(cmom)) # H, Xc, Yc, Xs, Ys, Th, O
            
            self.centering_GaussianFit[kframe][self.x]     = gaussP[1] + xlower
            self.centering_GaussianFit[kframe][self.y]     = gaussP[2] + ylower
            self.centering_GaussianMoment[kframe][self.x]  = cmom[1]   + xlower
            self.centering_GaussianMoment[kframe][self.y]  = cmom[2]   + ylower
            
            self.widths_GaussianFit[kframe][self.x]        = gaussP[3]
            self.widths_GaussianFit[kframe][self.y]        = gaussP[4]
            self.widths_GaussianMoment[kframe][self.x]     = cmom[3]
            self.widths_GaussianMoment[kframe][self.y]     = cmom[4]
            
            self.heights_GaussianFit[kframe]          = gaussP[0]
            self.heights_GaussianMoment[kframe]       = cmom[0]
            
            self.background_GaussianFit[kframe]       = gaussP[5]
            self.background_GaussMoment[kframe]       = cmom[5]
            
            if print_compare:
                print('Finished Frame ' + str(kframe) + ' with Yc = ' + \
                      str(self.centering_GaussianFit[kframe][self.y] - self.centering_GaussianMoment[kframe][self.y]) + '; Xc = ' + \
                      str(self.centering_GaussianFit[kframe][self.x] - self.centering_GaussianMoment[kframe][self.x]))
            
            del gaussP, cmom
        
        self.centering_df = DataFrame()
        self.centering_df['Gaussian_Fit_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Centers'] = self.centering_GaussianFit.T[self.x]
        self.centering_df['Gaussian_Mom_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Mom_X_Centers'] = self.centering_GaussianFit.T[self.x]
        
        self.centering_df['Gaussian_Fit_Y_Widths']  = self.widths_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Widths']  = self.widths_GaussianFit.T[self.x]
        self.centering_df['Gaussian_Mom_Y_Widths']  = self.widths_GaussianMoment.T[self.y]
        self.centering_df['Gaussian_Mom_X_Widths']  = self.widths_GaussianMoment.T[self.x]
        
        self.centering_df['Gaussian_Fit_Heights']   = self.heights_GaussianFit
        self.centering_df['Gaussian_Mom_Heights']   = self.heights_GaussianMoment
        
        self.centering_df['Gaussian_Fit_Offset']    = self.background_GaussianFit
        self.centering_df['Gaussian_Mom_Offset']    = self.background_GaussMoment
        
        # self.centering_df['Gaussian_Fit_Rotation']    = self.rotation_GaussianFit
    
    def mp_fit_gaussian_centering(self, method='la', initc='fw', subArray=False, print_compare=False):
        
        def fit_one_gauss(self, image, ylower, yupper, xlower, xupper):
            subFrameNow = image[ylower:yupper, xlower:xupper]
            subFrameNow[isnan(subFrameNow)] = median(~isnan(subFrameNow))
        
            cmom    = np.array(moments(subFrameNow))  # H, Xc, Yc, Xs, Ys, O
            gaussP  = fitgaussian(subFrameNow)#, xinds, yinds, np.copy(cmom)) # H, Xc, Yc, Xs, Ys, Th, O
        
            return gaussP, cmom
        
        y,x = 0,1
        
        yinds0, xinds0 = indices(self.imageCube[0].shape)
        
        ylower = self.yguess - self.npix
        yupper = self.yguess + self.npix
        xlower = self.xguess - self.npix
        xupper = self.xguess + self.npix
        
        ylower, xlower, yupper, xupper = np.int32([ylower, xlower, yupper, xupper])
        
        yinds = yinds0[ylower:yupper, xlower:xupper]
        xinds = xinds0[ylower:yupper, xlower:xupper]
        
        self.centering_GaussianFit    = zeros((self.imageCube.shape[0], 2))
        self.centering_GaussianMoment = zeros((self.imageCube.shape[0], 2))
        self.widths_GaussianFit       = zeros((self.imageCube.shape[0], 2))
        self.widths_GaussianMoment    = zeros((self.imageCube.shape[0], 2))
        
        self.heights_GaussianFit      = zeros(self.imageCube.shape[0])
        self.heights_GaussianMoment   = zeros(self.imageCube.shape[0])
        # self.rotation_GaussianFit     = zeros(self.imageCube.shape[0])
        self.background_GaussMoment   = zeros(self.imageCube.shape[0])
        self.background_GaussianFit   = zeros(self.imageCube.shape[0])
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(fit_one_gauss, ylower=ylower, yupper=yupper, xlower=xlower, xupper=xupper)  # In case you need more kwargs to satisfy your aperture_photometry routine
        
        gaussian_centers = pool.starmap(func, zip(self.imageCube)) # the order is very important
        
        pool.close()
        pool.join()
        
        print('Finished with Fitting Centers. Now assigning to instance values.')
        for kframe, (gaussP, cmom) in enumerate(gaussian_centers):
            self.centering_GaussianFit[kframe][self.x]     = gaussP[1] + xlower
            self.centering_GaussianFit[kframe][self.y]     = gaussP[2] + ylower
            self.centering_GaussianMoment[kframe][self.x]  = cmom[1]   + xlower
            self.centering_GaussianMoment[kframe][self.y]  = cmom[2]   + ylower
        
            self.widths_GaussianFit[kframe][self.x]        = gaussP[3]
            self.widths_GaussianFit[kframe][self.y]        = gaussP[4]
            self.widths_GaussianMoment[kframe][self.x]     = cmom[3]
            self.widths_GaussianMoment[kframe][self.y]     = cmom[4]
        
            self.heights_GaussianFit[kframe]               = gaussP[0]
            self.heights_GaussianMoment[kframe]            = cmom[0]
        
            self.background_GaussianFit[kframe]            = gaussP[5]
            self.background_GaussMoment[kframe]            = cmom[5]
        
        self.centering_df = DataFrame()
        self.centering_df['Gaussian_Fit_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Centers'] = self.centering_GaussianFit.T[self.x]
        self.centering_df['Gaussian_Mom_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Mom_X_Centers'] = self.centering_GaussianFit.T[self.x]
        
        self.centering_df['Gaussian_Fit_Y_Widths']  = self.widths_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Widths']  = self.widths_GaussianFit.T[self.x]
        self.centering_df['Gaussian_Mom_Y_Widths']  = self.widths_GaussianMoment.T[self.y]
        self.centering_df['Gaussian_Mom_X_Widths']  = self.widths_GaussianMoment.T[self.x]
        
        self.centering_df['Gaussian_Fit_Heights']   = self.heights_GaussianFit
        self.centering_df['Gaussian_Mom_Heights']   = self.heights_GaussianMoment
        
        self.centering_df['Gaussian_Fit_Offset']    = self.background_GaussianFit
        self.centering_df['Gaussian_Mom_Offset']    = self.background_GaussMoment
        
        # self.centering_df['Gaussian_Fit_Rotation']    = self.rotation_GaussianFit
    
    def fit_flux_weighted_centering(self):
        y,x = 0,1
        
        yinds0, xinds0 = indices(self.imageCube[0].shape)
        
        ylower = self.yguess - self.npix
        yupper = self.yguess + self.npix
        xlower = self.xguess - self.npix
        xupper = self.xguess + self.npix
        
        ylower, xlower, yupper, xupper = np.int32([ylower, xlower, yupper, xupper])
        
        yinds = yinds0[ylower:yupper, xlower:xupper]
        xinds = xinds0[ylower:yupper, xlower:xupper]
        
        nFWCParams                = 2 # Xc, Yc
        self.centering_FluxWeight = np.zeros((self.nFrames, nFWCParams))
        
        for kframe in tqdm_notebook(range(self.nFrames), desc='FWC', leave = False, total=self.nFrames):
            subFrameNow = self.imageCube[kframe][ylower:yupper, xlower:xupper]
            subFrameNow[isnan(subFrameNow)] = median(~isnan(subFrameNow))
            
            self.centering_FluxWeight[kframe] = flux_weighted_centroid(self.imageCube[kframe], 
                                                                       self.yguess, self.xguess, bSize = 7)
            self.centering_FluxWeight[kframe] = self.centering_FluxWeight[kframe][::-1]
        
        self.centering_df['FluxWeighted_Y_Centers'] = self.centering_FluxWeight.T[self.y]
        self.centering_df['FluxWeighted_X_Centers'] = self.centering_FluxWeight.T[self.x]
    
    def mp_fit_flux_weighted_centering(self):
        def fit_one_fwc(self, image, ylower, xlower, yupper, xupper):
            subFrameNow = image[ylower:yupper, xlower:xupper]
            subFrameNow[isnan(subFrameNow)] = median(~isnan(subFrameNow))
        
            self.centering_FluxWeight[kframe] = flux_weighted_centroid(self.imageCube[kframe], self.yguess, self.xguess, bSize = 7)
            self.centering_FluxWeight[kframe] = self.centering_FluxWeight[kframe][::-1]
        
        y,x = 0,1
        
        yinds0, xinds0 = indices(self.imageCube[0].shape)
        
        ylower = self.yguess - self.npix
        yupper = self.yguess + self.npix
        xlower = self.xguess - self.npix
        xupper = self.xguess + self.npix
        
        ylower, xlower, yupper, xupper = np.int32([ylower, xlower, yupper, xupper])
        
        yinds = yinds0[ylower:yupper, xlower:xupper]
        xinds = xinds0[ylower:yupper, xlower:xupper]
        
        nFWCParams                = 2 # Xc, Yc
        self.centering_FluxWeight = np.zeros((self.nFrames, nFWCParams))
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(fit_one_fwc, ylower=ylower, yupper=yupper, xlower=xlower, xupper=xupper)  # In case you need more kwargs to satisfy your aperture_photometry routine
        
        gaussian_centers = pool.starmap(func, zip(self.imageCube)) # the order is very important
        
        pool.close()
        pool.join()
        
        self.centering_df['FluxWeighted_Y_Centers'] = self.centering_FluxWeight.T[self.y]
        self.centering_df['FluxWeighted_X_Centers'] = self.centering_FluxWeight.T[self.x]
    
    def fit_least_asymmetry_centering(self):
        
        y,x = 0,1
        
        yinds0, xinds0 = indices(self.imageCube[0].shape)
        
        ylower = self.yguess - self.npix
        yupper = self.yguess + self.npix
        xlower = self.xguess - self.npix
        xupper = self.xguess + self.npix
        
        ylower, xlower, yupper, xupper = np.int32([ylower, xlower, yupper, xupper])
        
        yinds = yinds0[ylower:yupper+1, xlower:xupper+1]
        xinds = xinds0[ylower:yupper+1, xlower:xupper+1]
        
        nAsymParams = 2 # Xc, Yc
        self.centering_LeastAsym  = np.zeros((self.nFrames, nAsymParams))
        
        for kframe in tqdm_notebook(range(self.nFrames), desc='Asym', leave = False, total=self.nFrames):
            # The following is a sequence of error handling and reattempts to center fit
            #   using the least_asymmetry algorithm
            #
            # The least_asymmetry algorithm returns a RunError if the center is not found in the image
            # Our experience shows that these failures are strongly correlated with the existence of a 
            #   cosmic ray hit nearby the PSF.
            #
            # Option 1: We run `actr` with default settings -- developed for Spitzer exoplanet lightcurves
            # Option 2: We assume that there is a deformation in the PSF and square the image array (preserving signs)
            # Option 3: We assume that there is a cosmic ray hit nearby the PSF and shrink the asym_rad by half
            # Option 4: We assume that there is a deformation in the PSF AND (or caused by) a cosmic ray hit 
            #   nearby the PSF; so we square the image array (preserving signs) AND shrink the asym_rad by half
            # Option 5: We assume that there is a deformation in the PSF AND (or caused by) a cosmic ray hit 
            #   nearby the PSF; so we square the image array (preserving signs) AND shrink the asym_rad by half
            #   BUT this time we have to get crazy and shrink the asym_size to 2 (reduces accuracy dramatically)
            
            
            fitFailed = False # "Benefit of the Doubt"
            
            # Option 1: We run `actr` with default settings that were developed for Spitzer exoplanet lightcurves
            kframe, yguess, xguess = np.int32([kframe, self.yguess, self.xguess])
            
            try:
                center_asym = actr(self.imageCube[kframe], [yguess, xguess], \
                                   asym_rad=8, asym_size=5, maxcounts=2, method='gaus', \
                                   half_pix=False, resize=False, weights=False)[0]
            except:
                fitFailed = True
            
            # Option 2: We assume that there is a deformation in the PSF and square the image array
            #  (preserving signs)
            if fitFailed:
                try: 
                    center_asym = actr(np.sign(self.imageCube[kframe])*self.imageCube[kframe]**2, \
                                       [yguess, xguess], \
                                       asym_rad=8, asym_size=5, maxcounts=2, method='gaus', \
                                       half_pix=False, resize=False, weights=False)[0]
                    fitFailed = False
                except:
                    pass
            
            # Option 3: We assume that there is a cosmic ray hit nearby the PSF and shrink the asym_rad by half
            if fitFailed:
                try: 
                    center_asym = actr(self.imageCube[kframe], [yguess, xguess], \
                                       asym_rad=4, asym_size=5, maxcounts=2, method='gaus', \
                                       half_pix=False, resize=False, weights=False)[0]
                    fitFailed = False
                except:
                    pass
            
            # Option 4: We assume that there is a deformation in the PSF AND (or caused by) a cosmic ray hit 
            #   nearby the PSF; so we square the image array (preserving signs) AND shrink the asym_rad by half
            if fitFailed:
                try:
                    center_asym = actr(np.sign(self.imageCube[kframe])*self.imageCube[kframe]**2, \
                                       [yguess, xguess], \
                                       asym_rad=4, asym_size=5, maxcounts=2, method='gaus', \
                                       half_pix=False, resize=False, weights=False)[0]
                    fitFailed = False
                except:
                    pass
            
            # Option 5: We assume that there is a deformation in the PSF AND (or caused by) a cosmic ray hit 
            #   nearby the PSF; so we square the image array (preserving signs) AND shrink the asym_rad by half
            #   BUT this time we have to get crazy and shrink the asym_size to 3 (reduces accuracy dramatically)
            if fitFailed:
                try:
                    center_asym = actr(np.sign(self.imageCube[kframe])*self.imageCube[kframe]**2, \
                                       [yguess, xguess], \
                                       asym_rad=4, asym_size=3, maxcounts=2, method='gaus', \
                                       half_pix=False, resize=False, weights=False)[0]
                    fitFailed = False
                except:
                    pass
            
            if fitFailed:
                # I ran out of options -- literally
                print('Least Asymmetry FAILED: and returned `RuntimeError`')
            
            try:
                # This will work if the fit was successful
                self.centering_LeastAsym[kframe]  = center_asym[::-1]
            except:
                print('Least Asymmetry FAILED: and returned `NaN`')
                fitFailed = True
            
            if fitFailed:
                print('Least Asymmetry FAILED: Setting self.centering_LeastAsym[%s] to Initial Guess: [%s,%s]' \
                      % (kframe, self.yguess, self.xguess))
                self.centering_LeastAsym[kframe]  = np.array([yguess, xguess])
        
        self.centering_df['LeastAsymmetry_Y_Centers'] = self.centering_LeastAsym.T[self.y]
        self.centering_df['LeastAsymmetry_X_Centers'] = self.centering_LeastAsym.T[self.x]
    
    def mp_fit_least_asymmetry_centering(self):
        
        y,x = 0,1
        
        yinds0, xinds0 = indices(self.imageCube[0].shape)
        
        ylower = self.yguess - self.npix
        yupper = self.yguess + self.npix
        xlower = self.xguess - self.npix
        xupper = self.xguess + self.npix
        
        ylower, xlower, yupper, xupper = np.int32([ylower, xlower, yupper, xupper])
        
        yguess, xguess = np.int32([self.yguess, self.xguess])
        
        yinds = yinds0[ylower:yupper+1, xlower:xupper+1]
        xinds = xinds0[ylower:yupper+1, xlower:xupper+1]
        
        nAsymParams = 2 # Xc, Yc
        # self.centering_LeastAsym  = np.zeros((self.nFrames, nAsymParams))
        # for kframe in tqdm_notebook(range(self.nFrames), desc='Asym', leave = False, total=self.nFrames):
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(actr, asym_rad=8, asym_size=5, maxcounts=2, method='gaus', half_pix=False, resize=False, weights=False)
        
        self.centering_LeastAsym = pool.starmap(func, zip(self.imageCube, [[yguess, xguess]]*self.nframes)) # the order is very important
        
        pool.close()
        pool.join()
        
        self.centering_LeastAsym = np.array(self.centering_LeastAsym[0])
        
        self.centering_df['LeastAsymmetry_Y_Centers'] = self.centering_LeastAsym.T[self.y]
        self.centering_df['LeastAsymmetry_X_Centers'] = self.centering_LeastAsym.T[self.x]
    
    def fit_all_centering(self):
        print('Fit for Gaussian Fitting & Gaussian Moment Centers\n')
        self.fit_gaussian_centering()
        
        print('Fit for Flux Weighted Centers\n')
        self.fit_flux_weighted_centering()
        
        print('Fit for Least Asymmetry Centers\n')
        self.fit_least_asymmetry_centering()
    
    def measure_effective_width(self, subFrame=False):
        if subFrame:
            midFrame = self.imageCube.shape[1]//2
            lower    = midFrame - self.npix
            upper    = midFrame + self.npix
            
            image_view = self.imageCube[:,lower:upper,lower:upper]
            self.effective_widths = image_view.sum(axis=(1,2))**2. / (image_view**2).sum(axis=(1,2))
        else:
            self.effective_widths = self.imageCube.sum(axis=(1,2))**2. / ((self.imageCube)**2).sum(axis=(1,2))
    
        self.centering_df['Effective_Widths']   = self.effective_widths
        
        x_widths = self.centering_df['Gaussian_Fit_X_Widths']
        y_widths = self.centering_df['Gaussian_Fit_Y_Widths']
        
        self.quadrature_widths                  = sqrt(x_widths**2 + y_widths**2)
        self.centering_df['Quadrature_Widths']  = self.quadrature_widths
    
    def measure_background_circle_masked(self, aperRad=None, centering='FluxWeight'):
        """
            Assigning all zeros in the mask to NaNs because the `mean` and `median` 
                functions are set to `nanmean` functions, which will skip all NaNs
        """
        
        if aperRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                aperRad = 100
            else:
                aperRad = 10
        
        # medianCenter   = median(self.centering_FluxWeight, axis=0)
        
        self.background_CircleMask = np.zeros(self.nFrames)
        for kframe in tqdm_notebook(range(self.nFrames), desc='CircleBG', leave = False, total=self.nFrames):
            aperture       = CircularAperture(self.centering_FluxWeight[kframe], aperRad)
            
            aper_mask = aperture.to_mask(method='exact')[0]    # list of ApertureMask objects (one for each position)
            
            # backgroundMask = abs(aperture.get_fractions(np.ones(self.imageCube[0].shape))-1)
            backgroundMask = aper_mask.to_image(self.imageCube[0].shape).astype(bool)
            backgroundMask = ~backgroundMask#[backgroundMask == 0] = False
            
            self.background_CircleMask[kframe] = self.metric(self.imageCube[kframe][backgroundMask])
        
        self.background_df['CircleMask'] = self.background_CircleMask.copy()
    
    def mp_measure_background_circle_masked(self, centering='FluxWeight'):
        """
            Assigning all zeros in the mask to NaNs because the `mean` and `median` 
                functions are set to `nanmean` functions, which will skip all NaNs
        """
        def measure_bg1(self, image, center, aperRad, method='exact'):
            aperture  = CircularAperture(center, aperRad)
            aper_mask = aperture.to_mask(method=method)[0]    # list of ApertureMask objects (one for each position)
        
            backgroundMask = aper_mask.to_image(image).astype(bool)
            backgroundMask = ~backgroundMask
            
            return self.metric(image[backgroundMask])
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(measure_bg1, aperRad=aperRad, method=method)
        
        self.background_CircleMask = pool.starmap(func, zip(self.imageCube, self.centering_FluxWeight)) # the order is very important
        
        pool.close()
        pool.join()
        
        self.background_CircleMask       = np.array(self.background_CircleMask)
        self.background_df['CircleMask'] = self.background_CircleMask.copy()
    
    def measure_background_annular_mask(self, innerRad=None, outerRad=None):
        
        if innerRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                innerRad = 100
            else:
                innerRad = 8
        
        if outerRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                outerRad = 150
            else:
                outerRad = 13
        
        self.background_Annulus = np.zeros(self.nFrames)
        
        for kframe in tqdm_notebook(range(self.nFrames), desc='AnnularBG', leave = False, total=self.nFrames):
            innerAperture = CircularAperture(self.centering_FluxWeight[kframe], innerRad)
            outerAperture = CircularAperture(self.centering_FluxWeight[kframe], outerRad)
            
            inner_aper_mask = innerAperture.to_mask(method='exact')[0]
            inner_aper_mask = inner_aper_mask.to_image(self.imageCube[0].shape).astype(bool)
            
            outer_aper_mask = outerAperture.to_mask(method='exact')[0]
            outer_aper_mask = outer_aper_mask.to_image(self.imageCube[0].shape).astype(bool)
            
            backgroundMask = (~inner_aper_mask)*outer_aper_mask
            
            self.background_Annulus[kframe] = self.metric(self.imageCube[kframe][backgroundMask])
        
        self.background_df['AnnularMask'] = self.background_Annulus.copy()
    
    def mp_measure_background_annular_mask(self, innerRad=None, outerRad=None, method='exact'):
        
        def measure_bg1(self, image, center, innerRad, outerRad, method='exact'):
            innerAperture   = CircularAperture(center, innerRad)
            outerAperture   = CircularAperture(center, outerRad)
            
            inner_aper_mask = innerAperture.to_mask(method=method)[0]
            inner_aper_mask = inner_aper_mask.to_image(image.shape).astype(bool)
            
            outer_aper_mask = outerAperture.to_mask(method=method)[0]
            outer_aper_mask = outer_aper_mask.to_image(image.shape).astype(bool)
            
            backgroundMask = (~inner_aper_mask)*outer_aper_mask
            
            return self.metric(image[backgroundMask])
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(measure_bg1, aperRad=aperRad, method=method)
        
        self.background_Annulus = pool.starmap(func, zip(self.imageCube, self.centering_FluxWeight)) # the order is very important
        
        pool.close()
        pool.join()
        
        self.background_Annulus           = np.array(self.background_Annulus)
        self.background_df['AnnularMask'] = self.background_Annulus.copy()
    
    def measure_background_median_masked(self, aperRad=None, nSig=5):
        
        if aperRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                aperRad = 100
            else:
                aperRad = 10
        
        self.background_MedianMask  = np.zeros(self.nFrames)
        
        for kframe in tqdm_notebook(range(self.nFrames), desc='MedianMaskedBG', leave = False, total=self.nFrames):
            aperture       = CircularAperture(self.centering_FluxWeight[kframe], aperRad)
            aperture       = aperture.to_mask(method='exact')[0]
            aperture       = aperture.to_image(self.imageCube[0].shape).astype(bool)
            backgroundMask = ~aperture
            
            medFrame  = median(self.imageCube[kframe][backgroundMask])
            madFrame  = scale.mad(self.imageCube[kframe][backgroundMask])
            
            medianMask= abs(self.imageCube[kframe] - medFrame) < nSig*madFrame
            
            maskComb  = medianMask*backgroundMask
            # maskComb[maskComb == 0] = False
            
            self.background_MedianMask[kframe] = self.metric(self.imageCube[kframe][maskComb])
        
        self.background_df['MedianMask'] = self.background_MedianMask
    
    def mp_measure_background_median_masked(self, aperRad, nSig=5):
        
        def measure_bg1(self, image, center, aperRad, nSig = 5, method='exact'):
            aperture       = CircularAperture(center, aperRad)
            aperture       = aperture.to_mask(method=method)[0]
            aperture       = aperture.to_image(image.shape).astype(bool)
            backgroundMask = ~aperture
            
            medFrame  = median(image[backgroundMask])
            madFrame  = scale.mad(image[backgroundMask])
            
            medianMask= abs(image - medFrame) < nSig*madFrame
            
            maskComb  = medianMask*backgroundMask
            
            return self.metric(image[maskComb])
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(measure_bg1, method=method, aperRad=aperRad, nSig=nSig)
        
        self.background_Annulus = pool.starmap(func, zip(self.imageCube, self.centering_FluxWeight)) # the order is very important
        
        pool.close()
        pool.join()
        
        self.background_MedianMask       = np.array(self.background_Annulus)
        self.background_df['MedianMask'] = self.background_MedianMask
    
    def measure_background_KDE_Mode(self, aperRad=None):
        
        if aperRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                aperRad = 100
            else:
                aperRad = 10
        
        self.background_KDEUniv = np.zeros(self.nFrames)
        
        for kframe in tqdm_notebook(range(self.nFrames), desc='KDE_BG', leave = False, total=self.nFrames):
            aperture       = CircularAperture(self.centering_FluxWeight[kframe], aperRad)
            aperture       = aperture.to_mask(method='exact')[0]
            aperture       = aperture.to_image(self.imageCube[0].shape).astype(bool)
            backgroundMask = ~aperture
            
            kdeFrame = kde.KDEUnivariate(self.imageCube[kframe][backgroundMask])
            kdeFrame.fit()
            
            self.background_KDEUniv[kframe] = kdeFrame.support[kdeFrame.density.argmax()]
        
        self.background_df['KDEUnivMask'] = self.background_KDEUniv
    
    def mp_measure_background_KDE_Mode(self, aperRad=None):
        
        self.background_KDEUniv = np.zeros(self.nFrames)
        
        def measure_bg1(self, image, center, aperRad, method='exact'):
            aperture       = CircularAperture(center, aperRad)
            aperture       = aperture.to_mask(method=method)[0]
            aperture       = aperture.to_image(image.shape).astype(bool)
            backgroundMask = ~aperture
            
            kdeFrame = kde.KDEUnivariate(image[backgroundMask].ravel())
            kdeFrame.fit()
            
            return kdeFrame.support[kdeFrame.density.argmax()]
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(measure_bg1, method=method, aperRad=aperRad, nSig=nSig)
        
        self.background_KDEUniv = pool.starmap(func, zip(self.imageCube, self.centering_FluxWeight)) # the order is very important
        
        pool.close()
        pool.join()
        
        self.background_KDEUniv           = np.array(self.background_KDEUniv)
        self.background_df['KDEUnivMask'] = self.background_KDEUniv
    
    def measure_all_background(self, nSig=5):
        print('Measuring Background Using Circle Mask')
        self.measure_background_circle_masked()
        print('Measuring Background Using Annular Mask')
        self.measure_background_annular_mask()
        print('Measuring Background Using Median Mask')
        self.measure_background_median_masked(nSig=nSig)
        print('Measuring Background Using KDE Mode')
        self.measure_background_KDE_Mode()
    
    def compute_flux_over_time(self, aperRad=None, centering='GaussianFit', background='AnnularMask', useTheForce=False):
        y,x = 0,1
        
        if background not in self.background_df.columns:
            raise Exception("`background` must be in", self.background_df.columns)
        
        if centering not in ['Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', 'LeastAsymmetry']:
            raise Exception("`centering` must be either 'Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', or 'LeastAsymmetry'")
        
        if aperRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                staticRad = 70
            else:
                staticRad = 3
        
        # if varRad is not None and 'Effective_Widths' not in self.centering_df.columns:
        #     self.measure_effective_width()
        
        centering_Use = np.transpose([self.centering_df[centering + '_Y_Centers'], 
                                      self.centering_df[centering + '_X_Centers']])
        
        background_Use= self.background_df[background]
        
        flux_key_now  = centering + '_' + background+'_' + 'rad' + '_' + str(aperRad)
        # if varRad is not None:
        #     flux_key_now  = centering + '_' + background+'_' + 'rad' + '_' + str(aperRad)# + '_' + str(varRad)
        # else:
        #     flux_key_now  = centering + '_' + background+'_' + 'rad' + '_' + str(aperRad)# + '_None'
        
        if flux_key_now not in self.flux_TSO_df.keys() or useTheForce:
            flux_TSO_now  = np.zeros(self.nFrames)
            noise_TSO_now = np.zeros(self.nFrames)
            
            for kframe in tqdm_notebook(range(self.nFrames), desc='Flux', leave = False, total=self.nFrames):
                frameNow  = np.copy(self.imageCube[kframe]) - background_Use[kframe]
                frameNow[np.isnan(frameNow)] = median(frameNow)

                noiseNow  = np.copy(self.noiseCube[kframe])**2.
                noiseNow[np.isnan(noiseNow)] = median(noiseNow)
                
                # if varRad is not None:
                #     aperRad = staticRad + varRad * self.centering_df['Quadrature_Widths'][kframe]
                # else:
                #     aperRad = staticRad
                
                aperture  = CircularAperture([centering_Use[kframe][self.x], centering_Use[kframe][self.y]], r=aperRad)
                
                flux_TSO_now[kframe]  = aperture_photometry(frameNow, aperture)['aperture_sum']
                noise_TSO_now[kframe] = sqrt(aperture_photometry(noiseNow, aperture)['aperture_sum'])

            self.flux_TSO_df[flux_key_now]  = flux_TSO_now
            self.noise_TSO_df[flux_key_now] = noise_TSO_now
        else:
            print(flux_key_now + ' exists: if you want to overwrite, then you `useTheForce=True`')
    
    def compute_flux_over_time_over_aperRad(self, aperRads=[], centering_choices=[], background_choices=[], \
                                            useTheForce=False, verbose=False):
        
        ppm   = 1e6
        start = time()
        for bgNow in background_choices:
            for ctrNow in centering_choices:
                for aperRad in aperRads:
                    if verbose: 
                        print('Working on Background {} with Centering {} and AperRad {}'.format(bgNow, ctrNow, aperRad), end=" ")
                    
                    self.compute_flux_over_time(aperRad    = aperRad  , \
                                                centering  = ctrNow     , \
                                                background = bgNow      , \
                                                useTheForce= useTheForce, \
                                                verbose    = verbose    )
                    
                    flux_key_now  = "{}_{}_rad_{}".format(ctrNow, bgNow, aperRad)
                    
                    if verbose:
                        flux_now = self.flux_TSO_df[flux_key_now]
                        print( std( flux_now/ median(flux_now)) * ppm)
        
        print('Operation took: ', time()-start)
    
    def mp_compute_flux_over_time_per_aperRad(self, aperRad=3.0, centering='GaussianFit', \
                                                background='AnnularMask', useTheForce=False):
        '''
            aperRad=3.0 assumes the PSF is nyquist samples, such that aperRad == 1.5*PSF_FWHM
        '''
        def compute_flux_one_image(self, image, noise, yx_centers, background, aperRad):
        
            y,x = 0,1
        
            frameNow                     = np.copy(image) - background
            frameNow[np.isnan(frameNow)] = median(frameNow)
        
            noiseNow  = np.copy(noise)**2.
            noiseNow[np.isnan(noiseNow)] = median(noiseNow)
        
            aperture  = CircularAperture([yx_centers[x], yx_centers[y]], r=aperRad)
        
            fluxNow   = aperture_photometry(frameNow, aperture)['aperture_sum']
            noiseNow  = sqrt(aperture_photometry(noiseNow, aperture)['aperture_sum'])
        
            return fluxNow, noiseNow
        
        if background not in self.background_df.columns:
            raise Exception("`background` must be in", self.background_df.columns)
        
        if centering not in ['Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', 'LeastAsymmetry']:
            raise Exception("`centering` must be either 'Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', or 'LeastAsymmetry'")
        
        if aperRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                staticRad = 70
            else:
                staticRad = 3
        
        # if varRad is not None and 'Effective_Widths' not in self.centering_df.columns:
        #     self.measure_effective_width()
        
        centering_Use = np.transpose([self.centering_df[centering + '_Y_Centers'], 
                                      self.centering_df[centering + '_X_Centers']])
        
        background_Use= self.background_df[background]
        
        # Modeled after: compute_flux_one_image(self, image, noise, yx_centers, background, aperRad)
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(compute_flux_one_image, aperRad=aperRad)  # In case you need more kwargs to satisfy your aperture_photometry routine
        
        photometry = pool.starmap(func, zip(self.imageCube, self.noiseCube, centering_Use, background_Use)) # the order is very important
        
        pool.close()
        pool.join()
        
        flux_key_now                   = centering + '_' + background+'_' + 'rad' + '_' + str(aperRad)
        self.flux_TSO_df[flux_key_now] = np.array(photometry)

#
# # Necessary Constants for Both Module A and Module B
# # ---
#
# # In[ ]:
#
# ppm             = 1e6
# y,x             = 0,1
#
# yguess, xguess  = 160., 167. # Specific to JWST WLP Test Data
# filetype        = 'slp.fits' # Specific to JWST WLP Test Data
#
#
# # Load Stored Instance from Save Files
# # ---
#
# # In[ ]:
#
# dataDir     = '/path/to/fits/files/main/directory/'
# fitsFileDir = 'path/to/fits/subdirectories/'
#
# self.fitsFileDir = dataDir + fitsFileDir
#
#
# # In[ ]:
#
# method = 'mean'
# example_wanderer_mean = wanderer(fitsFileDir=self.fitsFileDir, filetype=filetype,
#                                             yguess=yguess, xguess=xguess, method=method)
#
# example_wanderer_mean.load_data_from_save_files(savefiledir='./SaveFiles/',
#                                                      saveFileNameHeader='Example_Wanderer_Mean_', saveFileType='.pickle.save')
#
#
# # In[ ]:
#
# method = 'median'
# self = wanderer(fitsFileDir=self.fitsFileDir_ModA, filetype=filetype,
#                                             yguess=yguess, xguess=xguess, method=method)
#
# self.load_data_from_save_files(savefiledir='./SaveFiles/', saveFileNameHeader='self_', saveFileType='.pickle.save')
#
#
# # Start a New Instance with Median for the Metric
# # ---
#
# # In[ ]:
#
# dataDir     = '/path/to/fits/files/main/directory/'
# fitsFileDir = 'path/to/fits/subdirectories/'
#
# self.fitsFileDir = dataDir + fitsFileDir
#
#
# # In[ ]:
#
# method = 'median'
#
# print('Initialize an instance of `wanderer` as `self`\n')
# self = wanderer(fitsFileDir=self.fitsFileDir_ModB, filetype=filetype,
#                                             yguess=yguess, xguess=xguess, method=method)
#
# print('Load Data From Fits Files in ' + fitsFileDir_ModB + '\n')
# self.load_data_from_fits_files()
#
# print('Skipping Load Data From Save Files in ' + fitsFileDir_ModB + '\n')
# # self.load_data_from_save_files()
#
# print('Find, flag, and NaN the "Bad Pixels" Outliers' + '\n')
# self.find_bad_pixels()
#
# print('Fit for All Centers: Flux Weighted, Gaussian Fitting, Gaussian Moments, Least Asymmetry' + '\n')
# # self.fit_gaussian_fitting_centering()
# # self.fit_flux_weighted_centering()
# # self.fit_least_asymmetry_centering()
# self.fit_all_centering()
#
# print('Measure Background Estimates with All Methods: Circle Masked, Annular Masked, KDE Mode, Median Masked' + '\n')
# # self.measure_background_circle_masked()
# # self.measure_background_annular_mask()
# # self.measure_background_KDE_Mode()
# # self.measure_background_median_masked()
# self.measure_all_background()
#
# print('Iterating over Background Techniques, Centering Techniques, Aperture Radii' + '\n')
# background_choices = self.background_df.columns
# centering_choices  = ['Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', 'LeastAsymmetry']
# aperRads           = np.arange(1, 100.5,0.5)
#
# start = time()
# self.SDNR_df = DataFrame()
# for kBG, bgNow in tqdm_notebook(enumerate(background_choices), desc='Background', \
#                                 leave = True, total=len(background_choices)):
#     for kCTR, ctrNow in tqdm_notebook(enumerate(centering_choices), desc='Centering', \
#                                       leave = True, total=len(centering_choices)):
#         for staticRad in tqdm_notebook(staticRads, desc='StaticRad', leave = True, total=len(staticRads)):
#             for varRad in tqdm_notebook(varRads, desc='VarRad', leave = True, total=len(varRads)):
#                 if varRad is not None:
#                     flux_key_now  = ctrNow + '_' + bgNow+'_' + 'rad' + '_' + str(staticRad) + '_' + str(varRad)
#                 else:
#                     flux_key_now  = ctrNow + '_' + bgNow+'_' + 'rad' + '_' + str(staticRad) + '_None'
#
#                 self.compute_flux_over_time(staticRad=staticRad, varRad=varRad, \
#                                                          centering=ctrNow, background=bgNow,useTheForce=useTheForce)
#
#                 fluxNow     = self.flux_TSO_df[flux_key_now]
#                 medFluxNow  = median(self.flux_TSO_df[flux_key_now])
#
#                 # average SDNR per `staticRad`
#                 fluxNow    /= len(varRads)
#                 medFluxNow /= len(varRads)
#
#                 # Standard Deviation of the Normalized Residuals
#                 SDNR        = std(fluxNow / medFluxNow)*ppm
#                 self.SDNR_df[flux_key_now]  = [SDNR]
#
#                 print('Finished with Background ' + str(kBG) + ' ' + bgNow + ' with Centering ' + str(kCTR) \
#                           + ' ' + ctrNow + ' and staticRad ' + str(staticRad) + ' and varRad ' + str(varRad), \
#                       int(np.round(SDNR)), 'ppm')
#
# print('Saving `self` to a set of pickles for various Image Cubes and the Storage Dictionary')
# self.save_data_to_save_files(savefiledir='./SaveFiles/', saveFileNameHeader='self_', saveFileType='.pickle.save')
#
#
# # Start a New Instance with Mean for the Metric
# # ---
#
# # In[ ]:
#
# method = 'mean'
#
# print('Initialize an instance of `wanderer` as `example_wanderer_mean`')
# example_wanderer_mean = wanderer(fitsFileDir=self.fitsFileDir_ModB, filetype = filetype,
#                                 yguess=yguess, xguess=xguess, method=method)
#
# print('Load Data From Fits Files in ' + self.fitsFileDir)
# example_wanderer_mean.load_data_from_fits_files()
#
# print('Skipping Load Data From Save Files in ' + self.fitsFileDir)
# # example_wanderer_mean.load_data_from_save_files()
#
# print('Find, flag, and NaN the "Bad Pixels" Outliers')
# example_wanderer_mean.find_bad_pixels()
#
# print('Fit for All Centers: Flux Weighted, Gaussian Fitting, Gaussian Moments, Least Asymmetry')
# # example_wanderer_mean.fit_gaussian_fitting_centering()
# # example_wanderer_mean.fit_flux_weighted_centering()
# # example_wanderer_mean.fit_least_asymmetry_centering()
# example_wanderer_mean.fit_all_centering()
#
# print('Measure Background Estimates with All Methods: Circle Masked, Annular Masked, KDE Mode, Median Masked')
# # example_wanderer_mean.measure_background_circle_masked()
# # example_wanderer_mean.measure_background_annular_mask()
# # example_wanderer_mean.measure_background_KDE_Mode()
# # example_wanderer_mean.measure_background_median_masked()
# example_wanderer_mean.measure_all_background()
#
# print('Iterating over Background Techniques, Centering Techniques, Aperture Radii')
# background_choices = example_wanderer_mean.background_df.columns
# centering_choices  = ['Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', 'LeastAsymmetry']
# aperRads           = np.arange(1, 100.5,0.5)
#
# useTheForce = False # True :: recompute flux for a given aperature radius, centering method, and background technique
#
# # print('Working on Background ' + str(kBG) + ' ' + bgNow + ' with Centering ' + str(kCTR) \
# #       + ' ' + ctrNow + ' and staticRad ' + str(staticRad) + ' with varRad ' + str(varRad), end=" ")
#
# start = time()
# example_wanderer_mean.SDNR_df = DataFrame()
# for kBG, bgNow in tqdm_notebook(enumerate(background_choices), desc='Background', \
#                                 leave = True, total=len(background_choices)):
#     for kCTR, ctrNow in tqdm_notebook(enumerate(centering_choices), desc='Centering', \
#                                       leave = True, total=len(centering_choices)):
#         for staticRad in tqdm_notebook(staticRads, desc='StaticRad', leave = True, total=len(staticRads)):
#             for varRad in tqdm_notebook(varRads, desc='VarRad', leave = True, total=len(varRads)):
#                 if varRad is not None:
#                     flux_key_now  = ctrNow + '_' + bgNow+'_' + 'rad' + '_' + str(staticRad) + '_' + str(varRad)
#                 else:
#                     flux_key_now  = ctrNow + '_' + bgNow+'_' + 'rad' + '_' + str(staticRad) + '_None'
#
#                 example_wanderer_mean.compute_flux_over_time(staticRad=staticRad, varRad=varRad, \
#                                                          centering=ctrNow, background=bgNow,useTheForce=useTheForce)
#
#                 fluxNow     = example_wanderer_mean.flux_TSO_df[flux_key_now]
#                 medFluxNow  = median(example_wanderer_mean.flux_TSO_df[flux_key_now])
#
#                 # average SDNR per `staticRad`
#                 fluxNow    /= len(varRads)
#                 medFluxNow /= len(varRads)
#
#                 # Standard Deviation of the Normalized Residuals
#                 SDNR        = std(fluxNow / medFluxNow)*ppm
#                 example_wanderer_mean.SDNR_df[flux_key_now]  = [SDNR]
#
#                 print('Finished with Background ' + str(kBG) + ' ' + bgNow + ' with Centering ' + str(kCTR) \
#                           + ' ' + ctrNow + ' and staticRad ' + str(staticRad) + ' and varRad ' + str(varRad), \
#                       int(np.round(SDNR)), 'ppm')
#
# print('Saving `example_wanderer_mean` to a set of pickles for various Image Cubes and the Storage Dictionary')
# example_wanderer_mean.save_data_to_save_files(savefiledir='./SaveFiles/', saveFileNameHeader='Example_Wanderer_Mean_', saveFileType='.pickle.save')
#
