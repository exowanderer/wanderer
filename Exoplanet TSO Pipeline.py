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


# In[ ]:

rcParams['image.interpolation'] = 'None'
rcParams['image.cmap']          = 'Blues_r'
rcParams['axes.grid']           = False


# In[ ]:

dataDir = '/path/to/fits/files/main/directory/'
fitsFileDir = 'path/to/fits/subdirectories/'


# In[ ]:

fitsFilenames = glob(dataDir + fitsFileDir + '*slp.fits')
fitsFilenames


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
    print('\n\n** Not all who wander are lost **\n\n')
    def __init__(self, fitsFileDir = './', filetype = 'slp.fits', 
                 yguess=None, xguess=None, npix=10, method='mean'):
        
        y,x = 0,1
        
        if method == 'mean':
            self.metric  = mean
        elif method == 'median':
            self.metric  = median
        else:
            raise Exception("`method` must be from the list ['mean', 'median']")
        
        self.fitsFileDir  = fitsFileDir
        self.fitsFilenames = glob(self.fitsFileDir + '/*' + filetype)
        self.nSlopeFiles  = len(self.fitsFilenames)
        
        if self.nSlopeFiles == 0:
            print('Pipeline found no Files in ' + self.fitsFileDir + ' of type /*' + filetype)
            exit(-1)
        
        self.centering_df   = DataFrame()
        self.background_df  = DataFrame()
        self.flux_TSO_df    = DataFrame()
        self.noise_TSO_df   = DataFrame()
        
        testfits            = fits.open(self.fitsFilenames[0])[0]
        
        self.imageCube      = np.zeros((self.nSlopeFiles, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.noiseCube      = np.zeros((self.nSlopeFiles, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.timeCube       = np.zeros(self.nSlopeFiles)
        
        if yguess == None:
            self.yguess = self.imageCube.shape[y]//2
        else:
            self.yguess = yguess
        if xguess == None:
            self.xguess = self.imageCube.shape[x]//2
        else:
            self.xguess = xguess
        
        self.npix = npix
        
    def load_data_from_fits_files(self):
        
        nGroups        = len(self.fitsFilenames)
        for kframe, fname in enumerate(self.fitsFilenames):
            
            fitsNow = fits.open(fname)
            
            self.imageCube[kframe] = fitsNow[0].data[0]
            self.noiseCube[kframe] = fitsNow[0].data[1]
            
            # re-write these 4 lines into `get_julian_date_from_header`
            day2sec        = 86400.
            startJD,endJD     = get_julian_date_from_header(fitsNow[0].header)
            timeSpan          = (endJD - startJD)*day2sec/nGroups
            self.timeCube[kframe]  = startJD  + timeSpan*(kframe+0.5) / day2sec - 2450000.
            
            del fitsNow[0].data
            fitsNow.close()
            del fitsNow
    
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
        
        date            = localtime()
        
        year            = date.tm_year
        month           = date.tm_mon
        day             = date.tm_mday
        
        hour            = date.tm_hour
        minute          = date.tm_min
        sec             = date.tm_sec
        
        date_string     = '_' + str(year) + '-' + str(month)  + '-' + str(day) + '_' +                                 str(hour) + 'h' + str(minute) + 'm' + str(sec) + 's'
        
        saveFileTypeBak = date_string + saveFileType
        
        initiate_save_dict()
        
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
        self.centering_df.to_pickle(savefiledir  + saveFileNameHeader + '_centering_dataframe'  + saveFileTypeBak)
        self.background_df.to_pickle(savefiledir + saveFileNameHeader + '_background_dataframe' + saveFileTypeBak)
        self.flux_TSO_df.to_pickle(savefiledir   + saveFileNameHeader + '_flux_TSO_dataframe'   + saveFileTypeBak)
        
        joblib.dump(self.imageCube, savefiledir  + saveFileNameHeader + '_image_cube_array' + saveFileTypeBak)
        joblib.dump(self.noiseCube, savefiledir  + saveFileNameHeader + '_noise_cube_array' + saveFileTypeBak)
        joblib.dump(self.timeCube , savefiledir  + saveFileNameHeader + '_time_cube_array'  + saveFileTypeBak)
        
        joblib.dump(self.imageBadPixMasks, savefiledir  + saveFileNameHeader + '_image_bad_pix_cube_array' + saveFileTypeBak)
        
        joblib.dump(self.save_dict, savefiledir + saveFileNameHeader + '_save_dict' + saveFileTypeBak)
    
    def initiate_save_dict(self):
        
        self.save_dict  = {} # DataFrame() -- test if this works later
        
        self.save_dict['fitsFileDir']               = self.fitsFileDir
        self.save_dict['fitsFilenames']              = self.fitsFilenames
        
        self.save_dict['background_Annulus']        = self.background_Annulus
        self.save_dict['background_CircleMask']     = self.background_CircleMask
        self.save_dict['background_GaussMoment']    = self.background_GaussMoment
        self.save_dict['background_GaussianFit']    = self.background_GaussianFit
        self.save_dict['background_KDEUniv']        = self.background_KDEUniv
        self.save_dict['background_MedianMask']     = self.background_MedianMask
        self.save_dict['centering_FluxWeight']      = self.centering_FluxWeight
        self.save_dict['centering_GaussianFit']     = self.centering_GaussianFit
        self.save_dict['centering_GaussianMoment']  = self.centering_GaussianMoment
        self.save_dict['centering_LeastAsym']       = self.centering_LeastAsym
        self.save_dict['fitsFileDir']               = self.fitsFileDir
        self.save_dict['heights_GaussianFit']       = self.heights_GaussianFit
        self.save_dict['heights_GaussianMoment']    = self.heights_GaussianMoment
        # self.save_dict['']                          = self.imageCubeMAD
        # self.save_dict['']                          = self.imageCubeMedian
        self.save_dict['method']                    = self.method
        self.save_dict['npix']                      = self.npix
        self.save_dict['fitsFilenames']              = self.fitsFilenames
        self.save_dict['yguess']                    = self.yguess
        self.save_dict['xguess']                    = self.xguess
        self.save_dict['widths_GaussianFit']        = self.widths_GaussianFit
        self.save_dict['widths_GaussianMoment']     = self.widths_GaussianMoment
    
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
        
        self.imageCube[self.imageBadPixMasks] = nan
    
    def fit_gaussian_fitting_centering(self, method='la', initc='fw', print_compare=False):
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
        
        for kframe in range(self.imageCube.shape[0]):
            subFrameNow = self.imageCube[kframe][ylower:yupper, xlower:xupper]
            subFrameNow[isnan(subFrameNow)] = median(~isnan(subFrameNow))
            
            cmom    = np.array(moments(subFrameNow))  # H, Xc, Yc, Xs, Ys, O
            
            if method == 'ap':
                if initc == 'fw' and self.centering_FluxWeight.sum():
                    FWCNow    = self.centering_FluxWeight[kframe]
                    FWCNow[y] = FWCNow[y] - ylower
                    FWCNow[x] = FWCNow[x] - xlower
                    gaussI    = hstack([cmom[0], FWCNow, cmom[3:]])
                if initc == 'cm':
                    gaussI  = hstack([cmom[0], cmom[1], cmom[2], cmom[3:]])
                
                gaussP  = fit_gauss(subFrameNow, xinds, yinds, gaussI) # H, Xc, Yc, Xs, Ys, Th, O
            
            if method == 'la':
                gaussP  = fitgaussian(subFrameNow)#, xinds, yinds, np.copy(cmom)) # H, Xc, Yc, Xs, Ys, Th, O
            
            self.centering_GaussianFit[kframe][x]     = gaussP[1] + xlower
            self.centering_GaussianFit[kframe][y]     = gaussP[2] + ylower
            self.centering_GaussianMoment[kframe][x]  = cmom[1]   + xlower
            self.centering_GaussianMoment[kframe][y]  = cmom[2]   + ylower
            
            self.widths_GaussianFit[kframe][x]        = gaussP[3]
            self.widths_GaussianFit[kframe][y]        = gaussP[4]
            self.widths_GaussianMoment[kframe][x]     = cmom[3]
            self.widths_GaussianMoment[kframe][y]     = cmom[4]
            
            self.heights_GaussianFit[kframe]          = gaussP[0]
            self.heights_GaussianMoment[kframe]       = cmom[0]
            
            self.background_GaussianFit[kframe]       = gaussP[5]
            self.background_GaussMoment[kframe]       = cmom[5]
            
            if print_compare:
                print('Finished Frame ' + str(kframe) + ' with Yc = ' +                       str(self.centering_GaussianFit[kframe][y] - self.centering_GaussianMoment[kframe][y]) + '; Xc = ' +                       str(self.centering_GaussianFit[kframe][x] - self.centering_GaussianMoment[kframe][x]))
            
            del gaussP, cmom
        
        self.centering_df = DataFrame()
        self.centering_df['Gaussian_Fit_Y_Centers'] = self.centering_GaussianFit.T[y]
        self.centering_df['Gaussian_Fit_X_Centers'] = self.centering_GaussianFit.T[x]
        self.centering_df['Gaussian_Mom_Y_Centers'] = self.centering_GaussianFit.T[y]
        self.centering_df['Gaussian_Mom_X_Centers'] = self.centering_GaussianFit.T[x]
        
        self.centering_df['Gaussian_Fit_Y_Widths']  = self.widths_GaussianFit.T[y]
        self.centering_df['Gaussian_Fit_X_Widths']  = self.widths_GaussianFit.T[x]
        self.centering_df['Gaussian_Mom_Y_Widths']  = self.widths_GaussianMoment.T[y]
        self.centering_df['Gaussian_Mom_X_Widths']  = self.widths_GaussianMoment.T[x]
        
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
        self.centering_FluxWeight = np.zeros((self.nSlopeFiles, nFWCParams))
        
        for kframe in range(self.nSlopeFiles):
            subFrameNow = self.imageCube[kframe][ylower:yupper, xlower:xupper]
            subFrameNow[isnan(subFrameNow)] = median(~isnan(subFrameNow))
            
            self.centering_FluxWeight[kframe] = flux_weighted_centroid(self.imageCube[kframe], 
                                                                       self.yguess, self.xguess, bSize = 7)
            self.centering_FluxWeight[kframe] = self.centering_FluxWeight[kframe][::-1]
        
        self.centering_df['FluxWeighted_Y_Centers'] = self.centering_FluxWeight.T[y]
        self.centering_df['FluxWeighted_X_Centers'] = self.centering_FluxWeight.T[x]

    def fit_least_asymmetry_centering(self):
        
        y,x = 0,1
        
        yinds0, xinds0 = indices(self.imageCube[0].shape)
        
        ylower = self.yguess - self.npix
        yupper = self.yguess + self.npix
        xlower = self.xguess - self.npix
        xupper = self.xguess + self.npix
        
        ylower, xlower, yupper, xupper = np.int32([ylower, xlower, yupper, xupper])
        
        yinds = yinds0[ylower:yupper, xlower:xupper]
        xinds = xinds0[ylower:yupper, xlower:xupper]
        
        nAsymParams = 2 # Xc, Yc
        self.centering_LeastAsym  = np.zeros((self.nSlopeFiles, nAsymParams))
        
        for kframe in range(self.nSlopeFiles):
            # print(kframe, ylower,yupper, xlower,xupper) # (0 143 163 150 170)
            subFrameNow = self.imageCube[kframe][ylower:yupper, xlower:xupper]
            subFrameNow[isnan(subFrameNow)]   = median(subFrameNow)
            
            center_asym = actr(self.imageCube[kframe], [self.yguess, self.xguess])[0]
            try:
                self.centering_LeastAsym[kframe]  = center_asym[::-1]
            except:
                self.centering_LeastAsym[kframe]  = [nan,nan]
        
        self.centering_df['LeastAsymmetry_Y_Centers'] = self.centering_FluxWeight.T[y]
        self.centering_df['LeastAsymmetry_X_Centers'] = self.centering_FluxWeight.T[x]

    def fit_all_centering(self):
        print('Fit for Gaussian Fitting & Gaussian Moment Centers\n')
        self.fit_gaussian_fitting_centering()
        print('Fit for Flux Weighted Centers\n')
        self.fit_flux_weighted_centering()
        print('Fit for Least Asymmetry Centers\n')
        self.fit_least_asymmetry_centering()
    
    def measure_effective_width(self):
        self.effective_widths = self.imageCube.sum(axis=(1,2))**2. / ((self.imageCube)**2).sum(axis=(1,2))
        self.centering_df['Effective_Widths'] = self.effective_widths
    
    def measure_background_circle_masked(self, aperRad=None, method='mean'):
        """
            Assigning all zeros in the mask to NaNs because the `mean` and `median` 
                functions are set to `nanmean` functions, which will skip all NaNs
        """
        
        if aperRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                aperRad = 100
            else:
                aperRad = 10
        
        medianCenter   = median(self.centering_FluxWeight, axis=0)
        aperture       = CircularAperture(medianCenter, aperRad)
        backgroundMask = abs(aperture.get_fractions(np.ones(self.imageCube[0].shape))-1)
        backgroundMask[backgroundMask == 0] = nan
        
        self.background_CircleMask = self.metric(self.imageCube*backgroundMask,axis=(1,2))
        
        self.background_df['CircleMask'] = self.background_CircleMask
    
    def measure_background_annular_mask(self, innerRad=None, outerRad=None, method='mean'):
        
        if innerRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                innerRad = 100
            else:
                innerRad = 10
        
        if outerRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                outerRad = 150
            else:
                outerRad = 15
        
        medianCenter  = median(self.centering_LeastAsym, axis=0)
        
        innerAperture = CircularAperture(medianCenter, innerRad).get_fractions(np.ones(self.imageCube[0].shape))
        outerAperture = CircularAperture(medianCenter, outerRad).get_fractions(np.ones(self.imageCube[0].shape))
                
        backgroundMask= abs((outerAperture - innerAperture))
        backgroundMask[backgroundMask == 0] = nan
        
        self.background_Annulus = self.metric(self.imageCube*backgroundMask, axis=(1,2))
        self.background_df['AnnularMask'] = self.background_Annulus
    
    def measure_background_median_masked(self, aperRad=None, nSig=5, method='mean'):
        
        if aperRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                aperRad = 100
            else:
                aperRad = 10
        
        self.background_MedianMask  = np.zeros(self.nSlopeFiles)
        
        medianCenter   = median(self.centering_FluxWeight, axis=0)
        aperture       = CircularAperture(medianCenter, aperRad)
        backgroundMask = abs(aperture.get_fractions(np.ones(self.imageCube[0].shape))-1)
        
        for kframe in range(self.nSlopeFiles):
            medFrame  = median(self.imageCube[kframe])
            madFrame  = scale.mad(self.imageCube[kframe])
            
            medianMask= abs(self.imageCube[kframe] - medFrame) < nSig*madFrame
            
            maskComb  = medianMask*backgroundMask
            maskComb[maskComb == 0] = nan
            
            self.background_MedianMask[kframe] = self.metric(self.imageCube[kframe]*maskComb)
        
        self.background_df['MedianMask'] = self.background_MedianMask
    
    def measure_background_KDE_Mode(self, aperRad=None):
        
        if aperRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                aperRad = 100
            else:
                aperRad = 10
        
        self.background_KDEUniv = np.zeros(self.nSlopeFiles)
        
        medianCenter   = median(self.centering_FluxWeight, axis=0)
        aperture       = CircularAperture(medianCenter, aperRad)
        backgroundMask = abs(aperture.get_fractions(np.ones(self.imageCube[0].shape))-1)
        
        for kframe in range(self.nSlopeFiles):
            frameNow = (self.imageCube[kframe]*backgroundMask).ravel()
            kdeFrame = kde.KDEUnivariate(frameNow[np.where(backgroundMask.ravel() != 0.0)])
            kdeFrame.fit()
            
            self.background_KDEUniv[kframe] = kdeFrame.support[kdeFrame.density.argmax()]
        
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
    
    def compute_flux_over_time(self, aperRad=None, centering='LeastAsymmetry', background='AnnularMask'):
        y,x = 0,1
        
        if background not in self.background_df.columns:
            raise Exception("`background` must be in", self.background_df.columns)
        
        if centering not in ['Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', 'LeastAsymmetry']:
            raise Exception("`centering` must be either 'Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', or 'LeastAsymmetry'")
        
        if aperRad is None:
            if 'wlp' in self.fitsFilenames[0].lower():
                aperRad = 70
            else:
                aperRad = 3
        
        centering_Use = np.transpose([self.centering_df[centering + '_Y_Centers'], 
                                      self.centering_df[centering + '_X_Centers']])
        
        background_Use= self.background_df[background]
        
        flux_key_now  = centering + '_' + background+'_' + 'rad' + '_' + str(aperRad)
        flux_TSO_now  = np.zeros(self.nSlopeFiles)
        noise_TSO_now = np.zeros(self.nSlopeFiles)
        for kframe in range(self.nSlopeFiles):
            frameNow  = np.copy(self.imageCube[kframe]) - background_Use[kframe]
            frameNow[np.isnan(frameNow)] = median(frameNow)
            
            noiseNow  = np.copy(self.noiseCube[kframe])**2.
            noiseNow[np.isnan(noiseNow)] = median(noiseNow)
            
            aperture  = CircularAperture([centering_Use[kframe][x], centering_Use[kframe][y]], r=aperRad)
            
            flux_TSO_now[kframe]  = aperture_photometry(frameNow, aperture)['aperture_sum']
            noise_TSO_now[kframe] = sqrt(aperture_photometry(noiseNow, aperture)['aperture_sum'])
        
        self.flux_TSO_df[flux_key_now]  = flux_TSO_now
        self.noise_TSO_df[flux_key_now] = noise_TSO_now

tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst = localtime()
print('Completed Class Definition at ' +
      str(tm_year) + '-' + str(tm_mon) + '-' + str(tm_mday) + ' ' + \
      str(tm_hour) + 'h' + str(tm_min) + 'm' + str(tm_sec) + 's')


# Necessary Constants for Both Module A and Module B
# ---

# In[ ]:

ppm             = 1e6
y,x             = 0,1

yguess, xguess  = 160., 167. # Specific to JWST WLP Test Data
filetype        = 'slp.fits' # Specific to JWST WLP Test Data


# Load Stored Instance from Save Files
# ---

# In[ ]:

dataDir     = '/path/to/fits/files/main/directory/'
fitsFileDir = 'path/to/fits/subdirectories/'

loadfitsdir = dataDir + fitsFileDir


# In[ ]:

method = 'mean'
example_wanderer_mean = wanderer(fitsFileDir=loadfitsdir, filetype=filetype, 
                                            yguess=yguess, xguess=xguess, method=method)

example_wanderer_mean.load_data_from_save_files(savefiledir='./SaveFiles/', 
                                                     saveFileNameHeader='Example_Wanderer_Mean_', saveFileType='.pickle.save')


# In[ ]:

method = 'median'
example_wanderer_median = wanderer(fitsFileDir=loadfitsdir_ModA, filetype=filetype, 
                                            yguess=yguess, xguess=xguess, method=method)

example_wanderer_median.load_data_from_save_files(savefiledir='./SaveFiles/', saveFileNameHeader='Example_Wanderer_Median_', saveFileType='.pickle.save')


# Start a New Instance with Median for the Metric
# ---

# In[ ]:

dataDir     = '/path/to/fits/files/main/directory/'
fitsFileDir = 'path/to/fits/subdirectories/'

loadfitsdir = dataDir + fitsFileDir


# In[ ]:

method = 'median'

print('Initialize an instance of `wanderer` as `example_wanderer_median`\n')
example_wanderer_median = wanderer(fitsFileDir=loadfitsdir_ModB, filetype=filetype, 
                                            yguess=yguess, xguess=xguess, method=method)

print('Load Data From Fits Files in ' + fitsFileDir_ModB + '\n')
example_wanderer_median.load_data_from_fits_files()

print('Skipping Load Data From Save Files in ' + fitsFileDir_ModB + '\n')
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


# Start a New Instance with Mean for the Metric
# ---

# In[ ]:

method = 'mean'

print('Initialize an instance of `wanderer` as `example_wanderer_mean`')
example_wanderer_mean = wanderer(fitsFileDir=loadfitsdir_ModB, filetype = filetype, 
                                yguess=yguess, xguess=xguess, method=method)

print('Load Data From Fits Files in ' + loadfitsdir)
example_wanderer_mean.load_data_from_fits_files()

print('Skipping Load Data From Save Files in ' + loadfitsdir)
# example_wanderer_mean.load_data_from_save_files()

print('Find, flag, and NaN the "Bad Pixels" Outliers')
example_wanderer_mean.find_bad_pixels()

print('Fit for All Centers: Flux Weighted, Gaussian Fitting, Gaussian Moments, Least Asymmetry')
# example_wanderer_mean.fit_gaussian_fitting_centering()
# example_wanderer_mean.fit_flux_weighted_centering()
# example_wanderer_mean.fit_least_asymmetry_centering()
example_wanderer_mean.fit_all_centering()

print('Measure Background Estimates with All Methods: Circle Masked, Annular Masked, KDE Mode, Median Masked')
# example_wanderer_mean.measure_background_circle_masked()
# example_wanderer_mean.measure_background_annular_mask()
# example_wanderer_mean.measure_background_KDE_Mode()
# example_wanderer_mean.measure_background_median_masked()
example_wanderer_mean.measure_all_background()

print('Iterating over Background Techniques, Centering Techniques, Aperture Radii')
background_choices = example_wanderer_mean.background_df.columns
centering_choices  = ['Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', 'LeastAsymmetry']
aperRads           = np.arange(1, 100.5,0.5)

start = time()
for bgNow in background_choices:
    for ctrNow in centering_choices:
        for aperRad in aperRads:
            print('Working on Background ' + bgNow + ' with Centering ' + ctrNow + ' and AperRad ' + str(aperRad), end=" ")
            example_wanderer_mean.compute_flux_over_time(aperRad=aperRad, centering=ctrNow, background=bgNow)
            flux_key_now  = ctrNow + '_' + bgNow+'_' + 'rad' + '_' + str(aperRad)
            print(std(example_wanderer_mean.flux_TSO_df[flux_key_now] / median(example_wanderer_mean.flux_TSO_df[flux_key_now]))*ppm)

print('Operation took: ', time()-start)

print('Saving `example_wanderer_mean` to a set of pickles for various Image Cubes and the Storage Dictionary')
example_wanderer_mean.save_data_to_save_files(savefiledir='./SaveFiles/', saveFileNameHeader='Example_Wanderer_Mean_', saveFileType='.pickle.save')

