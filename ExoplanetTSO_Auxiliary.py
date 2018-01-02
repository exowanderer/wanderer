from astroML.plotting          import hist
from astropy.io                import fits
from astropy.modeling          import models, fitting
from datetime                  import datetime
from functools                 import partial
# from image_registration        import cross_correlation_shifts
from glob                      import glob
from lmfit                     import Model, Parameters
from matplotlib.ticker         import MaxNLocator
from matplotlib                import style, colors
from multiprocessing           import cpu_count, Pool
from os                        import listdir, path, mkdir, chdir
from pandas                    import DataFrame, Series, read_csv, read_pickle, scatter_matrix
from photutils                 import CircularAperture, CircularAnnulus, aperture_photometry, findstars
from least_asymmetry.asym      import actr
from pylab                     import ion, gcf, sort, linspace, indices, nanmedian as median, nanmean as mean, nanstd as std, empty, figure, transpose, ceil
from pylab                     import concatenate, pi, sqrt, ones, diag, inf, rcParams, isnan, isfinite, array, nanmax, shape, zeros
from numpy                     import min as npmin, max as npmax, zeros, arange, sum, float, isnan, hstack, int32, exp
from numpy                     import int32 as npint, round as npround, nansum as sum, std as std, where, bitwise_and
from seaborn                   import *
from scipy.special             import erf
from scipy                     import stats
from sklearn.cluster           import DBSCAN
from sklearn.externals         import joblib
from sklearn.preprocessing     import StandardScaler
from skimage.filters           import gaussian as gaussianFilter
from socket                    import gethostname
from statsmodels.robust        import scale
from statsmodels.nonparametric import kde
from sys                       import exit
from time                      import time, localtime, sleep
from tqdm                      import tnrange, tqdm_notebook

from scipy                     import optimize
from sklearn.externals         import joblib
from skimage.filters           import gaussian as gaussianFilter
from sys                       import exit
import numpy as np

rcParams['image.interpolation'] = 'None'
rcParams['image.cmap']          = 'Blues_r'
rcParams['axes.grid']           = False

y,x = 0,1

'''Start: From least_asymmetry.asym by N.Lust (github.com/natelust/least_asymmetry) and modified (reversed XY -> YX)'''
def gaussian(height, center_y, center_x, width_y, width_x, offset, yy, xx):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    """Returns a gaussian function with the given parameters"""
    
    width_y = float(width_y)
    width_x = float(width_x)
    return height*np.exp(-0.5*(((center_y-yy)/width_y)**2+((center_x-xx)/width_x)**2))+offset

def moments(data, kernel_size=2):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    """Returns (height, x, y, width_x, width_y,offset)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    
    total = data.sum()
    Y, X = indices(data.shape)
    y = (Y*data).sum()/total
    x = (X*data).sum()/total
    height = gaussianFilter(data, kernel_size).max()
    firstq = median(data[data < median(data)])
    thirdq = median(data[data > median(data)])
    offset = median(data[np.where(np.bitwise_and(data > firstq,
                                                    data < thirdq))])
    places = where((data-offset) > 4*std(data[where(np.bitwise_and(data > firstq, data < thirdq))]))
    width_y = std(places[0])
    width_x = std(places[1])
    # These if statements take into account there might only be one significant
    # point above the background when that is the case it is assumend the width
    # of the gaussian must be smaller than one pixel
    if width_y == 0.0: width_y = 0.5
    if width_x == 0.0: width_x = 0.5
    
    height -= offset
    
    return height, y, x, width_y, width_x, offset

def fitgaussian(data, weights=False):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    """Returns (height, y, x, width_y, width_x)
    the gaussian parameters of a 2D distribution found by a fit
    Weights must be the same size as the data, but every point
    contains the value of the weight of the pixel"""
    if isinstance(weights, type(False)):
        weights = np.ones(data.shape, dtype=float)
    elif weights.dtype != np.dtype('float'):
        weights = np.array(weights, dtype=float)
    params = moments(data)
    
    yy,xx  = indices(data.shape)
    
    gausspartial = partial(gaussian, yy=yy, xx=xx)
    
    errorfunction = lambda p: np.ravel((gausspartial(*p) - data)*weights)
    params, success = optimize.leastsq(errorfunction, params)
    
    return params

def center_of_light(data, weights=False):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    if isinstance(weights, type(False)):
        weights = np.ones(data.shape, dtype=float)
    elif weights.dtype != np.dtype('float'):
        weights = np.array(weights, dtype=float)
    
    ny, nx = np.indices(data.shape)
    
    return [sum(weights*ny*data)/sum(weights*data), sum(weights*nx*data)/sum(weights*data)]

def get_julian_date_from_gregorian_date(*date):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
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
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
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

def clipOutlier(oneDarray, nSig=8):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    medarray = np.median(oneDarray)
    stdarray = np.std(oneDarray)
    outliers = abs(oneDarray - medarray) > nSig*stdarray
    
    oneDarray[outliers]= np.median(oneDarray[~outliers])
    return oneDarray

def flux_weighted_centroid(image, ypos, xpos, bSize = 7):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
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


def gaussian(height, center_y, center_x, width_y, width_x, offset, yy, xx):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    
    chiY    = (center_y - yy) / width_y
    chiX    = (center_x - xx) / width_x
    
    return height * exp(-0.5*(chiY**2 + chiX**2)) + offset

def moments(data):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    """Returns (height, x, y, width_x, width_y,offset)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    height = data.max()
    firstq = median(data[data < median(data)])
    thirdq = median(data[data > median(data)])
    offset = median(data[where(bitwise_and(data > firstq,
                                                    data < thirdq))])
    places = where((data-offset) > 4*std(data[where(bitwise_and(
                                      data > firstq, data < thirdq))]))
    width_y = std(places[0])
    width_x = std(places[1])
    # These if statements take into account there might only be one significant
    # point above the background when that is the case it is assumend the width
    # of the gaussian must be smaller than one pixel
    if width_y == 0.0:
        width_y = 0.5
    if width_x == 0.0:
        width_x = 0.5
    
    height -= offset
    return height, y, x, width_y, width_x, offset

def lame_lmfit_gaussian_centering(imageCube, yguess=15, xguess=15, subArraySize=10, init_params=None, nSig=False, useMoments=False, method='leastsq'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    imageSize  = imageCube.shape[1]
    
    nparams    = 6
    if init_params is None:
        useMoments = True
        init_params = moments(imageCube[0])
    
    ihg, iyc, ixc, iyw, ixw, ibg  = arange(nparams)
    lmfit_init_params = Parameters()
    lmfit_init_params.add_many(
        ('height'  , init_params[ihg], True  , 0.0 , inf   ),
        ('center_y', init_params[iyc], True  , 0.0 , imageSize),
        ('center_x', init_params[ixc], True  , 0.0 , imageSize),
        ('width_y' , init_params[iyw], True  , 0.0 , imageSize),
        ('width_x' , init_params[ixw], True  , 0.0 , imageSize),
        ('offset'  , init_params[ibg], True))
    
    gfit_model = Model(gaussian, independent_vars=['yy', 'xx'])
    
    yy0, xx0 = indices(imageCube[0].shape)
    
    npix   = subArraySize//2
    ylower = yguess - npix
    yupper = yguess + npix
    xlower = xguess - npix
    xupper = xguess + npix
    
    ylower, xlower, yupper, xupper = int32([ylower, xlower, yupper, xupper])
    
    yy = yy0[ylower:yupper, xlower:xupper]
    xx = xx0[ylower:yupper, xlower:xupper]
    
    heights, ycenters, xcenters, ywidths, xwidths, offsets = zeros((nparams, nFrames))
    
    for k, image in enumerate(imageCube):
        subFrameNow = image[ylower:yupper, xlower:xupper]
        subFrameNow[isnan(subFrameNow)] = median(subFrameNow)
        
        subFrameNow = gaussianFilter(subFrameNow, nSig) if not isinstance(nSig, bool) else subFrameNow
        
        init_params = moments(subFrameNow) if useMoments else init_params
        
        gfit_res    = gfit_model.fit(subFrameNow, params=lmfit_init_params, xx=xx, yy=yy, method=method)
        
        heights[k]  = gfit_res.best_values['height']
        ycenters[k] = gfit_res.best_values['center_y']
        xcenters[k] = gfit_res.best_values['center_x']
        ywidths[k]  = gfit_res.best_values['width_y']
        xwidths[k]  = gfit_res.best_values['width_x']
        offsets[k]  = gfit_res.best_values['offset']
    
    return heights, ycenters, xcenters, ywidths, xwidths, offsets

def lmfit_one_center(image, yy, xx, gfit_model, lmfit_init_params, yupper, ylower, xupper, xlower, useMoments=True, nSig=False, method='leastsq'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    subFrameNow = image[ylower:yupper, xlower:xupper]
    subFrameNow[isnan(subFrameNow)] = median(subFrameNow)
    
    subFrameNow = gaussianFilter(subFrameNow, nSig) if not isinstance(nSig, bool) else subFrameNow
    
    init_params = moments(subFrameNow) if useMoments else list(lmfit_init_params.valuesdict().values())
    
    nparams     = 6
    ihg, iyc, ixc, iyw, ixw, ibg  = arange(nparams)
    
    lmfit_init_params.height   = init_params[ihg]
    lmfit_init_params.center_y = init_params[iyc]
    lmfit_init_params.center_x = init_params[ixc]
    lmfit_init_params.widths_y = init_params[iyw]
    lmfit_init_params.widths_x = init_params[ixw]
    lmfit_init_params.offset   = init_params[ibg]
    
    # print(lmfit_init_params)
    
    gfit_res    = gfit_model.fit(subFrameNow, params=lmfit_init_params, xx=xx, yy=yy, method=method)
    
    return gfit_res.best_values

def fit_gauss(subFrameNow, xinds, yinds, initParams, print_compare=False):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
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

def fit_one_center(image, ylower, yupper, xlower, xupper, nSig=False, method='gauss', bSize = 7):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    subFrameNow = image[ylower:yupper, xlower:xupper]
    subFrameNow[isnan(subFrameNow)] = median(~isnan(subFrameNow))
    
    subFrameNow = gaussianFilter(subFrameNow, nSig) if not isinstance(nSig, bool) else subFrameNow
    
    if method == 'cmom':
        return np.array(moments(subFrameNow))  # H, Xc, Yc, Xs, Ys, O
    if method == 'gauss':
        return fitgaussian(subFrameNow)#, xinds, yinds, np.copy(cmom)) # H, Xc, Yc, Xs, Ys, Th, O
    if method == 'fwc':
        return flux_weighted_centroid(image, image.shape[y]//2, image.shape[x]//2, bSize = bSize)[::-1]

def compute_flux_one_frame(image, center, background, aperRad=3.0):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    frameNow  = image - background
    frameNow[np.isnan(frameNow)] = median(frameNow)
    
    aperture  = CircularAperture([center[x], center[y]], r=abs(aperRad))
    
    return aperture_photometry(frameNow, aperture)['aperture_sum'].data[0]

def measure_one_circle_bg(image, center, aperRad, metric, apMethod='exact'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    aperture  = CircularAperture(center, aperRad)
    aper_mask = aperture.to_mask(method=apMethod)[0]    # list of ApertureMask objects (one for each position)
    
    # backgroundMask = abs(aperture.get_fractions(np.ones(self.imageCube[0].shape))-1)
    backgroundMask = aper_mask.to_image(image.shape).astype(bool)
    backgroundMask = ~backgroundMask#[backgroundMask == 0] = False
    
    return metric(image[backgroundMask])

def measure_one_annular_bg(image, center, innerRad, outerRad, metric, apMethod='exact'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    innerAperture   = CircularAperture(center, innerRad)
    outerAperture   = CircularAperture(center, outerRad)
    
    inner_aper_mask = innerAperture.to_mask(method=apMethod)[0]
    inner_aper_mask = inner_aper_mask.to_image(image.shape).astype(bool)
    
    outer_aper_mask = outerAperture.to_mask(method=apMethod)[0]
    outer_aper_mask = outer_aper_mask.to_image(image.shape).astype(bool)
    
    backgroundMask = (~inner_aper_mask)*outer_aper_mask
    
    return metric(image[backgroundMask])

from numpy import median, std
def measure_one_median_bg(image, center, aperRad, metric, nSig, apMethod='exact'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    aperture       = CircularAperture(center, aperRad)
    aperture       = aperture.to_mask(method=apMethod)[0]
    aperture       = aperture.to_image(image.shape).astype(bool)
    backgroundMask = ~aperture
    
    medFrame  = median(image[backgroundMask])
    madFrame  = std(image[backgroundMask])
    
    medianMask= abs(image - medFrame) < nSig*madFrame
    
    maskComb  = medianMask*backgroundMask
    
    return median(image[maskComb])

def measure_one_KDE_bg(image, center, aperRad, metric, apMethod='exact'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    aperture       = CircularAperture(center, aperRad)
    aperture       = aperture.to_mask(method=apMethod)[0]
    aperture       = aperture.to_image(image.shape).astype(bool)
    backgroundMask = ~aperture
    
    kdeFrame = kde.KDEUnivariate(image[backgroundMask])
    kdeFrame.fit()
    
    return kdeFrame.support[kdeFrame.density.argmax()]

def measure_one_background(image, center, aperRad, metric, apMethod='exact', bgMethod='circle'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    if np.ndim(aperRad) == 0:
        aperture  = CircularAperture(center, aperRad)
        aperture  = aperture.to_mask(method=apMethod)[0]    # list of ApertureMask objects (one for each position)
        aperture  = ~aperture.to_image(image).astype(bool) # inverse to keep 'outside' aperture
    else:
        innerRad, outerRad = aperRad
        
        innerAperture   = CircularAperture(center, innerRad)
        outerAperture   = CircularAperture(center, outerRad)
        
        inner_aper_mask = innerAperture.to_mask(method=method)[0]
        inner_aper_mask = inner_aper_mask.to_image(image.shape).astype(bool)
    
        outer_aper_mask = outerAperture.to_mask(method=method)[0]
        outer_aper_mask = outer_aper_mask.to_image(image.shape).astype(bool)     
        
        aperture        = (~inner_aper_mask)*outer_aper_mask
    
    if bgMethod == 'median':
        medFrame  = median(image[aperture])
        madFrame  = scale.mad(image[aperture])
        
        medianMask= abs(image - medFrame) < nSig*madFrame
        
        aperture  = medianMask*aperture
    
    if bgMethod == 'kde':
        kdeFrame = kde.KDEUnivariate(image[aperture].ravel())
        kdeFrame.fit()
        
        return kdeFrame.support[kdeFrame.density.argmax()]
    
    return metric(image[aperture])

def DBScan_Flux(phots, ycenters, xcenters, dbsClean=0, useTheForce=False):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    
    dbsPhots    = DBSCAN()#n_jobs=-1)
    stdScaler   = StandardScaler()
    
    phots       = np.copy(phots.ravel())
    phots[~np.isfinite(phots)] = np.median(phots[np.isfinite(phots)])
    
    featuresNow = np.transpose([stdScaler.fit_transform(ycenters[:,None]).ravel(), \
                                stdScaler.fit_transform(xcenters[:,None]).ravel(), \
                                stdScaler.fit_transform(phots[:,None]).ravel()   ] )
    
    dbsPhotsPred= dbsPhots.fit_predict(featuresNow)
    
    return dbsPhotsPred == dbsClean

def DBScan_PLD(PLDNow, dbsClean=0, useTheForce=False):
    """Class methods are similar to regular functions.
    
    Note:
        Do not include the `self` parameter in the ``Args`` section.
    
    Args:
        param1: The first parameter.
        param2: The second parameter.
    
    Returns:
        True if successful, False otherwise.
    
    """
    
    dbsPLD      = DBSCAN()#n_jobs=-1)
    stdScaler   = StandardScaler()
    
    dbsPLDPred= dbsPLD.fit_predict(stdScaler.fit_transform(PLDNow[:,None]))
    
    return dbsPLDPred == dbsClean

class wanderer(object):
    """The summary line for a class docstring should fit on one line.

        If the class has public attributes, they may be documented here
        in an ``Attributes`` section and follow the same formatting as a
        function's ``Args`` section. Alternatively, attributes may be documented
        inline with the attribute's declaration (see __init__ method below).

        Properties created with the ``@property`` decorator should be documented
        in the property's getter method.

        Attributes:
            attr1 (str): Description of `attr1`.
            attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    
    def __init__(self, fitsFileDir = './', filetype = 'slp.fits', telescope = None,
                 yguess=None, xguess=None, npix=5, method='mean', nCores = None):
        """Example of docstring on the __init__ method.

                The __init__ method may be documented in either the class level
                docstring, or as a docstring on the __init__ method itself.

                Either form is acceptable, but the two should not be mixed. Choose one
                convention to document the __init__ method and be consistent with it.

                Note:
                    Do not include the `self` parameter in the ``Args`` section.

                Args:
                    param1 (str): Description of `param1`.
                    param2 (:obj:`int`, optional): Description of `param2`. Multiple
                        lines are supported.
                    param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
        print('\n\n** Not all who wander are lost **\n\n')
        
        self.method   = method
        self.filetype = filetype
        
        self.y,self.x = 0,1
        
        self.day2sec  = 86400.
        if method == 'mean':
            self.metric  = mean
        elif method == 'median':
            self.metric  = median
        else:
            raise Exception("`method` must be from the list ['mean', 'median']")
        
        self.fitsFileDir    = fitsFileDir
        self.fitsFilenames  = glob(self.fitsFileDir + '/*' + self.filetype)
        self.nSlopeFiles    = len(self.fitsFilenames)
        
        if telescope is None:
            raise Exception("Please specify `telescope` as either 'JWST' or 'Spitzer' or 'HST'")
        
        self.telescope    = telescope
        
        if self.telescope == 'Spitzer':
            fitsdir_split = self.fitsFileDir.replace('raw', 'cal').split('/')
            for _ in range(4):
                fitsdir_split.pop()
            
            # self.calDir        = ''
            # for thing in fitsdir_split:
            #     self.calDir = self.calDir + thing + '/' 
            # 
            # self.permBadPixels = fits.open(self.calDir + 'nov14_ch1_bcd_pmask_subarray.fits')
        
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
        
        self.nCores = cpu_count()//2 if nCores is None else int(nCores)
        
        tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst = localtime()
        print('Completed Class Definition at ' +
              str(tm_year) + '-' + str(tm_mon) + '-' + str(tm_mday) + ' ' + \
              str(tm_hour) + 'h' + str(tm_min) + 'm' + str(tm_sec) + 's')
        
    def jwst_load_fits_file(self):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        testfits            = fits.open(self.fitsFilenames[0])[0]
        
        self.nFrames        = self.nSlopeFiles
        self.imageCube      = np.zeros((self.nFrames, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.noiseCube      = np.zeros((self.nFrames, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.timeCube       = np.zeros(self.nFrames)
        
        del testfits
        
        for kf, fname in tqdm_notebook(enumerate(self.fitsFilenames), desc='JWST Load File', leave = False, total=self.nSlopeFiles):
            fitsNow = fits.open(fname)
            
            self.imageCube[kf] = fitsNow[0].data[0]
            self.noiseCube[kf] = fitsNow[0].data[1]

            # re-write these 4 lines into `get_julian_date_from_header`
            startJD,endJD          = get_julian_date_from_header(fitsNow[0].header)
            timeSpan               = (endJD - startJD)*day2sec / self.nFrames
            self.timeCube[kf]  = startJD  + timeSpan*(kf+0.5) / day2sec - 2450000.
            
            del fitsNow[0].data
            fitsNow.close()
            del fitsNow
    
    def spitzer_load_fits_file(self, outputUnits='electrons', remove_nans=True):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        # BMJD_2_BJD     = -0.5
        from scipy.constants import arcsec # pi / arcsec = 648000
        
        sec2day        = 1/(24*3600)
        nFramesPerFile = 64
        
        testfits       = fits.open(self.fitsFilenames[0])[0]
        testheader     = testfits.header
        
        bcd_shape      = testfits.data[0].shape
        
        self.nFrames        = self.nSlopeFiles * nFramesPerFile
        self.imageCube      = zeros((self.nFrames, bcd_shape[0], bcd_shape[1]))
        self.noiseCube      = zeros((self.nFrames, bcd_shape[0], bcd_shape[1]))
        self.timeCube       = zeros(self.nFrames)
        
        del testfits
        
        # Converts DN/s to microJy per pixel
        #   1) expTime * gain / fluxConv converts MJ/sr to electrons
        #   2) as2sr * MJ2mJ * testheader['PXSCAL1'] * testheader['PXSCAL2'] converts MJ/sr to muJ/pixel
        if outputUnits == 'electrons':
            fluxConv  = testheader['FLUXCONV']
            expTime   = testheader['EXPTIME']
            gain      = testheader['GAIN']
            fluxConversion = expTime*gain / fluxConv
        elif outputUnits == 'muJ_per_Pixel':
            as2sr = arcsec**2.0 # steradians per square arcsecond
            MJ2mJ = 1e12        # mircro-Janskeys per Mega-Jansky
            fluxConversion = abs(as2sr * MJ2mJ * testheader['PXSCAL1'] * testheader['PXSCAL2']) # converts MJ/
        else:
            raise Exception("`outputUnits` must be either 'electrons' or 'muJ_per_Pixel'")
        
        print('Loading Spitzer Data')
        for kfile, fname in tqdm_notebook(enumerate(self.fitsFilenames),
                                          desc='Spitzer Load File', leave = False, total=self.nSlopeFiles):
            bcdNow  = fits.open(fname)
            # buncNow = fits.open(fname[:-len(filetype)] + 'bunc.fits')
            buncNow = fits.open(fname.replace('bcd.fits', 'bunc.fits'))
            
            for iframe in range(nFramesPerFile):
                self.timeCube[kfile * nFramesPerFile + iframe]  = bcdNow[0].header['BMJD_OBS'] \
                                                                + (bcdNow[0].header['ET_OBS'] - bcdNow[0].header['UTCS_OBS'])/86400 \
                                                                + iframe * float(bcdNow[0].header['FRAMTIME']) * sec2day
                
                self.imageCube[kfile * nFramesPerFile + iframe] = bcdNow[0].data[iframe]  * fluxConversion
                self.noiseCube[kfile * nFramesPerFile + iframe] = buncNow[0].data[iframe] * fluxConversion
            
            del bcdNow[0].data
            bcdNow.close()
            del bcdNow
            
            del buncNow[0].data
            buncNow.close()
            del buncNow
        
        if remove_nans:
            self.imageCube[where(isnan(self.imageCube))] = median(self.imageCube)
    
    def hst_load_fits_file(fitsNow):
        """Not Yet Implemented"""
        raise Exception('NEED TO CODE THIS')
    
    def load_data_from_fits_files(self, remove_nans=True):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        if self.telescope == 'JWST':
            self.jwst_load_fits_file()
        
        if self.telescope == 'Spitzer':
            self.spitzer_load_fits_file(remove_nans=remove_nans)
        
        if self.telescope == 'HST':
            self.hst_load_fits_file()
        
    def load_data_from_save_files(self, savefiledir=None, saveFileNameHeader=None, saveFileType='.pickle.save'):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
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
        
        print('nFrames', 'nFrame' in self.save_dict.keys())
        print('Assigning Parts of `self.save_dict` to individual data structures')
        for key in self.save_dict.keys():
            exec("self." + key + " = self.save_dict['" + key + "']")
        
    def save_data_to_save_files(self, savefiledir=None, saveFileNameHeader=None, saveFileType='.pickle.save'):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
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
        
        print('\nSaving to Master File -- Overwriting Previous Master')
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
    
    def initiate_save_dict(self, dummy=None):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        _stored_variables = ['background_Annulus', 'background_CircleMask', 'background_GaussianFit', 'background_KDEUniv',
                             'background_MedianMask','centering_FluxWeight', 'centering_GaussianFit', 'centering_LeastAsym',
                             'effective_widths', 'fitsFileDir', 'fitsFilenames', 'heights_GaussianFit', 'inliers_Phots',
                             'inliers_PLD', 'inliers_Master', 'method', 'npix', 'nFrames', 'PLD_components', 'PLD_norm',
                             'quadrature_widths', 'yguess', 'xguess', 'widths_GaussianFit']
        
        _max_str_len = 0
        for thing in _stored_variables:
            if len(thing) > _max_str_len:
                _max_str_len = len(thing)
        
        print('Storing in `self.save_dict`: ')
        self.save_dict = {} # DataFrame does not work because each all `columns` must be 1D and have the same length
        for key, val in self.__dict__.items():
            self.save_dict[key] = val
            print('\tself.' + key)# + ' into save_dict')#+ ' '*(len(key) - _max_str_len)
        
        print('\n')
        
        notYet = True
        for thing in _stored_variables:
            if thing not in self.save_dict.keys():
                if notYet:
                    print('The following do not yet exist:')
                    notYet = False
                
                print("\tself."+thing)#+" does not yet exist")
        
        # for thing in _stored_variables:
        #     exec("try: self.save_dict['"+thing+"'] = self."+thing+"\nexcept: print('self."+thing+" does not yet exist')")
        #
        # try:self.save_dict['background_Annulus']      = self.background_Annulus
        # except: pass
        #
        # try:self.save_dict['background_CircleMask']   = self.background_CircleMask
        # except: pass
        #
        # try:self.save_dict['background_GaussianFit']  = self.background_GaussianFit
        # except: pass
        #
        # try:self.save_dict['background_KDEUniv']      = self.background_KDEUniv
        # except: pass
        #
        # try:self.save_dict['background_MedianMask']   = self.background_MedianMask
        # except: pass
        #
        # try:self.save_dict['centering_FluxWeight']    = self.centering_FluxWeight
        # except: pass
        #
        # try:self.save_dict['centering_GaussianFit']   = self.centering_GaussianFit
        # except: pass
        #
        # try:self.save_dict['centering_LeastAsym']     = self.centering_LeastAsym
        # except: pass
        #
        # try:self.save_dict['effective_widths']        = self.effective_widths
        # except: pass
        #
        # try:self.save_dict['fitsFileDir']             = self.fitsFileDir
        # except: pass
        #
        # try:self.save_dict['fitsFilenames']           = self.fitsFilenames
        # except: pass
        #
        # try:self.save_dict['heights_GaussianFit']     = self.heights_GaussianFit
        # except: pass
        #
        # try:self.save_dict['inliers_Phots']           = self.inliers_Phots
        # except: pass
        #
        # try:self.save_dict['inliersPLD']              = self.inliers_PLD
        # except: pass
        #
        # try:self.save_dict['method']                  = self.method
        # except: pass
        #
        # try:self.save_dict['npix']                    = self.npix
        # except: pass
        #
        # try:self.save_dict['nFrames']                 = self.nFrames
        # except: pass
        #
        # try:self.save_dict['PLD_components']          = self.PLD_components
        # except: pass
        #
        # try:self.save_dict['PLD_norm']                = self.PLD_norm
        # except: pass
        #
        # try:self.save_dict['quadrature_widths']       = self.quadrature_widths
        # except: pass
        #
        # try:self.save_dict['yguess']                  = self.yguess
        # except: pass
        #
        # try:self.save_dict['xguess']                  = self.xguess
        # except: pass
        #
        # try:self.save_dict['widths_GaussianFit']      = self.widths_GaussianFit
        # except: pass
    
    def copy_instance(self):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
               
        temp = wanderer(fitsFileDir = self.fitsFileDir, filetype = self.filetype, 
                        telescope = self.telescope, yguess=self.yguess, xguess=self.xguess, 
                        npix=self.npix, method=self.method, nCores = self.nCores)
        
        temp.centering_df       = self.centering_df
        temp.background_df      = self.background_df
        temp.flux_TSO_df        = self.flux_TSO_df
        temp.noise_TSO_df       = self.noise_TSO_df
        
        temp.imageCube          = self.imageCube
        temp.noiseCube          = self.noiseCube
        temp.timeCube           = self.timeCube
        
        temp.imageBadPixMasks   = self.imageBadPixMasks
        
        print('Assigning Parts of `temp.save_dict` to from `self.save_dict`')
        temp.save_dict = self.save_dict
        for thing in self.save_dict.keys():
            exec("temp." + thing + " = self.save_dict['" + thing + "']")
        
        return temp
    
    def find_bad_pixels(self, nSig=5):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        # we chose 5 arbitrarily, but from experience
        self.imageCubeMedian  = median(self.imageCube,axis=0)
        self.imageCubeMAD     = scale.mad(self.imageCube,axis=0)
        
        self.imageBadPixMasks = abs(self.imageCube - self.imageCubeMedian) > nSig*self.imageCubeMAD
        
        print("There are " + str(sum(self.imageBadPixMasks)) + " 'Hot' Pixels")
        
        # self.imageCube[self.imageBadPixMasks] = nan
    
    def fit_gaussian_centering(self, method='la', initc='fw', subArray=False, print_compare=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
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
        self.widths_GaussianFit       = zeros((self.imageCube.shape[0], 2))
        
        self.heights_GaussianFit      = zeros(self.imageCube.shape[0])
        # self.rotation_GaussianFit     = zeros(self.imageCube.shape[0])
        self.background_GaussianFit   = zeros(self.imageCube.shape[0])
        
        for kf in tqdm_notebook(range(self.nFrames), desc='GaussFit', leave = False, total=self.nFrames):
            subFrameNow = self.imageCube[kf][ylower:yupper, xlower:xupper]
            subFrameNow[isnan(subFrameNow)] = median(~isnan(subFrameNow))
            
            cmom    = np.array(moments(subFrameNow))  # H, Xc, Yc, Xs, Ys, O
            
            if method == 'ap':
                if initc == 'fw' and self.centering_FluxWeight.sum():
                    FWCNow    = self.centering_FluxWeight[kf]
                    FWCNow[self.y] = FWCNow[self.y] - ylower
                    FWCNow[self.x] = FWCNow[self.x] - xlower
                    gaussI    = hstack([cmom[0], FWCNow, cmom[3:]])
                if initc == 'cm':
                    gaussI  = hstack([cmom[0], cmom[1], cmom[2], cmom[3:]])
                
                gaussP  = fit_gauss(subFrameNow, xinds, yinds, gaussI) # H, Xc, Yc, Xs, Ys, Th, O
            
            if method == 'la':
                gaussP  = fitgaussian(subFrameNow)#, xinds, yinds, np.copy(cmom)) # H, Xc, Yc, Xs, Ys, Th, O
            
            self.centering_GaussianFit[kf][self.x]     = gaussP[1] + xlower
            self.centering_GaussianFit[kf][self.y]     = gaussP[2] + ylower
            
            self.widths_GaussianFit[kf][self.x]        = gaussP[3]
            self.widths_GaussianFit[kf][self.y]        = gaussP[4]
            
            self.heights_GaussianFit[kf]          = gaussP[0]
            self.background_GaussianFit[kf]       = gaussP[5]
            
            del gaussP, cmom
        
        self.centering_df = DataFrame()
        self.centering_df['Gaussian_Fit_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Centers'] = self.centering_GaussianFit.T[self.x]
        self.centering_df['Gaussian_Mom_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Mom_X_Centers'] = self.centering_GaussianFit.T[self.x]
        
        self.centering_df['Gaussian_Fit_Y_Widths']  = self.widths_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Widths']  = self.widths_GaussianFit.T[self.x]
        
        self.centering_df['Gaussian_Fit_Heights']   = self.heights_GaussianFit
        self.centering_df['Gaussian_Fit_Offset']    = self.background_GaussianFit
    
    def mp_lmfit_gaussian_centering(self, yguess=15, xguess=15, subArraySize=10, 
                                    init_params=None, useMoments=False, nCores=cpu_count(), 
                                    center_range=None, width_range=None, nSig=False, 
                                    method='leastsq', recheckMethod=None, verbose=False):
        
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        if isnan(self.imageCube).any():
            self.imageCube[where(isnan(self.imageCube))] = median(self.imageCube)
        
        imageSize  = self.imageCube.shape[1]
        
        nparams    = 6
        if init_params is None:
            useMoments = True
            init_params = moments(self.imageCube[0])
        
        if center_range is not None:
            ctr_min, ctr_max = center_range
        else:
            ctr_min, ctr_max = 0, imageSize
        
        if width_range is not None:
            wid_min, wid_max = width_range
        else:
            wid_min, wid_max = 0, imageSize
        
        ihg, iyc, ixc, iyw, ixw, ibg  = arange(nparams)
        
        lmfit_init_params = Parameters()
        lmfit_init_params.add_many(
            ('height'  , init_params[ihg], True  , 0.0    , inf    ),
            ('center_y', init_params[iyc], True  , ctr_min, ctr_max),
            ('center_x', init_params[ixc], True  , ctr_min, ctr_max),
            ('width_y' , init_params[iyw], True  , wid_min, wid_max),
            ('width_x' , init_params[ixw], True  , wid_min, wid_max),
            ('offset'  , init_params[ibg], True))
        
        gfit_model = Model(gaussian, independent_vars=['yy', 'xx'])
        
        yy0, xx0 = indices(self.imageCube[0].shape)
        
        npix   = subArraySize//2
        ylower = self.yguess - self.npix
        yupper = self.yguess + self.npix
        xlower = self.xguess - self.npix
        xupper = self.xguess + self.npix
        
        ylower, xlower, yupper, xupper = int32([ylower, xlower, yupper, xupper])
        
        yy = yy0[ylower:yupper, xlower:xupper]
        xx = xx0[ylower:yupper, xlower:xupper]
        
        pool = Pool(nCores)
        
        func = partial(lmfit_one_center, yy=yy, xx=xx, gfit_model=gfit_model, lmfit_init_params=lmfit_init_params, 
                                            yupper=yupper, ylower=ylower, xupper=xupper, xlower=xlower, method=method)
        
        gaussian_centers = pool.starmap(func, zip(self.imageCube))
        
        pool.close()
        pool.join()
        
        self.centering_GaussianFit    = zeros((self.imageCube.shape[0], 2))
        self.widths_GaussianFit       = zeros((self.imageCube.shape[0], 2))
        
        self.heights_GaussianFit      = zeros(self.imageCube.shape[0])
        self.background_GaussianFit   = zeros(self.imageCube.shape[0])
        
        print('Finished with Fitting Centers. Now assigning to instance values.')
        for kf, gaussP in enumerate(gaussian_centers):
            self.centering_GaussianFit[kf][self.y]     = gaussP['center_y']
            self.centering_GaussianFit[kf][self.x]     = gaussP['center_x']
            
            self.widths_GaussianFit[kf][self.y]        = gaussP['width_y']
            self.widths_GaussianFit[kf][self.x]        = gaussP['width_x']
            
            self.heights_GaussianFit[kf]               = gaussP['height']
            
            self.background_GaussianFit[kf]            = gaussP['offset']
        
        if recheckMethod is not None and isinstance(recheckMethod, str):
            if verbose: print('Rechecking corner cases:')
            medY, medX = median(self.centering_GaussianFit,axis=0)
            stdX, stdY = std(   self.centering_GaussianFit,axis=0)
            nSig       = 5.1
            outliers   = (((self.centering_GaussianFit.T[y] - medY)/stdY)**2 + ((self.centering_GaussianFit.T[x] - medX)/stdX)**2) > nSig
            
            for kf in where(outliers)[0]:
                if verbose: print('    Corner Case:\t{}\tPreviousSolution={}'.format(kf, self.centering_GaussianFit[kf]), end="\t")
                gaussP = lmfit_one_center(self.imageCube[kf], yy=yy, xx=xx, gfit_model=gfit_model, lmfit_init_params=lmfit_init_params, 
                                            yupper=yupper, ylower=ylower, xupper=xupper, xlower=xlower, method=recheckMethod)
                
                self.centering_GaussianFit[kf][self.y]     = gaussP['center_y']
                self.centering_GaussianFit[kf][self.x]     = gaussP['center_x']
            
                self.widths_GaussianFit[kf][self.y]        = gaussP['width_y']
                self.widths_GaussianFit[kf][self.x]        = gaussP['width_x']
            
                self.heights_GaussianFit[kf]               = gaussP['height']
            
                self.background_GaussianFit[kf]            = gaussP['offset']
                
                if verbose: print('NewSolution={}'.format(self.centering_GaussianFit[kf]))
        
        self.centering_df = DataFrame()
        self.centering_df['Gaussian_Fit_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Centers'] = self.centering_GaussianFit.T[self.x]
        
        self.centering_df['Gaussian_Fit_Y_Widths']  = self.widths_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Widths']  = self.widths_GaussianFit.T[self.x]
        
        self.centering_df['Gaussian_Fit_Heights']   = self.heights_GaussianFit
        self.centering_df['Gaussian_Fit_Offset']    = self.background_GaussianFit
        
    def mp_fit_gaussian_centering(self, nSig=False, method='la', initc='fw', subArray=False, print_compare=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        if isnan(self.imageCube).any():
            self.imageCube[where(isnan(self.imageCube))] = median(self.imageCube)
        
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
        self.widths_GaussianFit       = zeros((self.imageCube.shape[0], 2))
        
        self.heights_GaussianFit      = zeros(self.imageCube.shape[0])
        # self.rotation_GaussianFit     = zeros(self.imageCube.shape[0])
        self.background_GaussianFit   = zeros(self.imageCube.shape[0])
        
        # Gaussian fit centering
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(fit_one_center, nSig=nSig, method='gauss', ylower=ylower, yupper=yupper, xlower=xlower, xupper=xupper)
        
        gaussian_centers = pool.starmap(func, zip(self.imageCube)) # the order is very important
        
        pool.close()
        pool.join()
        
        print('Finished with Fitting Centers. Now assigning to instance values.')
        for kf, gaussP in enumerate(gaussian_centers):
            self.centering_GaussianFit[kf][self.x]     = gaussP[1] + xlower
            self.centering_GaussianFit[kf][self.y]     = gaussP[2] + ylower
            
            self.widths_GaussianFit[kf][self.x]        = gaussP[3]
            self.widths_GaussianFit[kf][self.y]        = gaussP[4]
            
            self.heights_GaussianFit[kf]               = gaussP[0]            
            self.background_GaussianFit[kf]            = gaussP[5]
        
        self.centering_df = DataFrame()
        self.centering_df['Gaussian_Fit_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Centers'] = self.centering_GaussianFit.T[self.x]
        self.centering_df['Gaussian_Mom_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Mom_X_Centers'] = self.centering_GaussianFit.T[self.x]
        
        self.centering_df['Gaussian_Fit_Y_Widths']  = self.widths_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Widths']  = self.widths_GaussianFit.T[self.x]
        
        self.centering_df['Gaussian_Fit_Heights']   = self.heights_GaussianFit
        self.centering_df['Gaussian_Fit_Offset']    = self.background_GaussianFit
        
        # self.centering_df['Gaussian_Fit_Rotation']    = self.rotation_GaussianFit
    
    def fit_flux_weighted_centering(self):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
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
        print(self.imageCube.shape)
        for kf in tqdm_notebook(range(self.nFrames), desc='FWC', leave = False, total=self.nFrames):
            subFrameNow = self.imageCube[kf][ylower:yupper, xlower:xupper]
            subFrameNow[isnan(subFrameNow)] = median(~isnan(subFrameNow))
            
            self.centering_FluxWeight[kf] = flux_weighted_centroid(self.imageCube[kf], 
                                                                       self.yguess, self.xguess, bSize = 7)
            self.centering_FluxWeight[kf] = self.centering_FluxWeight[kf][::-1]
        
        self.centering_FluxWeight[:,0] = clipOutlier(self.centering_FluxWeight.T[0])
        self.centering_FluxWeight[:,1] = clipOutlier(self.centering_FluxWeight.T[1])
        
        self.centering_df['FluxWeighted_Y_Centers'] = self.centering_FluxWeight.T[self.y]
        self.centering_df['FluxWeighted_X_Centers'] = self.centering_FluxWeight.T[self.x]
    
    def mp_fit_flux_weighted_centering(self, nSig=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        yinds0, xinds0 = indices(self.imageCube[0].shape)
        
        ylower = self.yguess - self.npix
        yupper = self.yguess + self.npix
        xlower = self.xguess - self.npix
        xupper = self.xguess + self.npix
        
        ylower, xlower, yupper, xupper = np.int32([ylower, xlower, yupper, xupper])
        
        yinds = yinds0[ylower:yupper, xlower:xupper]
        xinds = xinds0[ylower:yupper, xlower:xupper]
        
        nFWCParams                = 2 # Xc, Yc
        # self.centering_FluxWeight = np.zeros((self.nFrames, nFWCParams))
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(fit_one_center, nSig=nSig, method='fwc', ylower=ylower, yupper=yupper, xlower=xlower, xupper=xupper, bSize = 7)
        
        fwc_centers = pool.starmap(func, zip(self.imageCube)) # the order is very important
        
        pool.close()
        pool.join()
        
        self.centering_FluxWeight = np.array(fwc_centers)
        
        self.centering_df['FluxWeighted_Y_Centers'] = self.centering_FluxWeight.T[self.y]
        self.centering_df['FluxWeighted_X_Centers'] = self.centering_FluxWeight.T[self.x]
    
    def fit_least_asymmetry_centering(self):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
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
        
        for kf in tqdm_notebook(range(self.nFrames), desc='Asym', leave = False, total=self.nFrames):
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
            kf, yguess, xguess = np.int32([kf, self.yguess, self.xguess])
            
            try:
                center_asym = actr(self.imageCube[kf], [yguess, xguess], \
                                   asym_rad=8, asym_size=5, maxcounts=2, method='gaus', \
                                   half_pix=False, resize=False, weights=False)[0]
            except:
                fitFailed = True
            
            # Option 2: We assume that there is a deformation in the PSF and square the image array
            #  (preserving signs)
            if fitFailed:
                try: 
                    center_asym = actr(np.sign(self.imageCube[kf])*self.imageCube[kf]**2, \
                                       [yguess, xguess], \
                                       asym_rad=8, asym_size=5, maxcounts=2, method='gaus', \
                                       half_pix=False, resize=False, weights=False)[0]
                    fitFailed = False
                except:
                    pass
            
            # Option 3: We assume that there is a cosmic ray hit nearby the PSF and shrink the asym_rad by half
            if fitFailed:
                try: 
                    center_asym = actr(self.imageCube[kf], [yguess, xguess], \
                                       asym_rad=4, asym_size=5, maxcounts=2, method='gaus', \
                                       half_pix=False, resize=False, weights=False)[0]
                    fitFailed = False
                except:
                    pass
            
            # Option 4: We assume that there is a deformation in the PSF AND (or caused by) a cosmic ray hit 
            #   nearby the PSF; so we square the image array (preserving signs) AND shrink the asym_rad by half
            if fitFailed:
                try:
                    center_asym = actr(np.sign(self.imageCube[kf])*self.imageCube[kf]**2, \
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
                    center_asym = actr(np.sign(self.imageCube[kf])*self.imageCube[kf]**2, \
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
                self.centering_LeastAsym[kf]  = center_asym[::-1]
            except:
                print('Least Asymmetry FAILED: and returned `NaN`')
                fitFailed = True
            
            if fitFailed:
                print('Least Asymmetry FAILED: Setting self.centering_LeastAsym[%s] to Initial Guess: [%s,%s]' \
                      % (kf, self.yguess, self.xguess))
                self.centering_LeastAsym[kf]  = np.array([yguess, xguess])
        
        self.centering_df['LeastAsymmetry_Y_Centers'] = self.centering_LeastAsym.T[self.y]
        self.centering_df['LeastAsymmetry_X_Centers'] = self.centering_LeastAsym.T[self.x]
    
    def mp_fit_least_asymmetry_centering(self):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
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
        # for kf in tqdm_notebook(range(self.nFrames), desc='Asym', leave = False, total=self.nFrames):
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(actr, asym_rad=8, asym_size=5, maxcounts=2, method='gaus', half_pix=False, resize=False, weights=False)
        
        self.centering_LeastAsym = pool.starmap(func, zip(self.imageCube, [[yguess, xguess]]*self.nframes)) # the order is very important
        
        pool.close()
        pool.join()
        
        self.centering_LeastAsym = np.array(self.centering_LeastAsym[0])
        
        self.centering_df['LeastAsymmetry_Y_Centers'] = self.centering_LeastAsym.T[self.y]
        self.centering_df['LeastAsymmetry_X_Centers'] = self.centering_LeastAsym.T[self.x]
    
    def fit_all_centering(self):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        print('Fit for Gaussian Fitting Centers\n')
        self.mp_lmfit_gaussian_centering()
        
        print('Fit for Flux Weighted Centers\n')
        self.fit_flux_weighted_centering()
        
        print('Fit for Least Asymmetry Centers\n')
        self.fit_least_asymmetry_centering()
    
    def measure_effective_width(self, subFrame=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
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
    
    def measure_background_circle_masked(self, aperRad=10, centering='FluxWeight'):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        """
            Assigning all zeros in the mask to NaNs because the `mean` and `median` 
                functions are set to `nanmean` functions, which will skip all NaNs
        """
        
        self.background_CircleMask = np.zeros(self.nFrames)
        for kf in tqdm_notebook(range(self.nFrames), desc='CircleBG', leave = False, total=self.nFrames):
            aperture       = CircularAperture(self.centering_FluxWeight[kf], aperRad)
            
            aper_mask = aperture.to_mask(method='exact')[0]    # list of ApertureMask objects (one for each position)
            
            # backgroundMask = abs(aperture.get_fractions(np.ones(self.imageCube[0].shape))-1)
            backgroundMask = aper_mask.to_image(self.imageCube[0].shape).astype(bool)
            backgroundMask = ~backgroundMask#[backgroundMask == 0] = False
            
            self.background_CircleMask[kf] = self.metric(self.imageCube[kf][backgroundMask])
        
        self.background_df['CircleMask'] = self.background_CircleMask.copy()
    
    def mp_measure_background_circle_masked(self, aperRad=10, centering='Gauss'):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        """
            Assigning all zeros in the mask to NaNs because the `mean` and `median` 
                functions are set to `nanmean` functions, which will skip all NaNs
        """
        
        if centering=='Gauss':
            centers = self.centering_GaussianFit
        if centering=='FluxWeight':
            centers = self.centering_FluxWeight
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(measure_one_circle_bg, aperRad=aperRad, metric=self.metric, apMethod='exact')
        
        self.background_CircleMask = pool.starmap(func, zip(self.imageCube, centers))
        
        pool.close()
        pool.join()
        
        self.background_CircleMask       = np.array(self.background_CircleMask)
        self.background_df['CircleMask'] = self.background_CircleMask.copy()
    
    def measure_background_annular_mask(self, innerRad=8, outerRad=13):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        self.background_Annulus = np.zeros(self.nFrames)
        
        for kf in tqdm_notebook(range(self.nFrames), desc='AnnularBG', leave = False, total=self.nFrames):
            innerAperture = CircularAperture(self.centering_FluxWeight[kf], innerRad)
            outerAperture = CircularAperture(self.centering_FluxWeight[kf], outerRad)
            
            inner_aper_mask = innerAperture.to_mask(method='exact')[0]
            inner_aper_mask = inner_aper_mask.to_image(self.imageCube[0].shape).astype(bool)
            
            outer_aper_mask = outerAperture.to_mask(method='exact')[0]
            outer_aper_mask = outer_aper_mask.to_image(self.imageCube[0].shape).astype(bool)
            
            backgroundMask = (~inner_aper_mask)*outer_aper_mask
            
            self.background_Annulus[kf] = self.metric(self.imageCube[kf][backgroundMask])
        
        self.background_df['AnnularMask'] = self.background_Annulus.copy()
    
    def mp_measure_background_annular_mask(self, innerRad=8, outerRad=13, method='exact', centering='Gauss'):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        if centering=='Gauss':
            centers = self.centering_GaussianFit
        if centering=='FluxWeight':
            centers = self.centering_FluxWeight
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(measure_one_annular_bg, innerRad=innerRad, outerRad=outerRad, metric=self.metric, apMethod='exact')
        
        self.background_Annulus = pool.starmap(func, zip(self.imageCube, centers)) # the order is very important
        
        pool.close()
        pool.join()
        
        self.background_Annulus           = np.array(self.background_Annulus)
        self.background_df['AnnularMask'] = self.background_Annulus.copy()
    
    def measure_background_median_masked(self, aperRad=10, nSig=5):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        self.background_MedianMask  = np.zeros(self.nFrames)
        
        for kf in tqdm_notebook(range(self.nFrames), desc='MedianMaskedBG', leave = False, total=self.nFrames):
            aperture       = CircularAperture(self.centering_FluxWeight[kf], aperRad)
            aperture       = aperture.to_mask(method='exact')[0]
            aperture       = aperture.to_image(self.imageCube[0].shape).astype(bool)
            backgroundMask = ~aperture
            
            medFrame  = median(self.imageCube[kf][backgroundMask])
            madFrame  = std(self.imageCube[kf][backgroundMask]) # scale.mad(self.imageCube[kf][backgroundMask])
            
            medianMask= abs(self.imageCube[kf] - medFrame) < nSig*madFrame
            
            maskComb  = medianMask*backgroundMask
            # maskComb[maskComb == 0] = False
            
            self.background_MedianMask[kf] = median(self.imageCube[kf][maskComb])
        
        self.background_df['MedianMask'] = self.background_MedianMask
    
    def mp_measure_background_median_masked(self, aperRad=10, nSig=5, centering='Gauss'):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        if centering=='Gauss':
            centers = self.centering_GaussianFit
        if centering=='FluxWeight':
            centers = self.centering_FluxWeight
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(measure_one_median_bg, aperRad=aperRad, apMethod='exact', metric=self.metric, nSig=nSig)
        
        self.background_MedianMask = pool.starmap(func, zip(self.imageCube, centers))
        
        pool.close()
        pool.join()
        
        self.background_MedianMask       = np.array(self.background_MedianMask)
        self.background_df['MedianMask'] = self.background_MedianMask
    
    def measure_background_KDE_Mode(self, aperRad=10):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        self.background_KDEUniv = np.zeros(self.nFrames)
        
        for kf in tqdm_notebook(range(self.nFrames), desc='KDE_BG', leave = False, total=self.nFrames):
            aperture       = CircularAperture(self.centering_FluxWeight[kf], aperRad)
            aperture       = aperture.to_mask(method='exact')[0]
            aperture       = aperture.to_image(self.imageCube[0].shape).astype(bool)
            backgroundMask = ~aperture
            
            kdeFrame = kde.KDEUnivariate(self.imageCube[kf][backgroundMask].ravel())
            kdeFrame.fit()
            
            self.background_KDEUniv[kf] = kdeFrame.support[kdeFrame.density.argmax()]
        
        self.background_df['KDEUnivMask'] = self.background_KDEUniv
    
    def mp_measure_background_KDE_Mode(self, aperRad=10, centering='Gauss'):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        if centering=='Gauss':
            centers = self.centering_GaussianFit
        if centering=='FluxWeight':
            centers = self.centering_FluxWeight
        
        self.background_KDEUniv = np.zeros(self.nFrames)
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(measure_one_KDE_bg, aperRad=aperRad, apMethod='exact', metric=self.metric)
        
        self.background_KDEUniv = pool.starmap(func, zip(self.imageCube, centers)) # the order is very important
        
        pool.close()
        pool.join()
        
        self.background_KDEUniv              = np.array(self.background_KDEUniv)
        self.background_df['KDEUnivMask_mp'] = self.background_KDEUniv
    
    def measure_all_background(self, aperRad=10, nSig=5):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        pInner  = 0.2 # Percent Inner = -20%
        pOuter  = 0.3 # Percent Outer = +30%
        
        print('Measuring Background Using Circle Mask with Multiprocessing')
        self.mp_measure_background_circle_masked(aperRad=aperRad)
        
        print('Measuring Background Using Annular Mask with Multiprocessing')
        self.mp_measure_background_annular_mask(innerRad=(1-pInner)*aperRad, outerRad=(1+pOuter)*aperRad)
        
        print('Measuring Background Using Median Mask with Multiprocessing')
        self.mp_measure_background_median_masked(aperRad=aperRad, nSig=nSig)
        
        print('Measuring Background Using KDE Mode with Multiprocessing')
        self.mp_measure_background_KDE_Mode(aperRad=aperRad)
    
    def compute_flux_over_time(self, aperRad=None, centering='GaussianFit', background='AnnularMask', useTheForce=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
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
        
        centering_Use = np.transpose([self.centering_df[centering + '_Y_Centers'], 
                                      self.centering_df[centering + '_X_Centers']])
        
        background_Use= self.background_df[background]
        
        flux_key_now  = centering + '_' + background+'_' + 'rad' + '_' + str(aperRad)
        
        if flux_key_now not in self.flux_TSO_df.keys() or useTheForce:
            flux_TSO_now  = np.zeros(self.nFrames)
            noise_TSO_now = np.zeros(self.nFrames)
            
            for kf in tqdm_notebook(range(self.nFrames), desc='Flux', leave = False, total=self.nFrames):
                frameNow  = np.copy(self.imageCube[kf]) - background_Use[kf]
                frameNow[np.isnan(frameNow)] = median(frameNow)

                noiseNow  = np.copy(self.noiseCube[kf])**2.
                noiseNow[np.isnan(noiseNow)] = median(noiseNow)
                
                aperture  = CircularAperture([centering_Use[kf][self.x], centering_Use[kf][self.y]], r=aperRad)
                
                flux_TSO_now[kf]  = aperture_photometry(frameNow, aperture)['aperture_sum']
                noise_TSO_now[kf] = sqrt(aperture_photometry(noiseNow, aperture)['aperture_sum'])

            self.flux_TSO_df[flux_key_now]  = flux_TSO_now
            self.noise_TSO_df[flux_key_now] = noise_TSO_now
        else:
            print(flux_key_now + ' exists: if you want to overwrite, then you `useTheForce=True`')
    
    def compute_flux_over_time_over_aperRad(self, aperRads=[], centering_choices=[], background_choices=[], \
                                            useTheForce=False, verbose=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
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
                                                useTheForce= useTheForce)
                    
                    flux_key_now  = "{}_{}_rad_{}".format(ctrNow, bgNow, aperRad)
                    
                    if verbose:
                        flux_now = self.flux_TSO_df[flux_key_now]
                        print( std( flux_now/ median(flux_now)) * ppm)
        
        print('Operation took: ', time()-start)
    
    def mp_compute_flux_over_time(self, aperRad=3.0, centering='GaussianFit', background='AnnularMask', useTheForce=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        y,x = 0,1
        
        if background not in self.background_df.columns:
            raise Exception("`background` must be in", self.background_df.columns)
        
        if centering not in ['Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', 'LeastAsymmetry']:
            raise Exception("`centering` must be either 'Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', or 'LeastAsymmetry'")
        
        centering_Use = np.transpose([self.centering_df[centering + '_Y_Centers'], 
                                      self.centering_df[centering + '_X_Centers']])
        
        background_Use= self.background_df[background]
        
        flux_key_now  = centering + '_' + background+'_' + 'rad' + '_' + str(aperRad)
        
        if flux_key_now not in self.flux_TSO_df.keys() or useTheForce:            
            # for kf in tqdm_notebook(range(self.nFrames), desc='Flux', leave = False, total=self.nFrames):
                        
            pool = Pool(self.nCores)
            
            func = partial(compute_flux_one_frame, aperRad=aperRad)
            
            fluxNow = pool.starmap(func, zip(self.imageCube, centering_Use, background_Use))
            
            pool.close()
            pool.join()
            
            # fluxNow[~np.isfinite(fluxNow)]  = np.median(fluxNow[np.isfinite(fluxNow)])
            # fluxNow[fluxNow < 0]            = np.median(fluxNow[fluxNow > 0])
            
            self.flux_TSO_df[flux_key_now]  = fluxNow
            self.noise_TSO_df[flux_key_now] = np.sqrt(fluxNow)
            
        else:
            print(flux_key_now + ' exists: if you want to overwrite, then you `useTheForce=True`')
    
    def mp_compute_flux_over_time_varRad(self, staticRad, varRad=None, centering='GaussianFit', background='AnnularMask', useTheForce=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        y,x = 0,1
        
        if background not in self.background_df.columns:
            raise Exception("`background` must be in", self.background_df.columns)
        
        if centering not in ['Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', 'LeastAsymmetry']:
            raise Exception("`centering` must be either 'Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', or 'LeastAsymmetry'")
        
        centering_Use = np.transpose([self.centering_df[centering + '_Y_Centers'], 
                                      self.centering_df[centering + '_X_Centers']])
        
        background_Use= self.background_df[background]
        
        flux_key_now  = centering + '_' + background+'_' + 'rad' + '_' + str(staticRad) + '_' + str(varRad)
        
        if varRad is None and isinstance(staticRad, (float, int)):
            aperRad = [staticRad]*self.nFrames
        
        else:
            quad_rad_dist = self.quadrature_widths.copy() - np.median(self.quadrature_widths)
            quad_rad_dist = clipOutlier(quad_rad_dist, nSig=5)
            aperRads = staticRad + varRad*quad_rad_dist
        
        if flux_key_now not in self.flux_TSO_df.keys() or useTheForce:            
            # for kf in tqdm_notebook(range(self.nFrames), desc='Flux', leave = False, total=self.nFrames):
                        
            pool = Pool(self.nCores)
            
            # func = partial(compute_flux_one_frame)
            
            fluxNow = pool.starmap(compute_flux_one_frame, 
                                    zip(self.imageCube, centering_Use, background_Use, aperRads))
            
            pool.close()
            pool.join()
            
            # fluxNow[~np.isfinite(fluxNow)]  = np.median(fluxNow[np.isfinite(fluxNow)])
            # fluxNow[fluxNow < 0]            = np.median(fluxNow[fluxNow > 0])
            
            self.flux_TSO_df[flux_key_now]  = fluxNow
            self.noise_TSO_df[flux_key_now] = np.sqrt(fluxNow)
            
        else:
            print(flux_key_now + ' exists: if you want to overwrite, then you `useTheForce=True`')
    
    def extract_PLD_components(self, ycenter=None, xcenter=None, nCols=3, nRows=3, order=1):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        nPLDComp= nCols*nRows
        
        # nCols   = nCols // 2 # User input assumes matrix structure, which starts at y- and x-center
        # nRows   = nRows // 2 #   and moves +\- nRows/2 and nCols/2, respectively
        
        ycenter = int(ycenter) if ycenter is not None else self.imageCube.shape[1]//2-1 # nominally 15
        xcenter = int(xcenter) if xcenter is not None else self.imageCube.shape[2]//2-1 # nominally 15
        
        ylower  = ycenter - nRows // 2     # nominally 14
        yupper  = ycenter + nRows // 2 + 1 # nominally 17 (to include 16)
        xlower  = xcenter - nCols // 2     # nominally 14
        xupper  = xcenter + nCols // 2 + 1 # nominally 17 (to include 16)
        
        PLD_comps_local = self.imageCube[:,ylower:yupper,xlower:xupper].reshape((self.imageCube.shape[0],nPLDComp)).T
        
        PLD_norm       = sum(PLD_comps_local,axis=0)
        PLD_comps_local= PLD_comps_local / PLD_norm
        
        self.PLD_components = PLD_comps_local
        self.PLD_norm       = PLD_norm
        
        if order > 1:
            for o in range(2, order+1):
                self.PLD_components = vstack([self.PLD_components, PLD_comps_local**o])
    
    def DBScan_Flux_All(self, centering='gaussian', dbsClean=0):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        if centering == 'gaussian':
            ycenters    = self.centering_GaussianFit.T[y]
            xcenters    = self.centering_GaussianFit.T[x]
        else:
            print('Only Gaussian Centering is supported at the moment')
            print('Continuting with Gaussian Centering')
            ycenters    = self.centering_GaussianFit.T[y]
            xcenters    = self.centering_GaussianFit.T[x]
        
        try: 
            self.inliers_Phots = self.inliers_Phots
        except: 
            self.inliers_Phots = {}
        
        for flux_key_now in self.flux_TSO_df.keys():
            
            phots       = self.flux_TSO_df[flux_key_now]
            
            if flux_key_now not in self.inliers_Phots.keys() or useTheForce:
                self.inliers_Phots[flux_key_now]  = DBScan_Flux(phots, ycenters, xcenters, dbsClean=dbsClean)
            else:
                print(flux_key_now + ' exists: if you want to overwrite, then you `useTheForce=True`')
    
    def mp_DBScan_Flux_All(self, centering='gaussian', dbsClean=0):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        if centering == 'gaussian':
            ycenters    = self.centering_GaussianFit.T[y]
            xcenters    = self.centering_GaussianFit.T[x]
        else:
            print('Only Gaussian Centering is supported at the moment')
            print('Continuting with Gaussian Centering')
            ycenters    = self.centering_GaussianFit.T[y]
            xcenters    = self.centering_GaussianFit.T[x]
        
        try: 
            self.inliers_Phots = self.inliers_Phots
        except: 
            self.inliers_Phots = {}
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(DBScan_Flux, ycenters=ycenters, xcenters=xcenters, dbsClean=dbsClean)
        
        inliersMP = pool.starmap(func, zip(self.flux_TSO_df.values.T)) # the order is very important
        
        pool.close()
        pool.join()
        
        for k_mp, flux_key_now in enumerate(self.flux_TSO_df.keys()):
            self.inliers_Phots[flux_key_now] = inliersMP[k_mp]
    
    def mp_DBScan_PLD_All(self, dbsClean=0):
        raise Exception('This Function is Not Working; please use lame_`DBScan_PLD_all`')
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        
        try: 
            self.inliers_PLD = self.inliers_PLD
        except: 
            self.inliers_PLD = np.ones(self.PLD_components.shape, dtype=bool)
        
        pool = Pool(self.nCores) # This starts the multiprocessing call to arms
        
        func = partial(DBScan_PLD, dbsClean=dbsClean)
        
        inliersMP = pool.starmap(func, zip(self.PLD_components.T)) # the order is very important
        
        pool.close()
        pool.join()
        
        for k_mp, inlier in enumerate(inliersMP):
            self.inliers_PLD[k_mp] = inlier
    
    def mp_DBScan_PLD_All(self, dbsClean=0):
        """Class methods are similar to regular functions.
    
        Note:
            Do not include the `self` parameter in the ``Args`` section.
    
        Args:
            param1: The first parameter.
            param2: The second parameter.
    
        Returns:
            True if successful, False otherwise.
    
        """
        
        try: 
            self.inliers_PLD = self.inliers_PLD
        except: 
            self.inliers_PLD = np.ones(self.PLD_components.shape, dtype=bool)
        
        for kPLD, PLDnow in enumerate(self.PLD_components):
            self.inliers_PLD[kPLD] = DBScan_PLD(PLDnow, dbsClean=dbsClean)






# END OF FILE