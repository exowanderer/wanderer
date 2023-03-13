import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy as sp
import sys

# import image_registration as ir

from astropy.io import fits
from astropy.modeling import models, fitting
from datetime import datetime
from functools import partial
from glob import glob
from lmfit import Model, Parameters
from multiprocessing import cpu_count, Pool
from photutils import (
    CircularAperture,
    CircularAnnulus,
    aperture_photometry,
    findstars
)
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler
from skimage.filters import gaussian as gaussianFilter
from socket import gethostname
from statsmodels.robust import scale
from statsmodels.nonparametric import kde
from time import time, localtime, sleep
from tqdm import tqdm, tqdm_notebook

from skimage.filters import gaussian as gaussianFilter

y, x = 0, 1

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

    y_dist_sq = ((center_y - yy)/width_y)**2
    x_dist_sq = ((center_x - xx)/width_x)**2

    return height * np.exp(-0.5*(y_dist_sq + x_dist_sq)) + offset


def moments(data, kernel_size=2, n_sig=4):
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
    Y, X = np.indices(data.shape)
    y = (Y * data).sum()/total
    x = (X * data).sum()/total
    height = gaussianFilter(data, kernel_size).max()

    med_data = np.nanmedian(data)
    firstq = np.nanmedian(data[data < med_data])
    thirdq = np.nanmedian(data[data > med_data])

    in_range_ = np.where(np.bitwise_and(data > firstq, data < thirdq))
    offset = np.nanmedian(data[in_range_])
    places = np.where(data > n_sig * np.nanstd(data[in_range_]) + offset)

    width_y = np.nanstd(places[0])
    width_x = np.nanstd(places[1])

    # These if statements take into account there might only be one significant
    # point above the background when that is the case it is assumend the width
    # of the gaussian must be smaller than one pixel

    if width_y == 0.0:
        width_y = 0.5

    if width_x == 0.0:
        width_x = 0.5

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

    yy, xx = np.indices(data.shape)

    gausspartial = partial(gaussian, yy=yy, xx=xx)

    def errorfunction(p): return np.ravel((gausspartial(*p) - data)*weights)
    params, success = sp.optimize.leastsq(errorfunction, params)

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
    verbose = False
    if verbose:
        print(date)

    date = list(date)

    if len(date) < 3:
        raise ValueError("You must enter a date of the form (2009, 02, 25)!")

    while len(date) < 6:
        date.append(0)

    # Old code kept for backwards compatibility
    if len(date) == 3:
        date.extend([0] * 3)
    elif len(date) == 4:
        date.extend([0] * 2)
    elif len(date) == 5:
        date.append(0)

    yyyy = int(date[0])
    mm = int(date[1])
    dd = float(date[2])
    hh = float(date[3])
    mns = float(date[4])
    sec = float(date[5])

    UT = hh + mns/60 + sec/3600
    sig = 1 if 100*yyyy + mm > 190002.5 else -1

    JD = 367 * yyyy
    JD = JD - int(7 * (yyyy + int((mm + 9) / 12)) / 4)
    JD = JD + int(275 * mm / 9)
    JD = JD + dd
    JD = JD + 1721013.5
    JD = JD + UT / 24
    JD = JD - 0.5 * sig + 0.5

    # Now calculate the fractional year. Do we have a leap year?
    daylist = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    daylist2 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    days = daylist2  # TODO: Check if conditional can be simplified
    if (yyyy % 4 != 0) or (yyyy % 400 == 0) or yyyy % 100 != 0:
        days = daylist2
    else:
        days = daylist

    """ # To be deleted
    if (yyyy % 4 != 0):
        days = daylist2
    elif (yyyy % 400 == 0):
        days = daylist2
    elif (yyyy % 100 == 0):
        days = daylist
    else:
        days = daylist2
    """

    daysum = 0
    for y in range(mm-1):
        daysum = daysum + days[y]

    daysum = daysum + dd - 1 + UT / 24

    # Fraction of year with respect to leap year or default
    fracyear = yyyy + daysum/366 if days[1] == 29 else yyyy + daysum/365

    # if days[1] == 29:
    #     fracyear = yyyy + daysum/366
    # else:
    #     fracyear = yyyy + daysum/365

    if verbose:
        sec2day = 1 / 86400
        total_seconds = hh * 3600 + mns * 60 + sec
        fracday = total_seconds * sec2day

        months = [
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"
        ]

        print("yyyy | mm | dd | hh | mns | sec")
        print(f"{yyyy} | {mm} | {dd} | {hh} | {mns} | {sec}")
        print(f"UT = {UT}")
        print(f"Fractional day: {fracday}")
        print(
            f"{months[mm-1]} {dd}, {yyyy}, {hh}:{mns}:{sec} UT = {JD} JD",
            f" = {fracyear}"
        )
        print()

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
    fitsDate = header['DATE-OBS']
    startTimeStr = header['TIME-OBS']
    endTimeStr = header['TIME-END']

    yyyy, mm, dd = fitsDate.split('-')

    hh1, mn1, ss1 = np.array(startTimeStr.split(':')).astype(float)
    hh2, mn2, ss2 = np.array(endTimeStr.split(':')).astype(float)

    yyyy = float(yyyy)
    mm = float(mm)
    dd = float(dd)

    hh1 = float(hh1)
    mn1 = float(mn1)
    ss1 = float(ss1)

    hh2 = float(hh2)
    mn2 = float(mn2)
    ss2 = float(ss2)

    startDate = get_julian_date_from_gregorian_date(
        yyyy, mm, dd, hh1, mn1, ss1)
    endDate = get_julian_date_from_gregorian_date(yyyy, mm, dd, hh2, mn2, ss2)

    return startDate, endDate


def clipOutlier(oneDarray, n_sig=8):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    medarray = np.nanmedian(oneDarray)
    stdarray = np.nanstd(oneDarray)
    outliers = abs(oneDarray - medarray) > n_sig*stdarray

    oneDarray[outliers] = np.nanmedian(oneDarray[~outliers])
    return oneDarray


def flux_weighted_centroid(image, ypos, xpos, b_size=7):
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

    ypos, xpos, bsize = np.int32([ypos, xpos, b_size])
    # extract a box around the star:
    #im = a[ypos-b_size:ypos+b_size, xpos-b_size:xpos+b_size].copy()

    ystart = ypos - b_size
    ystop = ypos + b_size
    xstart = xpos - b_size
    xstop = xpos + b_size
    subImage = image[ystart:ystop, xstart:xstop].transpose().copy()

    y, x = 0, 1

    ydim = subImage.shape[y]
    xdim = subImage.shape[x]

    # add up the flux along x and y
    xflux = np.zeros(xdim)
    xrng = np.arange(xdim)

    yflux = np.zeros(ydim)
    yrng = np.arange(ydim)

    for i in range(xdim):
        xflux[i] = sum(subImage[i, :])

    for j in range(ydim):
        yflux[j] = sum(subImage[:, j])

    # get the flux weighted average position:
    ypeak = sum(yflux * yrng) / sum(yflux) + ypos - float(b_size)
    xpeak = sum(xflux * xrng) / sum(xflux) + xpos - float(b_size)

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

    chiY = (center_y - yy) / width_y
    chiX = (center_x - xx) / width_x

    return height * np.exp(-0.5*(chiY**2 + chiX**2)) + offset


def moments(data, n_sig=4):
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
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    height = data.max()
    firstq = np.nanmedian(data[data < np.nanmedian(data)])
    thirdq = np.nanmedian(data[data > np.nanmedian(data)])

    in_range_ = np.where(np.bitwise_and(data > firstq, data < thirdq))
    offset = np.nanmedian(data[in_range_])
    places = np.where(data > n_sig * np.nanstd(data[in_range_]) + offset)

    """
    in_range_ = np.where(np.bitwise_and(data > firstq, data < thirdq))
    offset = np.nanmedian(data[in_range_])
    places = np.where(data > n_sig * np.nanstd(data[in_range_]) + offset)
    """

    width_y = np.nanstd(places[0])
    width_x = np.nanstd(places[1])
    # These if statements take into account there might only be one significant
    # point above the background when that is the case it is assumend the width
    # of the gaussian must be smaller than one pixel
    if width_y == 0.0:
        width_y = 0.5
    if width_x == 0.0:
        width_x = 0.5

    height -= offset
    return height, y, x, width_y, width_x, offset


def lame_lmfit_gaussian_centering(
        imageCube, yguess=15, xguess=15, subArraySize=10,
        init_params=None, n_sig=False, useMoments=False, method='leastsq'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    n_frames, imageSize = imageCube.shape[:2]

    n_params = 6
    if init_params is None:
        useMoments = True
        init_params = moments(imageCube[0])

    ihg, iyc, ixc, iyw, ixw, ibg = np.arange(n_params)
    lmfit_init_params = Parameters()
    lmfit_init_params.add_many(
        ('height', init_params[ihg], True, 0.0, np.inf),
        ('center_y', init_params[iyc], True, 0.0, imageSize),
        ('center_x', init_params[ixc], True, 0.0, imageSize),
        ('width_y', init_params[iyw], True, 0.0, imageSize),
        ('width_x', init_params[ixw], True, 0.0, imageSize),
        ('offset', init_params[ibg], True)
    )

    gfit_model = Model(gaussian, independent_vars=['yy', 'xx'])

    yy0, xx0 = np.indices(imageCube[0].shape)

    npix = subArraySize//2
    ylower = yguess - npix
    yupper = yguess + npix
    xlower = xguess - npix
    xupper = xguess + npix

    ylower, xlower, yupper, xupper = np.int32([ylower, xlower, yupper, xupper])

    yy = yy0[ylower:yupper, xlower:xupper]
    xx = xx0[ylower:yupper, xlower:xupper]

    heights, ycenters, xcenters, ywidths, xwidths, offsets = np.zeros(
        (n_params, n_frames))

    for k, image in enumerate(imageCube):
        subFrameNow = image[ylower:yupper, xlower:xupper]
        subFrameNow[np.isnan(subFrameNow)] = np.nanmedian(subFrameNow)

        # subFrameNow = gaussianFilter(subFrameNow, n_sig) if not isinstance(
        #     n_sig, bool) else subFrameNow

        if not isinstance(n_sig, bool):
            subFrameNow = gaussianFilter(subFrameNow, n_sig)

        init_params = moments(subFrameNow) if useMoments else init_params

        gfit_res = gfit_model.fit(
            subFrameNow, params=lmfit_init_params, xx=xx, yy=yy, method=method)

        heights[k] = gfit_res.best_values['height']
        ycenters[k] = gfit_res.best_values['center_y']
        xcenters[k] = gfit_res.best_values['center_x']
        ywidths[k] = gfit_res.best_values['width_y']
        xwidths[k] = gfit_res.best_values['width_x']
        offsets[k] = gfit_res.best_values['offset']

    return heights, ycenters, xcenters, ywidths, xwidths, offsets


def lmfit_one_center(image, yy, xx, gfit_model, lmfit_init_params, yupper, ylower, xupper, xlower, useMoments=True, n_sig=False, method='leastsq'):
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
    subFrameNow[np.isnan(subFrameNow)] = np.nanmedian(subFrameNow)

    if not isinstance(n_sig, bool):
        subFrameNow = gaussianFilter(subFrameNow, n_sig)

    init_params = moments(subFrameNow) if useMoments else list(
        lmfit_init_params.valuesdict().values())

    n_params = 6
    ihg, iyc, ixc, iyw, ixw, ibg = np.arange(n_params)

    lmfit_init_params.height = init_params[ihg]
    lmfit_init_params.center_y = init_params[iyc]
    lmfit_init_params.center_x = init_params[ixc]
    lmfit_init_params.width_y = init_params[iyw]
    lmfit_init_params.width_x = init_params[ixw]
    lmfit_init_params.offset = init_params[ibg]

    # print(lmfit_init_params)

    gfit_res = gfit_model.fit(
        subFrameNow, params=lmfit_init_params, xx=xx, yy=yy, method=method)
    # print(list(gfit_res.best_values.values()))

    fit_values = gfit_res.best_values

    return (
        fit_values['center_y'],
        fit_values['center_x'],
        fit_values['width_y'],
        fit_values['width_x'],
        fit_values['height'],
        fit_values['offset']
    )


# TODO Rename this here and in `fit_gauss`
def print_model_params(model, initParams):
    print(model.amplitude_0 - initParams[0], end=" ")
    print(model.x_mean_0 - initParams[1], end=" ")
    print(model.y_mean_0 - initParams[2], end=" ")
    print(model.x_stddev_0 - initParams[3], end=" ")
    print(model.y_stddev_0 - initParams[4], end=" ")
    print(model.amplitude_1 - initParams[5])


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
    model0 = models.Gaussian2D(
        amplitude=initParams[0],
        x_mean=initParams[1],
        y_mean=initParams[2],
        x_stddev=initParams[3],
        y_stddev=initParams[4],
        theta=0.0
    )
    model0 = model0 + models.Const2D(amplitude=initParams[5])

    model1 = fit_lvmq(model0, xinds, yinds, subFrameNow)
    model1 = fit_lvmq(model1, xinds, yinds, subFrameNow)

    if print_compare:
        print_model_params(model1, initParams)

    return model1.parameters


def fit_one_center(
        image, ylower, yupper, xlower, xupper,
        n_sig=False, method='gaussian', b_size=7):
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
    subFrameNow[np.isnan(subFrameNow)] = np.nanmedian(subFrameNow)

    # subFrameNow = gaussianFilter(subFrameNow, n_sig) if not isinstance(
    #     n_sig, bool) else subFrameNow

    if not isinstance(n_sig, bool):
        subFrameNow = gaussianFilter(subFrameNow, n_sig)

    if method == 'moments':
        return np.array(moments(subFrameNow))  # H, Xc, Yc, Xs, Ys, O
    if method == 'gaussian':
        # , xinds, yinds, np.copy(cmom)) # H, Xc, Yc, Xs, Ys, Th, O
        return fitgaussian(subFrameNow)
    if method == 'flux_weighted':
        return flux_weighted_centroid(
            image,
            image.shape[y]//2,
            image.shape[x]//2,
            b_size=b_size
        )[::-1]


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

    frameNow = image - background
    frameNow[np.isnan(frameNow)] = np.nanmedian(frameNow)

    aperture = CircularAperture([center[x], center[y]], r=abs(aperRad))

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

    aperture = CircularAperture(center, aperRad)
    # list of ApertureMask objects (one for each position)
    aper_mask = aperture.to_mask(method=apMethod)[0]

    # backgroundMask = abs(aperture.get_fractions(np.ones(self.imageCube[0].shape))-1)
    backgroundMask = aper_mask.to_image(image.shape).astype(bool)
    backgroundMask = ~backgroundMask  # [backgroundMask == 0] = False

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

    innerAperture = CircularAperture(center, innerRad)
    outerAperture = CircularAperture(center, outerRad)

    inner_aper_mask = innerAperture.to_mask(method=apMethod)[0]
    inner_aper_mask = inner_aper_mask.to_image(image.shape).astype(bool)

    outer_aper_mask = outerAperture.to_mask(method=apMethod)[0]
    outer_aper_mask = outer_aper_mask.to_image(image.shape).astype(bool)

    backgroundMask = (~inner_aper_mask)*outer_aper_mask

    return metric(image[backgroundMask])


def measure_one_median_bg(image, center, aperRad, metric, n_sig, apMethod='exact'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    aperture = CircularAperture(center, aperRad)
    aperture = aperture.to_mask(method=apMethod)[0]
    aperture = aperture.to_image(image.shape).astype(bool)
    backgroundMask = ~aperture

    medFrame = np.nanmedian(image[backgroundMask])
    madFrame = np.nanstd(image[backgroundMask])

    medianMask = abs(image - medFrame) < n_sig*madFrame

    maskComb = medianMask*backgroundMask

    return np.nanmedian(image[maskComb])


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

    aperture = CircularAperture(center, aperRad)
    aperture = aperture.to_mask(method=apMethod)[0]
    aperture = aperture.to_image(image.shape).astype(bool)
    backgroundMask = ~aperture

    kdeFrame = kde.KDEUnivariate(image[backgroundMask])
    kdeFrame.fit()

    return kdeFrame.support[kdeFrame.density.argmax()]


def compute_annular_mask(
        aperRad, center, image, method='exact'):

    innerRad, outerRad = aperRad

    innerAperture = CircularAperture(center, innerRad)
    outerAperture = CircularAperture(center, outerRad)

    inner_aper_mask = innerAperture.to_mask(method=method)[0]
    inner_aper_mask = inner_aper_mask.to_image(image.shape).astype(bool)

    outer_aper_mask = outerAperture.to_mask(method=method)[0]
    outer_aper_mask = outer_aper_mask.to_image(image.shape).astype(bool)

    return (~inner_aper_mask)*outer_aper_mask


def measure_one_background(
        image, center, aperRad, metric, n_sig=5,
        apMethod='exact', bgMethod='circle'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    if np.ndim(aperRad):
        aperture = compute_annular_mask(
            aperRad, center, image, method=apMethod)
    else:
        aperture = CircularAperture(center, aperRad)
        # list of ApertureMask objects (one for each position)
        aperture = aperture.to_mask(method=apMethod)[0]

        # inverse to keep 'outside' aperture
        aperture = ~aperture.to_image(image).astype(bool)

    if bgMethod == 'median':
        medFrame = np.nanmedian(image[aperture])
        madFrame = scale.mad(image[aperture])

        medianMask = abs(image - medFrame) < n_sig*madFrame

        aperture = medianMask*aperture

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

    dbsPhots = DBSCAN()  # n_jobs=-1)
    stdScaler = StandardScaler()

    phots = np.copy(phots.ravel())
    phots[~np.isfinite(phots)] = np.nanmedian(phots[np.isfinite(phots)])

    featuresNow = np.transpose([stdScaler.fit_transform(ycenters[:, None]).ravel(),
                                stdScaler.fit_transform(
                                    xcenters[:, None]).ravel(),
                                stdScaler.fit_transform(phots[:, None]).ravel()])

    # print(featuresNow.shape)
    dbsPhotsPred = dbsPhots.fit_predict(featuresNow)

    return dbsPhotsPred == dbsClean


def factor(numberToFactor, arr=list()):
    i = 2
    maximum = numberToFactor / 2 + 1
    while i < maximum:
        if numberToFactor % i == 0:
            return factor(numberToFactor/i, arr + [i])
        i += 1
    return list(set(arr + [numberToFactor]))


def DBScan_Segmented_Flux(phots, ycenters, xcenters, dbsClean=0, nSegments=None, maxSegment=int(6e4), useTheForce=False):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    if phots.size <= maxSegment:
        # Default to un-segmented
        return DBScan_Flux(phots, ycenters, xcenters, dbsClean=dbsClean, useTheForce=useTheForce)

    dbsPhots = DBSCAN()  # n_jobs=-1)
    stdScaler = StandardScaler()

    if nSegments is None:
        nSegments = phots.size // maxSegment

    segSize = phots.size // nSegments
    max_in_segs = nSegments * segSize

    segments = list(np.arange(max_in_segs).reshape(nSegments, -1))

    leftovers = np.arange(max_in_segs, phots.size)

    segments[-1] = np.hstack([segments[-1], leftovers])

    phots = np.copy(phots.ravel())
    phots[~np.isfinite(phots)] = np.nanmedian(phots[np.isfinite(phots)])

    # default to array of `dbsClean` values
    dbsPhotsPred = np.zeros(phots.size) + dbsClean

    for segment in segments:
        dbsPhotsPred[segment] = DBScan_Flux(
            phots[segment], ycenters[segment], xcenters[segment], dbsClean=dbsClean, useTheForce=useTheForce)

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

    dbsPLD = DBSCAN()  # n_jobs=-1)
    stdScaler = StandardScaler()

    dbsPLDPred = dbsPLD.fit_predict(stdScaler.fit_transform(PLDNow[:, None]))

    return dbsPLDPred == dbsClean


def DBScan_Segmented_PLD(PLDNow, dbsClean=0, nSegments=None, maxSegment=int(6e4), useTheForce=False):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    if PLDNow.size <= maxSegment:
        # Default to un-segmented
        return DBScan_PLD(PLDNow, dbsClean=dbsClean, useTheForce=useTheForce)

    # dbsPLD      = DBSCAN()#n_jobs=-1)
    # stdScaler   = StandardScaler()
    #
    if nSegments is None:
        nSegments = PLDNow.size // maxSegment

    segSize = PLDNow.size // nSegments
    max_in_segs = nSegments * segSize

    segments = list(np.arange(max_in_segs).reshape(nSegments, -1))

    leftovers = np.arange(max_in_segs, PLDNow.size)

    segments[-1] = np.hstack([segments[-1], leftovers])

    # default to array of `dbsClean` values
    dbsPLDPred = np.zeros(PLDNow.size) + dbsClean

    for segment in segments:
        dbsPLDPred[segment] = DBScan_PLD(
            PLDNow[segment], dbsClean=dbsClean, useTheForce=useTheForce)

    return dbsPLDPred == dbsClean


def cross_correlation_HST_diff_NDR():
    # Cross correlated the differential non-destructive reads from HST Scanning mode with WFC3 and G141
    fitsfiles = glob("*ima*fits")

    ylow = 50
    yhigh = 90
    nExts = 36
    extGap = 5

    shifts_ndr = np.zeros((len(fitsfiles), (nExts-1)//extGap, 2))
    fitsfile0 = fits.open(fitsfiles[0])
    for kf, fitsfilenow in enumerate(fitsfiles):
        fitsnow = fits.open(fitsfilenow)
        for kndr in range(extGap+1, nExts+1)[::extGap][::-1]:
            fits0_dndrnow = fitsfile0[kndr-extGap].data[ylow:yhigh] - \
                fitsfile0[kndr].data[ylow:yhigh]
            fitsN_dndrnow = fitsnow[kndr-extGap].data[ylow:yhigh] - \
                fitsnow[kndr].data[ylow:yhigh]

            idx_ = (kndr-1)//extGap - 1
            shifts_ndr[kf][idx_] = ir.chi2_shift(fits0_dndrnow, fitsN_dndrnow)
            shifts_ndr[kf][idx_] = shifts_ndr[kf][idx_][:2]

            # ax.clear()
            # plt.imshow(fitsN_dndrnow)
            # ax.set_aspect('auto')
            # plt.pause(1e-3)

    plt.plot(shifts_ndr[:, :-1, 0], 'o')  # x-shifts
    plt.plot(shifts_ndr[:, :-1, 1], 'o')  # y-shifts
