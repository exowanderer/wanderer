import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from argparse import ArgumentParser
from astropy.io import fits
from astropy.modeling import models, fitting
from dataclasses import dataclass
from functools import partial
from glob import glob
from lmfit import Model, Parameters
from multiprocessing import cpu_count, Pool
from photutils.aperture import (
    CircularAperture,
    # CircularAnnulus,
    aperture_photometry,
    # findstars
)
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler
from skimage.filters import gaussian as gaussianFilter
from statsmodels.robust import scale
from statsmodels.nonparametric import kde

y, x = 0, 1

'''
Start: From least_asymmetry.asym by N.Lust 
(github.com/natelust/least_asymmetry) and modified (reversed XY -> YX)
'''


@dataclass
class WandererCLI:
    planet_name: str = 'planetname'
    channel: str = None
    aor_dir: str = None
    planets_dir: str = './'
    save_sub_dir: str = 'ExtracedData'
    data_sub_dir: str = '/data/raw/'
    data_tail_dir: str = ''
    fits_format: str = 'bcd'
    unc_format: str = 'bunc'
    save_file_type: str = '.joblib.save'
    method: str = 'median'
    telescope: str = 'Spitzer'
    output_units: str = 'electrons'
    data_dir: str = ''
    num_cores: int = 1
    verbose: bool = False


def command_line_inputs(check_defaults=True):
    ap = ArgumentParser()
    ap.add_argument('-pn', '--planet_name', type=str, default='planetname',
                    help='Directory Name for the Planet (i.e. hd209458b).')
    ap.add_argument('-c', '--channel', type=str, default=None,
                    help='Channel number string (i.e. ch1 or ch2).')
    ap.add_argument('-ad', '--aor_dir', type=str, default=None,
                    help='AOR director (i.e. r11235813).')
    ap.add_argument('-pd', '--planets_dir', type=str, default='./',
                    help='Location of planet directory name from $HOME.')
    ap.add_argument('-sd', '--save_sub_dir', type=str, default='ExtracedData',
                    help='Subdirectory inside Planet_Directory to '
                    'store extracted outputs.'
                    )
    ap.add_argument('-ds', '--data_sub_dir', type=str, default='/data/raw/',
                    help='Sub directory structure from '
                    '$HOME/Planet_Name/THIS/aor_dir/..'
                    )
    ap.add_argument('-dt', '--data_tail_dir', required=False,
                    type=str, default='', help='String inside AOR DIR.')
    ap.add_argument('-ff', '--fits_format', type=str,
                    default='bcd', help='Format of the fits files (i.e. bcd).')
    ap.add_argument('-uf', '--unc_format', type=str, default='bunc',
                    help='Format of the photometric noise files (i.e. bcd).')
    ap.add_argument('-sft', '--save_file_type', type=str,
                    default='.joblib.save',
                    help='file name extension for save files after processing'
                    )
    ap.add_argument('-m', '--method', type=str, default='median',
                    help='method for photmetric extraction (i.e. median).')
    ap.add_argument('-t', '--telescope', type=str, default='Spitzer',
                    help='Telescope: [Spitzer, Hubble, JWST].')
    ap.add_argument('-ou', '--output_units', type=str, default='electrons',
                    help='Units for the extracted photometry '
                    '[electrons, muJ_per_Pixel, etc].'
                    )
    ap.add_argument('-d', '--data_dir', type=str, default=None,
                    help='Set location of all `bcd` and `bunc` files: '
                    'bypass previous setup.'
                    )
    ap.add_argument('-nc', '--num_cores', type=int, default=cpu_count()-1)
    ap.add_argument('-v', '--verbose', type=bool,
                    default=False, help='Print out normally irrelevent things.')

    args = vars(ap.parse_args())

    return convert_args_to_dataclass(args, check_defaults=check_defaults)


def convert_args_to_dataclass(args, check_defaults=True):
    clargs = WandererCLI()
    clargs.args_obj = WandererCLI()
    clargs.planet_name = args['planet_name']
    clargs.channel = args['channel']
    clargs.aor_dir = args['aor_dir']
    clargs.planets_dir = args['planets_dir']
    clargs.save_sub_dir = args['save_sub_dir']
    clargs.data_sub_dir = args['data_sub_dir']
    clargs.data_tail_dir = args['data_tail_dir']
    clargs.fits_format = args['fits_format']
    clargs.unc_format = args['unc_format']
    clargs.method = args['method']
    clargs.telescope = args['telescope']
    clargs.output_units = args['output_units']
    clargs.data_dir = args['data_dir']
    clargs.num_cores = args['num_cores']
    clargs.verbose = args['verbose']

    if check_defaults:
        # Check important defaults directly
        if clargs.planet_name == 'planetname':
            print(
                UserWarning(
                    '\n[WARNING] User is using default planet_name="planetname"'
                    '\nPlease call command line with `-pn` or `--planet_name`\n'
                )
            )

        assert (clargs.channel is not None), \
            'Please call command line with `-c` or `--channel`'
        assert (clargs.aor_dir is not None), \
            'Please call command line with `-ad` or `--aor_dir`'

    return clargs


def pool_run_func(func, zipper, num_cores=cpu_count()-1):
    with Pool(num_cores) as pool:
        return pool.starmap(func, zipper)


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
    yinds, xinds = np.indices(data.shape)
    y = (yinds * data).sum()/total
    x = (xinds * data).sum()/total
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
    params, _ = sp.optimize.leastsq(errorfunction, params)

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

    return [
        sum(weights*ny*data)/sum(weights*data),
        sum(weights*nx*data)/sum(weights*data)
    ]


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
    fits_date = header['DATE-OBS']
    start_time_str = header['TIME-OBS']
    end_time_str = header['TIME-END']

    yyyy, mm, dd = fits_date.split('-')

    hh1, mn1, ss1 = np.array(start_time_str.split(':')).astype(float)
    hh2, mn2, ss2 = np.array(end_time_str.split(':')).astype(float)

    yyyy = float(yyyy)
    mm = float(mm)
    dd = float(dd)

    hh1 = float(hh1)
    mn1 = float(mn1)
    ss1 = float(ss1)

    hh2 = float(hh2)
    mn2 = float(mn2)
    ss2 = float(ss2)

    start_date = get_julian_date_from_gregorian_date(
        yyyy, mm, dd, hh1, mn1, ss1
    )
    end_date = get_julian_date_from_gregorian_date(yyyy, mm, dd, hh2, mn2, ss2)

    return start_date, end_date


def clipOutlier(vector, n_sig=8):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    medarray = np.nanmedian(vector)
    stdarray = np.nanstd(vector)
    outliers = abs(vector - medarray) > n_sig*stdarray

    vector[outliers] = np.nanmedian(vector[~outliers])
    return vector


def clipOutlier2D(arr2d, n_sig=10):
    arr2d = arr2d.copy()
    medArr2D = np.nanmedian(arr2d, axis=0)
    sclArr2D = np.sqrt(((scale.mad(arr2d)**2.).sum()))
    outliers = abs(arr2d - medArr2D) > n_sig*sclArr2D
    inliers = abs(arr2d - medArr2D) <= n_sig*sclArr2D
    arr2d[outliers] = np.nanmedian(arr2d[inliers], axis=0)
    return arr2d


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

    ypos, xpos, b_size = np.int32([ypos, xpos, b_size])
    # extract a box around the star:
    # im = a[ypos-b_size:ypos+b_size, xpos-b_size:xpos+b_size].copy()

    ystart = ypos - b_size
    ystop = ypos + b_size
    xstart = xpos - b_size
    xstop = xpos + b_size
    sub_img = image[ystart:ystop, xstart:xstop].transpose().copy()

    y, x = 0, 1

    ydim = sub_img.shape[y]
    xdim = sub_img.shape[x]

    # add up the flux along x and y
    xflux = np.zeros(xdim)
    xrng = np.arange(xdim)

    yflux = np.zeros(ydim)
    yrng = np.arange(ydim)

    for i in range(xdim):
        xflux[i] = sum(sub_img[i, :])

    for j in range(ydim):
        yflux[j] = sum(sub_img[:, j])

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

    chi_y = (center_y - yy) / width_y
    chi_x = (center_x - xx) / width_x

    return height * np.exp(-0.5*(chi_y**2 + chi_x**2)) + offset


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
    # TODO: Check if this is backwards
    xinds, yinds = np.indices(data.shape)
    x = (xinds*data).sum()/total
    y = (yinds*data).sum()/total

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
        image_cube, yguess=15, xguess=15, sub_array_size=10,
        init_params=None, n_sig=None, use_moments=False, method='leastsq'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    n_frames, image_size = image_cube.shape[:2]

    n_params = 6
    if init_params is None:
        use_moments = True
        init_params = moments(image_cube[0])

    ihg, iyc, ixc, iyw, ixw, ibg = np.arange(n_params)
    lmfit_init_params = Parameters()
    lmfit_init_params.add_many(
        ('height', init_params[ihg], True, 0.0, np.inf),
        ('center_y', init_params[iyc], True, 0.0, image_size),
        ('center_x', init_params[ixc], True, 0.0, image_size),
        ('width_y', init_params[iyw], True, 0.0, image_size),
        ('width_x', init_params[ixw], True, 0.0, image_size),
        ('offset', init_params[ibg], True)
    )

    gfit_model = Model(gaussian, independent_vars=['yy', 'xx'])

    yy0, xx0 = np.indices(image_cube[0].shape)

    npix = sub_array_size//2
    ylower = np.int32(yguess - npix)
    yupper = np.int32(yguess + npix)
    xlower = np.int32(xguess - npix)
    xupper = np.int32(xguess + npix)

    # ylower, xlower, yupper, xupper = np.int32([
    #   ylower, xlower, yupper, xupper
    # ])

    yy = yy0[ylower:yupper, xlower:xupper]
    xx = xx0[ylower:yupper, xlower:xupper]

    heights, ycenters, xcenters, ywidths, xwidths, offsets = np.zeros(
        (n_params, n_frames)
    )

    for k, image in enumerate(image_cube):
        sub_frame_now = image[ylower:yupper, xlower:xupper]
        sub_frame_now[np.isnan(sub_frame_now)] = np.nanmedian(sub_frame_now)

        # sub_frame_now = gaussianFilter(sub_frame_now, n_sig)
        # if not isinstance(n_sig, bool) else sub_frame_now

        if n_sig is not None:
            sub_frame_now = gaussianFilter(sub_frame_now, n_sig)

        init_params = moments(sub_frame_now) if use_moments else init_params

        gfit_res = gfit_model.fit(
            sub_frame_now,
            params=lmfit_init_params,
            xx=xx,
            yy=yy,
            method=method
        )

        heights[k] = gfit_res.best_values['height']
        ycenters[k] = gfit_res.best_values['center_y']
        xcenters[k] = gfit_res.best_values['center_x']
        ywidths[k] = gfit_res.best_values['width_y']
        xwidths[k] = gfit_res.best_values['width_x']
        offsets[k] = gfit_res.best_values['offset']

    return heights, ycenters, xcenters, ywidths, xwidths, offsets


def lmfit_one_center(
        image, yy, xx, gfit_model, lmfit_init_params, yupper, ylower, xupper,
        xlower, use_moments=True, n_sig=None, method='leastsq'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    sub_frame_now = image[ylower:yupper, xlower:xupper]
    sub_frame_now[np.isnan(sub_frame_now)] = np.nanmedian(sub_frame_now)

    if n_sig is not None:
        sub_frame_now = gaussianFilter(sub_frame_now, n_sig)

    init_params = moments(sub_frame_now) if use_moments else list(
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
        sub_frame_now, params=lmfit_init_params, xx=xx, yy=yy, method=method)
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
def print_model_params(model, init_params):
    print(model.amplitude_0 - init_params[0], end=" ")
    print(model.x_mean_0 - init_params[1], end=" ")
    print(model.y_mean_0 - init_params[2], end=" ")
    print(model.x_stddev_0 - init_params[3], end=" ")
    print(model.y_stddev_0 - init_params[4], end=" ")
    print(model.amplitude_1 - init_params[5])


def fit_gauss(sub_frame_now, xinds, yinds, init_params, print_compare=False):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    # init_params = (height, x, y, width_x, width_y, offset)
    fit_lvmq = fitting.LevMarLSQFitter()
    model0 = models.Gaussian2D(
        amplitude=init_params[0],
        x_mean=init_params[1],
        y_mean=init_params[2],
        x_stddev=init_params[3],
        y_stddev=init_params[4],
        theta=0.0
    )
    model0 = model0 + models.Const2D(amplitude=init_params[5])

    model1 = fit_lvmq(model0, xinds, yinds, sub_frame_now)
    model1 = fit_lvmq(model1, xinds, yinds, sub_frame_now)

    if print_compare:
        print_model_params(model1, init_params)

    return model1.parameters


def fit_one_center(
        image, ylower, yupper, xlower, xupper,
        n_sig=None, method='gaussian', b_size=7):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    sub_frame_now = image[ylower:yupper, xlower:xupper]
    sub_frame_now[np.isnan(sub_frame_now)] = np.nanmedian(sub_frame_now)

    # sub_frame_now = gaussianFilter(sub_frame_now, n_sig) if not isinstance(
    #     n_sig, bool) else sub_frame_now

    if n_sig is not None:
        sub_frame_now = gaussianFilter(sub_frame_now, n_sig)

    if method == 'moments':
        return np.array(moments(sub_frame_now))  # H, Xc, Yc, Xs, Ys, O
    if method == 'gaussian':
        # , xinds, yinds, np.copy(cmom)) # H, Xc, Yc, Xs, Ys, Th, O
        return fitgaussian(sub_frame_now)
    if method == 'flux_weighted':
        return flux_weighted_centroid(
            image,
            image.shape[y]//2,
            image.shape[x]//2,
            b_size=b_size
        )[::-1]


def create_aper_mask(centering, aper_rad, image_shape, method='exact'):
    aperture = CircularAperture(centering, aper_rad)
    aperture = aperture.to_mask(method=method)

    if isinstance(aperture, (list, tuple, np.ndarray)):
        aperture = aperture[0]

    return aperture.to_image(image_shape).astype(bool)


def compute_flux_one_frame(image, center, background, aper_rad=3.0):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    frame_now = image - background
    frame_now[np.isnan(frame_now)] = np.nanmedian(frame_now)

    aperture = CircularAperture([center[x], center[y]], r=abs(aper_rad))

    return aperture_photometry(frame_now, aperture)['aperture_sum'].data[0]


def measure_one_circle_bg(image, center, aper_rad, metric, aper_method='exact'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    aper_mask = create_aper_mask(
        centering=center,
        aper_rad=aper_rad,
        image_shape=image.shape,
        method=aper_method
    )
    """
    aperture = CircularAperture(center, aper_rad)
    # list of ApertureMask objects (one for each position)

    aper_mask = aperture.to_mask(method=aper_method)
    aper_mask = aper_mask[0]

    # backgroundMask = abs(
    #   aperture.get_fractions(np.ones(image.shape)) - 1
    # )
    backgroundMask = aper_mask.to_image(image.shape).astype(bool)
    """
    backgroundMask = ~aper_mask  # [backgroundMask == 0] = False

    return metric(image[backgroundMask])


def measure_one_annular_bg(
        image, center, inner_rad, outer_rad, metric, aper_method='exact'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    inner_aper_mask = create_aper_mask(
        centering=center,
        aper_rad=inner_rad,
        image_shape=image.shape,
        method=aper_method
    )

    outer_aper_mask = create_aper_mask(
        centering=center,
        aper_rad=outer_rad,
        image_shape=image.shape,
        method=aper_method
    )

    """
    inner_aper = CircularAperture(center, inner_rad)
    outer_aper = CircularAperture(center, outer_rad)

    inner_aper_mask = inner_aper.to_mask(method=aper_method)[0]
    inner_aper_mask = inner_aper_mask.to_image(image.shape).astype(bool)

    outer_aper_mask = outer_aper.to_mask(method=aper_method)[0]
    outer_aper_mask = outer_aper_mask.to_image(image.shape).astype(bool)
    """

    # Make an annulus by inverting a cirle and multiplying by a larger circle
    # This makes the inner circle 0 (annular hole), as well as
    #   outside the outer circle is zero
    backgroundMask = (~inner_aper_mask)*outer_aper_mask

    return metric(image[backgroundMask])


def measure_one_median_bg(
        image, center, aper_rad, metric, n_sig, aper_method='exact'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    aperture = create_aper_mask(
        centering=center,
        aper_rad=aper_rad,
        image_shape=image.shape,
        method=aper_method
    )
    """
    aperture = CircularAperture(center, aper_rad)
    aperture = aperture.to_mask(method=aper_method)[0]
    aperture = aperture.to_image(image.shape).astype(bool)
    """
    backgroundMask = ~aperture

    med_frame = np.nanmedian(image[backgroundMask])
    mad_frame = np.nanstd(image[backgroundMask])

    medianMask = abs(image - med_frame) < n_sig*mad_frame

    maskComb = medianMask*backgroundMask

    return np.nanmedian(image[maskComb])


def measure_one_kde_bg(image, center, aper_rad, metric, aper_method='exact'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    aperture = create_aper_mask(
        centering=center,
        aper_rad=aper_rad,
        image_shape=image.shape,
        method=aper_method
    )
    """
    aperture = CircularAperture(center, aper_rad)
    aperture = aperture.to_mask(method=aper_method)[0]
    aperture = aperture.to_image(image.shape).astype(bool)
    """
    backgroundMask = ~aperture

    kde_frame = kde.KDEUnivariate(image[backgroundMask])
    kde_frame.fit()

    return kde_frame.support[kde_frame.density.argmax()]


def compute_annular_mask(
        aper_rad, center, image, method='exact'):

    inner_rad, outer_rad = aper_rad

    inner_aper_mask = create_aper_mask(
        centering=center,
        aper_rad=inner_rad,
        image_shape=image.shape,
        method=method
    )

    outer_aper_mask = create_aper_mask(
        centering=center,
        aper_rad=outer_rad,
        image_shape=image.shape,
        method=method
    )
    """
    inner_aper = CircularAperture(center, inner_rad)
    outer_aper = CircularAperture(center, outer_rad)

    inner_aper_mask = inner_aper.to_mask(method=method)[0]
    inner_aper_mask = inner_aper_mask.to_image(image.shape).astype(bool)

    outer_aper_mask = outer_aper.to_mask(method=method)[0]
    outer_aper_mask = outer_aper_mask.to_image(image.shape).astype(bool)
    """

    # Make an annulus by inverting a cirle and multiplying by a larger circle
    # This makes the inner circle 0 (annular hole), as well as
    #   outside the outer circle is zero
    return (~inner_aper_mask)*outer_aper_mask


def measure_one_background(
        image, center, aper_rad, metric, n_sig=5,
        aper_method='exact', bg_method='circle'):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    if np.ndim(aper_rad):
        aperture = compute_annular_mask(
            aper_rad, center, image, method=aper_method
        )
    else:
        aperture = create_aper_mask(
            centering=center,
            aper_rad=aper_rad,
            image_shape=image.shape,
            method=aper_method
        )

        """
        aperture = CircularAperture(center, aper_rad)
        # list of ApertureMask objects (one for each position)
        aperture = aperture.to_mask(method=aper_method)[0]

        # inverse to keep 'outside' aperture
        aperture = ~aperture.to_image(image).astype(bool)
        """
        aperture = ~aperture

    if bg_method == 'median':
        med_frame = np.nanmedian(image[aperture])
        mad_frame = scale.mad(image[aperture])

        medianMask = abs(image - med_frame) < n_sig*mad_frame

        aperture = medianMask*aperture

    if bg_method == 'kde':
        kde_frame = kde.KDEUnivariate(image[aperture].ravel())
        kde_frame.fit()

        return kde_frame.support[kde_frame.density.argmax()]

    return metric(image[aperture])


def dbscan_flux(phots, ycenters, xcenters, dbs_clean=0, use_the_force=False):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    dbs_phots = DBSCAN()  # n_jobs=-1)
    stdScaler = StandardScaler()

    phots = np.copy(phots.ravel())
    phots[~np.isfinite(phots)] = np.nanmedian(phots[np.isfinite(phots)])

    features_now = np.transpose([
        stdScaler.fit_transform(ycenters[:, None]).ravel(),
        stdScaler.fit_transform(xcenters[:, None]).ravel(),
        stdScaler.fit_transform(phots[:, None]).ravel()
    ])

    # print(features_now.shape)
    dbs_phots_pred = dbs_phots.fit_predict(features_now)

    return dbs_phots_pred == dbs_clean


def factor(num2factor, arr=list()):
    i = 2
    maximum = num2factor / 2 + 1
    while i < maximum:
        if num2factor % i == 0:
            return factor(num2factor/i, arr + [i])
        i += 1
    return list(set(arr + [num2factor]))


def dbscan_segmented_flux(
        phots, ycenters, xcenters, dbs_clean=0, n_segments=None,
        max_segment=int(6e4), use_the_force=False):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    if phots.size <= max_segment:
        # Default to un-segmented
        return dbscan_flux(
            phots,
            ycenters,
            xcenters,
            dbs_clean=dbs_clean,
            use_the_force=use_the_force
        )

    dbs_phots = DBSCAN()  # n_jobs=-1)
    stdScaler = StandardScaler()

    if n_segments is None:
        n_segments = phots.size // max_segment

    segSize = phots.size // n_segments
    max_in_segs = n_segments * segSize

    segments = list(np.arange(max_in_segs).reshape(n_segments, -1))

    leftovers = np.arange(max_in_segs, phots.size)

    segments[-1] = np.hstack([segments[-1], leftovers])

    phots = np.copy(phots.ravel())
    phots[~np.isfinite(phots)] = np.nanmedian(phots[np.isfinite(phots)])

    # default to array of `dbs_clean` values
    dbs_phots_pred = np.zeros(phots.size) + dbs_clean

    for segment in segments:
        dbs_phots_pred[segment] = dbscan_flux(
            phots[segment],
            ycenters[segment],
            xcenters[segment],
            dbs_clean=dbs_clean,
            use_the_force=use_the_force
        )

    return dbs_phots_pred == dbs_clean


def dbscan_pld(pld_now, dbs_clean=0, use_the_force=False):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """

    dbs_pld = DBSCAN()  # n_jobs=-1)
    stdScaler = StandardScaler()

    dbs_pld_pred = dbs_pld.fit_predict(
        stdScaler.fit_transform(pld_now[:, None]))

    return dbs_pld_pred == dbs_clean


def dbscan_segmented_pld(
        pld_now, dbs_clean=0, n_segments=None, max_segment=int(6e4),
        use_the_force=False):
    """Class methods are similar to regular functions.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
    if pld_now.size <= max_segment:
        # Default to un-segmented
        return dbscan_pld(
            pld_now,
            dbs_clean=dbs_clean,
            use_the_force=use_the_force
        )

    # dbs_pld = DBSCAN()#n_jobs=-1)
    # stdScaler = StandardScaler()
    #
    if n_segments is None:
        n_segments = pld_now.size // max_segment

    segSize = pld_now.size // n_segments
    max_in_segs = n_segments * segSize

    segments = list(np.arange(max_in_segs).reshape(n_segments, -1))

    leftovers = np.arange(max_in_segs, pld_now.size)

    segments[-1] = np.hstack([segments[-1], leftovers])

    # default to array of `dbs_clean` values
    dbs_pld_pred = np.zeros(pld_now.size) + dbs_clean

    for segment in segments:
        dbs_pld_pred[segment] = dbscan_pld(
            pld_now[segment],
            dbs_clean=dbs_clean,
            use_the_force=use_the_force
        )

    return dbs_pld_pred == dbs_clean


def cross_correlation_HST_diff_NDR():
    # Cross correlated the differential non-destructive reads
    #   from HST Scanning mode with WFC3 and G141
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
