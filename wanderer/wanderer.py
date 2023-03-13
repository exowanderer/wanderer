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
    # CircularAnnulus,
    aperture_photometry,
    # findstars
)
# from sklearn.cluster import DBSCAN

# from sklearn.preprocessing import StandardScaler
# from skimage.filters import gaussian as gaussianFilter
# from socket import gethostname
from statsmodels.robust import scale
from statsmodels.nonparametric import kde
from time import time, localtime  # , sleep
from tqdm import tqdm, tqdm_notebook

# from skimage.filters import gaussian as gaussianFilter
# import everything that `wanderer` needs to operate
# from .auxiliary import *
from .auxiliary import (
    actr,
    clipOutlier,
    compute_flux_one_frame,
    DBScan_Flux,
    DBScan_PLD,
    DBScan_Segmented_Flux,
    fit_gauss,
    fitgaussian,
    fit_one_center,
    flux_weighted_centroid,
    gaussian,
    get_julian_date_from_header,
    lmfit_one_center,
    measure_one_annular_bg,
    measure_one_circle_bg,
    measure_one_median_bg,
    measure_one_KDE_bg,
    moments,
    pool_run_func
)


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
    day2sec = 86400.

    def __init__(
            self, fitsFileDir='./', filetype='slp.fits', telescope=None,
            yguess=None, xguess=None, pix_rad=5, method='mean', nCores=None,
            jupyter=False):
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

        self.y, self.x = 0, 1
        self.tqdm = tqdm_notebook if jupyter else tqdm
        self.method = method
        self.filetype = filetype

        if method == 'mean':
            self.metric = np.nanmean
        elif method == 'median':
            self.metric = np.nanmedian
        else:
            raise ValueError(
                "`method` must be from the list ['mean', 'median']"
            )

        self.fitsFileDir = fitsFileDir
        self.fitsFilenames = glob(f'{self.fitsFileDir}/*{self.filetype}')
        # self.fitsFilenames = glob(self.fitsFileDir + '/*' + self.filetype)
        self.nSlopeFiles = len(self.fitsFilenames)

        if telescope is None:
            raise ValueError(
                "Please specify `telescope` as either 'JWST' "
                "or 'Spitzer' or 'HST'"
            )

        self.telescope = telescope

        if self.telescope == 'Spitzer':
            fitsdir_split = self.fitsFileDir.replace('raw', 'cal').split('/')
            for _ in range(4):
                fitsdir_split.pop()

            # self.calDir = ''
            # for thing in fitsdir_split:
            #     self.calDir = self.calDir + thing + '/'
            #
            # self.permBadPixels = fits.open(
            #     self.calDir + 'nov14_ch1_bcd_pmask_subarray.fits'
            # )

        if self.nSlopeFiles == 0:
            print(
                'Pipeline found no Files in ' +
                self.fitsFileDir + ' of type /*' +
                filetype
            )
            exit(-1)

        self.centering_df = pd.DataFrame()
        self.background_df = pd.DataFrame()
        self.flux_TSO_df = pd.DataFrame()
        self.noise_TSO_df = pd.DataFrame()

        self.yguess = yguess
        if self.yguess is None:
            self.yguess = self.imageCube.shape[self.y]//2

        self.xguess = xguess
        if self.xguess is None:
            self.xguess = self.imageCube.shape[self.x]//2

        self.pix_rad = pix_rad
        self.nCores = cpu_count()//2 if nCores is None else int(nCores)

        tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst = localtime()
        print(
            'Completed Class Definition at ' +
            str(tm_year) + '-' + str(tm_mon) + '-' + str(tm_mday) + ' ' +
            str(tm_hour) + 'h' + str(tm_min) + 'm' + str(tm_sec) + 's'
        )

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
        testfits = fits.open(self.fitsFilenames[0])[0]

        self.nFrames = self.nSlopeFiles
        self.imageCube = np.zeros(
            (self.nFrames, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.noiseCube = np.zeros(
            (self.nFrames, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.timeCube = np.zeros(self.nFrames)

        self.imageBadPixMasks = None

        del testfits

        progress_fits_filenames = self.tqdm(
            enumerate(self.fitsFilenames),
            desc='JWST Load File',
            leave=False,
            total=self.nSlopeFiles
        )

        for kf, fname in progress_fits_filenames:
            fitsNow = fits.open(fname)

            self.imageCube[kf] = fitsNow[0].data[0]
            self.noiseCube[kf] = fitsNow[0].data[1]

            # re-write these 4 lines into `get_julian_date_from_header`
            startJD, endJD = get_julian_date_from_header(fitsNow[0].header)
            timeSpan = (endJD - startJD) * self.day2sec / self.nFrames

            self.timeCube[kf] = startJD + timeSpan * (kf + 0.5)
            self.timeCube[kf] = self.timeCube[kf] / self.day2sec - 2450000.

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
        # BMJD_2_BJD = -0.5
        from scipy.constants import arcsec  # pi / arcsec = 648000

        # sec2day = 1/(24*3600)
        sec2day = 1 / self.day2sec
        nFramesPerFile = 64

        testfits = fits.open(self.fitsFilenames[0])[0]
        testheader = testfits.header

        bcd_shape = testfits.data[0].shape

        self.nFrames = self.nSlopeFiles * nFramesPerFile
        self.imageCube = np.zeros((self.nFrames, bcd_shape[0], bcd_shape[1]))
        self.noiseCube = np.zeros((self.nFrames, bcd_shape[0], bcd_shape[1]))
        self.timeCube = np.zeros(self.nFrames)

        self.imageBadPixMasks = None

        del testfits

        # Converts DN/s to microJy per pixel
        #   1) expTime * gain / fluxConv converts MJ/sr to electrons
        #   2) as2sr * MJ2mJ * testheader['PXSCAL1'] * testheader['PXSCAL2']
        #       converts MJ/sr to muJ/pixel

        if outputUnits == 'electrons':
            fluxConv = testheader['FLUXCONV']
            expTime = testheader['EXPTIME']
            gain = testheader['GAIN']
            fluxConversion = expTime*gain / fluxConv

        elif outputUnits == 'muJ_per_Pixel':
            as2sr = arcsec**2.0  # steradians per square arcsecond
            MJ2mJ = 1e12  # mircro-Janskeys per Mega-Jansky
            # converts MJ
            fluxConversion = abs(
                as2sr * MJ2mJ * testheader['PXSCAL1'] * testheader['PXSCAL2']
            )
        else:
            raise ValueError(
                "`outputUnits` must be either 'electrons' or 'muJ_per_Pixel'"
            )

        print('Loading Spitzer Data')

        progress_fits_filenames = self.tqdm(
            enumerate(self.fitsFilenames),
            desc='Spitzer Load File',
            leave=False,
            total=self.nSlopeFiles
        )
        for kfile, fname in progress_fits_filenames:
            bcdNow = fits.open(fname)
            buncNow = fits.open(fname.replace('bcd.fits', 'bunc.fits'))

            for iframe in range(nFramesPerFile):
                idx_ = kfile * nFramesPerFile + iframe
                header_ = bcdNow[0].header
                bmjd_obs_ = header_['BMJD_OBS']
                et_obs_ = header_['ET_OBS']
                utcs_obs_ = header_['UTCS_OBS']
                frametime_ = float(header_['FRAMTIME'])

                # Convert from exposure time to UTC
                et_to_utc = (et_obs_ - utcs_obs_) * sec2day

                # Convert from exposure time to frame time in UTC
                frametime_adjust = et_to_utc + iframe * frametime_ * sec2day

                # Same time as BMJD with UTC correction
                self.timeCube[idx_] = bmjd_obs_ + frametime_adjust

                self.imageCube[idx_] = bcdNow[0].data[iframe] * fluxConversion
                self.noiseCube[idx_] = buncNow[0].data[iframe] * fluxConversion

            def delete_fits_data(fits_data):
                del fits_data[0].data
                fits_data.close()
                del fits_data

            delete_fits_data(bcdNow)
            delete_fits_data(buncNow)
            # del bcdNow[0].data
            # bcdNow.close()
            # del bcdNow

            # del buncNow[0].data
            # buncNow.close()
            # del buncNow

        if remove_nans:
            # Set NaNs to Median
            # where_is_nan_ = np.where(np.isnan(self.imageCube))
            is_nan_ = np.isnan(self.imageCube)
            self.imageCube[is_nan_] = np.nanmedian(self.imageCube)

    def hst_load_fits_file(self, fitsNow):
        """Not Yet Implemented"""
        raise NotImplementedError('HST Load Fits File does not exist, yet')

    def load_data_from_fits_files(self, remove_nans=True):
        """ Case function for loading fits from various telescope configurations

        Args:
            remove_nans (bool, optional): Whether to change the value of NaNs.
                Defaults to True.
        """

        if self.telescope == 'JWST':
            self.jwst_load_fits_file()

        if self.telescope == 'Spitzer':
            self.spitzer_load_fits_file(remove_nans=remove_nans)

        if self.telescope == 'HST':
            self.hst_load_fits_file()

    def load_data_from_save_files(
            self, savefiledir=None, saveFileNameHeader=None,
            saveFileType='.pickle.save'):
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
            raise ValueError(
                '`saveFileNameHeader` should be the beginning '
                'of each save file name'
            )

        if savefiledir is None:
            savefiledir = './'

        print('Loading from Master Files')
        files_path_start = os.path.join(savefiledir, saveFileNameHeader)
        file_path_template = files_path_start + '{0}' + saveFileType

        self.centering_df = pd.read_pickle(
            file_path_template.format('_centering_dataframe')
        )
        self.background_df = pd.read_pickle(
            file_path_template.format('_background_dataframe')
        )
        self.flux_TSO_df = pd.read_pickle(
            file_path_template.format('_flux_TSO_dataframe')
        )

        noise_file_path = file_path_template.format('_noise_TSO_dataframe')

        if os.path.exists(noise_file_path):
            self.noise_TSO_df = pd.read_pickle(noise_file_path)
        else:
            self.noise_TSO_df = None

        self.imageCube = joblib.load(
            file_path_template.format('_image_cube_array')
        )
        self.noiseCube = joblib.load(
            file_path_template.format('_noise_cube_array')
        )
        self.timeCube = joblib.load(
            file_path_template.format('_time_cube_array')
        )

        self.imageBadPixMasks = joblib.load(
            file_path_template.format('_image_bad_pix_cube_array')
        )

        self.save_dict = joblib.load(
            file_path_template.format('_save_dict')
        )

        print('nFrames', 'nFrame' in self.save_dict.keys())
        print('Assigning Parts of `self.save_dict` to individual data structures')
        for key in self.save_dict.keys():
            exec("self." + key + " = self.save_dict['" + key + "']")

    def save_collection(self, file_path_template):
        dump_dict = {
            '_centering_dataframe': self.centering_df,
            '_background_dataframe': self.background_df,
            '_flux_TSO_dataframe': self.flux_TSO_df,
            '_image_cube_array': self.imageCube,
            '_noise_cube_array': self.noiseCube,
            '_time_cube_array': self.timeCube,
            '_image_bad_pix_cube_array': self.imageBadPixMasks,
            '_save_dict': self.save_dict,
        }

        for filename_, df_ in dump_dict.items():
            joblib.dump(df_, file_path_template.format(filename_))

    def save_data_to_save_files(
            self, savefiledir=None, saveFileNameHeader=None,
            saveFileType='.joblib.save', SaveMaster=True, SaveTime=True):
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
            raise ValueError(
                '`saveFileNameHeader` should be the beginning '
                'of each save file name'
            )

        if savefiledir is None:
            savefiledir = './'

        if savefiledir[-1] != '/':
            savefiledir = savefiledir + '/'

        if not os.path.exists(savefiledir):
            os.mkdir(savefiledir)

        if not os.path.exists(savefiledir + 'TimeStamped'):
            os.mkdir(savefiledir + 'TimeStamped')

        date = localtime()

        year = date.tm_year
        month = date.tm_mon
        day = date.tm_mday

        hour = date.tm_hour
        minute = date.tm_min
        sec = date.tm_sec

        date_string = (
            f'_{str(year)}-{str(month)}-{str(day)}_'
            f'{str(hour)}h{str(minute)}m{str(sec)}s'
        )

        saveFileTypeBak = date_string + saveFileType
        self.initiate_save_dict()

        if SaveMaster:
            print('\nSaving to Master File -- Overwriting Previous Master')

            files_path_start = os.path.join(savefiledir, saveFileNameHeader)
            file_path_template = files_path_start + "{0}" + saveFileType

            self.save_collection(file_path_template)

        if SaveTime:
            print('Saving to New TimeStamped File -- These Tend to Pile Up!')

            file_path_base = os.path.join(
                savefiledir, 'TimeStamped', saveFileNameHeader
            )
            file_path_template = file_path_base + '{0}' + saveFileTypeBak
            self.save_collection(file_path_template)

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

        _stored_variables = [
            'background_Annulus',
            'background_CircleMask',
            'background_GaussianFit',
            'background_KDEUniv',
            'background_MedianMask',
            'centering_FluxWeight',
            'centering_GaussianFit',
            'centering_LeastAsym',
            'effective_widths',
            'fitsFileDir',
            'fitsFilenames',
            'heights_GaussianFit',
            'inliers_Phots',
            'inliers_PLD',
            'inliers_Master',
            'method',
            'pix_rad',
            'nFrames',
            'PLD_components',
            'PLD_norm',
            'quadrature_widths',
            'yguess',
            'xguess',
            'widths_GaussianFit',
            'AOR',
            'planetName',
            'channel'
        ]

        _max_str_len = 0
        for thing in _stored_variables:
            if len(thing) > _max_str_len:
                _max_str_len = len(thing)

        print('Storing in `self.save_dict`: ')
        # DataFrame does not work because each all `columns` must be 1D and have the same length
        self.save_dict = {}
        for key, val in self.__dict__.items():
            self.save_dict[key] = val
            # + ' into save_dict')#+ ' '*(len(key) - _max_str_len)
            print('\tself.' + key)

        print('\n')

        notYet = True
        for thing in _stored_variables:
            if thing not in self.save_dict.keys():
                if notYet:
                    print('The following do not yet exist:')
                    notYet = False

                print("\tself."+thing)  # +" does not yet exist")

        # for thing in _stored_variables:
        #     exec("try: self.save_dict['"+thing+"'] = self."+thing+"\nexcept: print('self."+thing+" does not yet exist')")
        #
        # try:self.save_dict['background_Annulus'] = self.background_Annulus
        # except: pass
        #
        # try:self.save_dict['background_CircleMask'] = self.background_CircleMask
        # except: pass
        #
        # try:self.save_dict['background_GaussianFit'] = self.background_GaussianFit
        # except: pass
        #
        # try:self.save_dict['background_KDEUniv'] = self.background_KDEUniv
        # except: pass
        #
        # try:self.save_dict['background_MedianMask'] = self.background_MedianMask
        # except: pass
        #
        # try:self.save_dict['centering_FluxWeight'] = self.centering_FluxWeight
        # except: pass
        #
        # try:self.save_dict['centering_GaussianFit'] = self.centering_GaussianFit
        # except: pass
        #
        # try:self.save_dict['centering_LeastAsym'] = self.centering_LeastAsym
        # except: pass
        #
        # try:self.save_dict['effective_widths'] = self.effective_widths
        # except: pass
        #
        # try:self.save_dict['fitsFileDir'] = self.fitsFileDir
        # except: pass
        #
        # try:self.save_dict['fitsFilenames'] = self.fitsFilenames
        # except: pass
        #
        # try:self.save_dict['heights_GaussianFit'] = self.heights_GaussianFit
        # except: pass
        #
        # try:self.save_dict['inliers_Phots'] = self.inliers_Phots
        # except: pass
        #
        # try:self.save_dict['inliersPLD'] = self.inliers_PLD
        # except: pass
        #
        # try:self.save_dict['method'] = self.method
        # except: pass
        #
        # try:self.save_dict['pix_rad'] = self.pix_rad
        # except: pass
        #
        # try:self.save_dict['nFrames'] = self.nFrames
        # except: pass
        #
        # try:self.save_dict['PLD_components'] = self.PLD_components
        # except: pass
        #
        # try:self.save_dict['PLD_norm'] = self.PLD_norm
        # except: pass
        #
        # try:self.save_dict['quadrature_widths'] = self.quadrature_widths
        # except: pass
        #
        # try:self.save_dict['yguess'] = self.yguess
        # except: pass
        #
        # try:self.save_dict['xguess'] = self.xguess
        # except: pass
        #
        # try:self.save_dict['widths_GaussianFit'] = self.widths_GaussianFit
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

        temp = wanderer(
            fitsFileDir=self.fitsFileDir,
            filetype=self.filetype,
            telescope=self.telescope,
            yguess=self.yguess,
            xguess=self.xguess,
            pix_rad=self.pix_rad,
            method=self.method,
            nCores=self.nCores
        )

        temp.centering_df = self.centering_df
        temp.background_df = self.background_df
        temp.flux_TSO_df = self.flux_TSO_df
        temp.noise_TSO_df = self.noise_TSO_df

        temp.imageCube = self.imageCube
        temp.noiseCube = self.noiseCube
        temp.timeCube = self.timeCube

        temp.imageBadPixMasks = self.imageBadPixMasks

        print('Assigning Parts of `temp.save_dict` to from `self.save_dict`')
        temp.save_dict = self.save_dict
        for thing in self.save_dict.keys():
            exec("temp." + thing + " = self.save_dict['" + thing + "']")

        return temp

    def find_bad_pixels(self, n_sig=5):
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
        self.imageCubeMedian = np.nanmedian(self.imageCube, axis=0)
        self.imageCubeMAD = scale.mad(self.imageCube, axis=0)

        self.imageBadPixMasks = abs(
            self.imageCube - self.imageCubeMedian) > n_sig*self.imageCubeMAD

        # print(
        #   "There are " + str(sum(self.imageBadPixMasks)) + " 'Hot' Pixels"
        # )
        print(f"There are {str(sum(self.imageBadPixMasks))} 'Hot' Pixels")

        # self.imageCube[self.imageBadPixMasks] = nan

    def fit_gaussian_centering(
            self, method='la', initc='fw', subArray=False, print_compare=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        y, x = 0, 1

        yinds0, xinds0 = np.indices(self.imageCube[0].shape)

        ylower = np.int32(self.yguess - self.pix_rad)
        yupper = np.int32(self.yguess + self.pix_rad)
        xlower = np.int32(self.xguess - self.pix_rad)
        xupper = np.int32(self.xguess + self.pix_rad)

        # ylower, xlower, yupper, xupper = np.int32(
        #     [ylower, xlower, yupper, xupper]
        # )

        yinds = yinds0[ylower:yupper, xlower:xupper]
        xinds = xinds0[ylower:yupper, xlower:xupper]

        self.centering_GaussianFit = np.zeros((self.imageCube.shape[0], 2))
        self.widths_GaussianFit = np.zeros((self.imageCube.shape[0], 2))

        self.heights_GaussianFit = np.zeros(self.imageCube.shape[0])
        # self.rotation_GaussianFit = np.zeros(self.imageCube.shape[0])
        self.background_GaussianFit = np.zeros(self.imageCube.shape[0])

        progress_kframe = self.tqdm(
            range(self.nFrames),
            desc='GaussFit',
            leave=False,
            total=self.nFrames
        )
        for kf in progress_kframe:
            subFrameNow = self.imageCube[kf][ylower:yupper, xlower:xupper]
            # subFrameNow[np.isnan(subFrameNow)] = np.nanmedian(~np.isnan(subFrameNow))
            subFrameNow[np.isnan(subFrameNow)] = np.nanmedian(subFrameNow)

            cmom = np.array(moments(subFrameNow))  # H, Xc, Yc, Xs, Ys, O

            if method == 'aperture_photometry':
                if initc == 'flux_weighted' and self.centering_FluxWeight.sum():
                    fwc_ = self.centering_FluxWeight[kf]
                    fwc_[self.y] = fwc_[self.y] - ylower
                    fwc_[self.x] = fwc_[self.x] - xlower
                    gaussI = np.hstack([cmom[0], fwc_, cmom[3:]])
                if initc == 'cm':
                    gaussI = np.hstack([cmom[0], cmom[1], cmom[2], cmom[3:]])

                # H, Xc, Yc, Xs, Ys, Th, O
                gaussP = fit_gauss(subFrameNow, xinds, yinds, gaussI)

            if method == 'la':
                # , xinds, yinds, np.copy(cmom)) # H, Xc, Yc, Xs, Ys, Th, O
                gaussP = fitgaussian(subFrameNow)

            self.centering_GaussianFit[kf][self.x] = gaussP[1] + xlower
            self.centering_GaussianFit[kf][self.y] = gaussP[2] + ylower

            self.widths_GaussianFit[kf][self.x] = gaussP[3]
            self.widths_GaussianFit[kf][self.y] = gaussP[4]

            self.heights_GaussianFit[kf] = gaussP[0]
            self.background_GaussianFit[kf] = gaussP[5]

            del gaussP, cmom

        self.centering_df = pd.DataFrame()
        self.centering_df['Gaussian_Fit_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Centers'] = self.centering_GaussianFit.T[self.x]
        self.centering_df['Gaussian_Mom_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Mom_X_Centers'] = self.centering_GaussianFit.T[self.x]

        self.centering_df['Gaussian_Fit_Y_Widths'] = self.widths_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Widths'] = self.widths_GaussianFit.T[self.x]

        self.centering_df['Gaussian_Fit_Heights'] = self.heights_GaussianFit
        self.centering_df['Gaussian_Fit_Offset'] = self.background_GaussianFit

    def mp_lmfit_gaussian_centering(
            self, yguess=15, xguess=15, subArraySize=10, init_params=None, useMoments=False, nCores=cpu_count(), center_range=None, width_range=None, n_sig=6.1, method='leastsq', recheckMethod=None,
            median_crop=False, verbose=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        y, x = 0, 1
        if np.isnan(self.imageCube).any():
            is_nan_ = np.isnan(self.imageCube)
            self.imageCube[is_nan_] = np.nanmedian(self.imageCube)

        imageSize = self.imageCube.shape[1]

        nparams = 6
        if init_params is None:
            useMoments = True
            init_params = moments(self.imageCube[0])

        ctr_min, ctr_max = 0, imageSize  # Set default as full fimage
        if center_range is not None:
            ctr_min, ctr_max = center_range

        wid_min, wid_max = 0, imageSize  # Set default as full fimage
        if width_range is not None:
            wid_min, wid_max = width_range

        ihg, iyc, ixc, iyw, ixw, ibg = np.arange(nparams)

        lmfit_init_params = Parameters()
        lmfit_init_params.add_many(
            ('height', init_params[ihg], True, 0.0, np.inf),
            ('center_y', init_params[iyc], True, ctr_min, ctr_max),
            ('center_x', init_params[ixc], True, ctr_min, ctr_max),
            ('width_y', init_params[iyw], True, wid_min, wid_max),
            ('width_x', init_params[ixw], True, wid_min, wid_max),
            ('offset', init_params[ibg], True)
        )

        gfit_model = Model(gaussian, independent_vars=['yy', 'xx'])

        yy0, xx0 = np.indices(self.imageCube[0].shape)

        pix_rad = subArraySize//2
        ylower = self.yguess - self.pix_rad
        yupper = self.yguess + self.pix_rad
        xlower = self.xguess - self.pix_rad
        xupper = self.xguess + self.pix_rad

        ylower, xlower, yupper, xupper = np.int32(
            [ylower, xlower, yupper, xupper])

        yy = yy0[ylower:yupper, xlower:xupper]
        xx = xx0[ylower:yupper, xlower:xupper]

        # pool = Pool(nCores)

        func = partial(
            lmfit_one_center,
            yy=yy,
            xx=xx,
            gfit_model=gfit_model,
            lmfit_init_params=lmfit_init_params,
            yupper=yupper,
            ylower=ylower,
            xupper=xupper,
            xlower=xlower,
            method=method
        )

        gaussian_centers = pool_run_func(func, zip(self.imageCube))

        # pool.close()
        # pool.join()

        print(
            'Finished with Fitting Centers. Now assigning to instance values.'
        )

        self.centering_GaussianFit = np.zeros((self.imageCube.shape[0], 2))
        self.widths_GaussianFit = np.zeros((self.imageCube.shape[0], 2))

        self.heights_GaussianFit = np.zeros(self.imageCube.shape[0])
        self.background_GaussianFit = np.zeros(self.imageCube.shape[0])

        gaussian_centers = np.array(gaussian_centers)

        # ['center_y']
        self.centering_GaussianFit.T[self.y] = gaussian_centers.T[0]
        # ['center_x']
        self.centering_GaussianFit.T[self.x] = gaussian_centers.T[1]

        # ['width_y']
        self.widths_GaussianFit.T[self.y] = gaussian_centers.T[2]
        # ['width_x']
        self.widths_GaussianFit.T[self.x] = gaussian_centers.T[3]

        self.heights_GaussianFit[:] = gaussian_centers.T[4]  # ['height']
        self.background_GaussianFit[:] = gaussian_centers.T[5]  # ['offset']

        if verbose:
            print('Rechecking corner cases:')

        medY, medX = np.nanmedian(self.centering_GaussianFit, axis=0)
        stdX, stdY = np.nanstd(self.centering_GaussianFit, axis=0)

        outliers = (((self.centering_GaussianFit.T[y] - medY)/stdY)**2 + (
            (self.centering_GaussianFit.T[x] - medX)/stdX)**2) > n_sig

        if recheckMethod is not None and isinstance(recheckMethod, str):
            for kf in np.where(outliers)[0]:
                if verbose:
                    # print('    Corner Case:\t{}\tPreviousSolution={}'.format(
                    #     kf, self.centering_GaussianFit[kf]), end="\t")
                    print(
                        f'    Corner Case:\t{kf}'
                        f'\tPreviousSolution={self.centering_GaussianFit[kf]}',
                        end="\t"
                    )

                gaussP = lmfit_one_center(
                    self.imageCube[kf],
                    yy=yy,
                    xx=xx,
                    gfit_model=gfit_model,
                    lmfit_init_params=lmfit_init_params,
                    yupper=yupper,
                    ylower=ylower,
                    xupper=xupper,
                    xlower=xlower,
                    method=recheckMethod
                )

                # ['center_y']
                self.centering_GaussianFit[kf][self.y] = gaussP[0]
                # ['center_x']
                self.centering_GaussianFit[kf][self.x] = gaussP[1]

                self.widths_GaussianFit[kf][self.y] = gaussP[2]  # ['width_y']
                self.widths_GaussianFit[kf][self.x] = gaussP[3]  # ['width_x']

                self.heights_GaussianFit[kf] = gaussP[4]  # ['height']

                self.background_GaussianFit[kf] = gaussP[5]  # ['offset']

                if verbose:
                    print(f'NewSolution={self.centering_GaussianFit[kf]}')

        elif median_crop:
            print('Setting Gaussian Centerintg Outliers to the Median')
            y_gaussball = ((self.centering_GaussianFit.T[y] - medY)/stdY)**2
            x_gaussball = ((self.centering_GaussianFit.T[x] - medX)/stdX)**2

            inliers = (y_gaussball+x_gaussball) <= n_sig
            medY, medX = np.nanmedian(
                self.centering_GaussianFit[inliers], axis=0)

            self.centering_GaussianFit.T[y][outliers] = medY
            self.centering_GaussianFit.T[x][outliers] = medX

        try:
            self.centering_df = self.centering_df  # check if it exists
        except:
            self.centering_df = pd.DataFrame()       # create it if it does not exist

        self.centering_df['Gaussian_Fit_Y_Centers'] = self.centering_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Centers'] = self.centering_GaussianFit.T[self.x]

        self.centering_df['Gaussian_Fit_Y_Widths'] = self.widths_GaussianFit.T[self.y]
        self.centering_df['Gaussian_Fit_X_Widths'] = self.widths_GaussianFit.T[self.x]

        self.centering_df['Gaussian_Fit_Heights'] = self.heights_GaussianFit
        self.centering_df['Gaussian_Fit_Offset'] = self.background_GaussianFit

    def assign_gaussian_centering(self, gaussP, xlower, kf, ylower):

        y, x = self.y, self.x
        self.centering_GaussianFit[kf][x] = gaussP[1] + xlower
        self.centering_GaussianFit[kf][y] = gaussP[2] + ylower

        self.widths_GaussianFit[kf][x] = gaussP[3]
        self.widths_GaussianFit[kf][y] = gaussP[4]

        self.heights_GaussianFit[kf] = gaussP[0]
        self.background_GaussianFit[kf] = gaussP[5]

    def mp_fit_gaussian_centering(self, n_sig=False, method='la', initc='fw', subArray=False, print_compare=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        if np.isnan(self.imageCube).any():
            is_nan_ = np.isnan(self.imageCube)
            self.imageCube[is_nan_] = np.nanmedian(self.imageCube)

        y, x = self.y, self.x

        yinds0, xinds0 = np.indices(self.imageCube[0].shape)

        ylower = self.yguess - self.pix_rad
        yupper = self.yguess + self.pix_rad
        xlower = self.xguess - self.pix_rad
        xupper = self.xguess + self.pix_rad

        ylower, xlower, yupper, xupper = np.int32(
            [ylower, xlower, yupper, xupper])

        yinds = yinds0[ylower:yupper, xlower:xupper]
        xinds = xinds0[ylower:yupper, xlower:xupper]

        self.centering_GaussianFit = np.zeros((self.imageCube.shape[0], 2))
        self.widths_GaussianFit = np.zeros((self.imageCube.shape[0], 2))

        self.heights_GaussianFit = np.zeros(self.imageCube.shape[0])
        # self.rotation_GaussianFit = np.zeros(self.imageCube.shape[0])
        self.background_GaussianFit = np.zeros(self.imageCube.shape[0])

        # Gaussian fit centering
        # This starts the multiprocessing call to arms
        # pool = Pool(self.nCores)

        func = partial(
            fit_one_center,
            n_sig=n_sig,
            method='gauss',
            ylower=ylower,
            yupper=yupper,
            xlower=xlower,
            xupper=xupper
        )

        # the order is very important
        gaussian_centers = pool_run_func(func, zip(self.imageCube))

        # pool.close()
        # pool.join()

        print('Finished with Fitting Centers. Now assigning to instance values.')
        for kf, gaussP in enumerate(gaussian_centers):
            self.assign_gaussian_centering(self, gaussP, xlower, kf, ylower)
            """ # TODO: confirm this is correct
            self.centering_GaussianFit[kf][self.x] = gaussP[1] + xlower
            self.centering_GaussianFit[kf][self.y] = gaussP[2] + ylower

            self.widths_GaussianFit[kf][self.x] = gaussP[3]
            self.widths_GaussianFit[kf][self.y] = gaussP[4]

            self.heights_GaussianFit[kf] = gaussP[0]
            self.background_GaussianFit[kf] = gaussP[5]
            """

        self.centering_df = pd.DataFrame()
        self.centering_df['Gaussian_Fit_Y_Centers'] = self.centering_GaussianFit.T[y]
        self.centering_df['Gaussian_Fit_X_Centers'] = self.centering_GaussianFit.T[x]
        self.centering_df['Gaussian_Mom_Y_Centers'] = self.centering_GaussianFit.T[y]
        self.centering_df['Gaussian_Mom_X_Centers'] = self.centering_GaussianFit.T[x]

        self.centering_df['Gaussian_Fit_Y_Widths'] = self.widths_GaussianFit.T[y]
        self.centering_df['Gaussian_Fit_X_Widths'] = self.widths_GaussianFit.T[x]

        self.centering_df['Gaussian_Fit_Heights'] = self.heights_GaussianFit
        self.centering_df['Gaussian_Fit_Offset'] = self.background_GaussianFit

        # self.centering_df['Gaussian_Fit_Rotation'] = self.rotation_GaussianFit

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

        y, x = 0, 1

        yinds0, xinds0 = np.indices(self.imageCube[0].shape)

        ylower = self.yguess - self.pix_rad
        yupper = self.yguess + self.pix_rad
        xlower = self.xguess - self.pix_rad
        xupper = self.xguess + self.pix_rad

        ylower, xlower, yupper, xupper = np.int32(
            [ylower, xlower, yupper, xupper])

        yinds = yinds0[ylower:yupper, xlower:xupper]
        xinds = xinds0[ylower:yupper, xlower:xupper]

        nFWCParams = 2  # Xc, Yc
        self.centering_FluxWeight = np.zeros((self.nFrames, nFWCParams))
        print(self.imageCube.shape)
        for kf in self.tqdm(range(self.nFrames), desc='FWC', leave=False, total=self.nFrames):
            subFrameNow = self.imageCube[kf][ylower:yupper, xlower:xupper]
            subFrameNow[np.isnan(subFrameNow)] = np.nanmedian(
                ~np.isnan(subFrameNow))

            self.centering_FluxWeight[kf] = flux_weighted_centroid(
                self.imageCube[kf],
                self.yguess,
                self.xguess,
                bSize=7
            )
            self.centering_FluxWeight[kf] = self.centering_FluxWeight[kf][::-1]

        self.centering_FluxWeight[:, 0] = clipOutlier(
            self.centering_FluxWeight.T[0])
        self.centering_FluxWeight[:, 1] = clipOutlier(
            self.centering_FluxWeight.T[1])

        self.centering_df['FluxWeighted_Y_Centers'] = self.centering_FluxWeight.T[self.y]
        self.centering_df['FluxWeighted_X_Centers'] = self.centering_FluxWeight.T[self.x]

    def mp_fit_flux_weighted_centering(self, n_sig=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        yinds0, xinds0 = np.indices(self.imageCube[0].shape)

        ylower = self.yguess - self.pix_rad
        yupper = self.yguess + self.pix_rad
        xlower = self.xguess - self.pix_rad
        xupper = self.xguess + self.pix_rad

        ylower, xlower, yupper, xupper = np.int32(
            [ylower, xlower, yupper, xupper])

        yinds = yinds0[ylower:yupper, xlower:xupper]
        xinds = xinds0[ylower:yupper, xlower:xupper]

        nFWCParams = 2  # Xc, Yc
        # self.centering_FluxWeight = np.zeros((self.nFrames, nFWCParams))

        # This starts the multiprocessing call to arms
        # pool = Pool(self.nCores)

        func = partial(
            fit_one_center,
            n_sig=n_sig,
            method='fwc',
            ylower=ylower,
            yupper=yupper,
            xlower=xlower,
            xupper=xupper,
            bSize=7
        )

        # the order is very important
        fwc_centers = pool_run_func(func, zip(self.imageCube))

        # pool.close()
        # pool.join()

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

        y, x = 0, 1

        yinds0, xinds0 = np.indices(self.imageCube[0].shape)

        ylower = self.yguess - self.pix_rad
        yupper = self.yguess + self.pix_rad
        xlower = self.xguess - self.pix_rad
        xupper = self.xguess + self.pix_rad

        ylower, xlower, yupper, xupper = np.int32(
            [ylower, xlower, yupper, xupper])

        yinds = yinds0[ylower:yupper+1, xlower:xupper+1]
        xinds = xinds0[ylower:yupper+1, xlower:xupper+1]

        nAsymParams = 2  # Xc, Yc
        self.centering_LeastAsym = np.zeros((self.nFrames, nAsymParams))

        for kf in self.tqdm(range(self.nFrames), desc='Asym', leave=False, total=self.nFrames):
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

            fitFailed = False  # "Benefit of the Doubt"

            # Option 1: We run `actr` with default settings that were developed for Spitzer exoplanet lightcurves
            kf, yguess, xguess = np.int32([kf, self.yguess, self.xguess])

            try:
                center_asym = actr(self.imageCube[kf], [yguess, xguess],
                                   asym_rad=8, asym_size=5, maxcounts=2, method='gaus',
                                   half_pix=False, resize=False, weights=False)[0]
            except:
                fitFailed = True

            # Option 2: We assume that there is a deformation in the PSF and square the image array
            #  (preserving signs)
            if fitFailed:
                try:
                    center_asym = actr(np.sign(self.imageCube[kf])*self.imageCube[kf]**2,
                                       [yguess, xguess],
                                       asym_rad=8, asym_size=5, maxcounts=2, method='gaus',
                                       half_pix=False, resize=False, weights=False)[0]
                    fitFailed = False
                except:
                    pass

            # Option 3: We assume that there is a cosmic ray hit nearby the PSF and shrink the asym_rad by half
            if fitFailed:
                try:
                    center_asym = actr(self.imageCube[kf], [yguess, xguess],
                                       asym_rad=4, asym_size=5, maxcounts=2, method='gaus',
                                       half_pix=False, resize=False, weights=False)[0]
                    fitFailed = False
                except:
                    pass

            # Option 4: We assume that there is a deformation in the PSF AND (or caused by) a cosmic ray hit
            #   nearby the PSF; so we square the image array (preserving signs) AND shrink the asym_rad by half
            if fitFailed:
                try:
                    center_asym = actr(np.sign(self.imageCube[kf])*self.imageCube[kf]**2,
                                       [yguess, xguess],
                                       asym_rad=4, asym_size=5, maxcounts=2, method='gaus',
                                       half_pix=False, resize=False, weights=False)[0]
                    fitFailed = False
                except:
                    pass

            # Option 5: We assume that there is a deformation in the PSF AND (or caused by) a cosmic ray hit
            #   nearby the PSF; so we square the image array (preserving signs) AND shrink the asym_rad by half
            #   BUT this time we have to get crazy and shrink the asym_size to 3 (reduces accuracy dramatically)
            if fitFailed:
                try:
                    center_asym = actr(np.sign(self.imageCube[kf])*self.imageCube[kf]**2,
                                       [yguess, xguess],
                                       asym_rad=4, asym_size=3, maxcounts=2, method='gaus',
                                       half_pix=False, resize=False, weights=False)[0]
                    fitFailed = False
                except:
                    pass

            if fitFailed:
                # I ran out of options -- literally
                print('Least Asymmetry FAILED: and returned `RuntimeError`')

            try:
                # This will work if the fit was successful
                self.centering_LeastAsym[kf] = center_asym[::-1]
            except:
                print('Least Asymmetry FAILED: and returned `NaN`')
                fitFailed = True

            if fitFailed:
                print('Least Asymmetry FAILED: Setting self.centering_LeastAsym[%s] to Initial Guess: [%s,%s]'
                      % (kf, self.yguess, self.xguess))
                self.centering_LeastAsym[kf] = np.array([yguess, xguess])

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

        y, x = 0, 1

        yinds0, xinds0 = np.indices(self.imageCube[0].shape)

        ylower = self.yguess - self.pix_rad
        yupper = self.yguess + self.pix_rad
        xlower = self.xguess - self.pix_rad
        xupper = self.xguess + self.pix_rad

        ylower, xlower, yupper, xupper = np.int32(
            [ylower, xlower, yupper, xupper])

        yguess, xguess = np.int32([self.yguess, self.xguess])

        yinds = yinds0[ylower:yupper+1, xlower:xupper+1]
        xinds = xinds0[ylower:yupper+1, xlower:xupper+1]

        nAsymParams = 2  # Xc, Yc
        # self.centering_LeastAsym = np.zeros((self.nFrames, nAsymParams))
        # for kf in self.tqdm(range(self.nFrames), desc='Asym', leave = False, total=self.nFrames):
        # This starts the multiprocessing call to arms
        # pool = Pool(self.nCores)

        func = partial(
            actr,
            asym_rad=8,
            asym_size=5,
            maxcounts=2,
            method='gaus',
            half_pix=False,
            resize=False,
            weights=False
        )

        self.centering_LeastAsym = pool_run_func(func, zip(
            self.imageCube, [[yguess, xguess]]*self.nframes))  # the order is very important

        # pool.close()
        # pool.join()

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
            lower = midFrame - self.pix_rad
            upper = midFrame + self.pix_rad

            image_view = self.imageCube[:, lower:upper, lower:upper]
            self.effective_widths = image_view.sum(
                axis=(1, 2))**2. / (image_view**2).sum(axis=(1, 2))
        else:
            self.effective_widths = self.imageCube.sum(
                axis=(1, 2))**2. / ((self.imageCube)**2).sum(axis=(1, 2))

        self.centering_df['Effective_Widths'] = self.effective_widths

        x_widths = self.centering_df['Gaussian_Fit_X_Widths']
        y_widths = self.centering_df['Gaussian_Fit_Y_Widths']

        self.quadrature_widths = np.sqrt(x_widths**2 + y_widths**2)
        self.centering_df['Quadrature_Widths'] = self.quadrature_widths

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
            Assigning all np.zeros in the mask to NaNs because the `mean` and `median`
                functions are set to `nanmean` functions, which will skip all NaNs
        """

        self.background_CircleMask = np.zeros(self.nFrames)
        for kf in self.tqdm(range(self.nFrames), desc='CircleBG', leave=False, total=self.nFrames):
            aperture = CircularAperture(self.centering_FluxWeight[kf], aperRad)

            # list of ApertureMask objects (one for each position)
            aper_mask = aperture.to_mask(method='exact')[0]

            # backgroundMask = abs(aperture.get_fractions(np.ones(self.imageCube[0].shape))-1)
            backgroundMask = aper_mask.to_image(
                self.imageCube[0].shape).astype(bool)
            backgroundMask = ~backgroundMask  # [backgroundMask == 0] = False

            self.background_CircleMask[kf] = self.metric(
                self.imageCube[kf][backgroundMask])

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
            Assigning all np.zeros in the mask to NaNs because the `mean` and `median`
                functions are set to `nanmean` functions, which will skip all NaNs
        """

        if centering == 'Gauss':
            centers = self.centering_GaussianFit
        if centering == 'FluxWeight':
            centers = self.centering_FluxWeight

        # This starts the multiprocessing call to arms
        # pool = Pool(self.nCores)

        func = partial(
            measure_one_circle_bg,
            aperRad=aperRad,
            metric=self.metric,
            apMethod='exact'
        )

        self.background_CircleMask = pool_run_func(
            func, zip(self.imageCube, centers)
        )

        # pool.close()
        # pool.join()

        self.background_CircleMask = np.array(self.background_CircleMask)
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

        for kf in self.tqdm(range(self.nFrames), desc='AnnularBG', leave=False, total=self.nFrames):
            innerAperture = CircularAperture(
                self.centering_FluxWeight[kf], innerRad)
            outerAperture = CircularAperture(
                self.centering_FluxWeight[kf], outerRad)

            inner_aper_mask = innerAperture.to_mask(method='exact')[0]
            inner_aper_mask = inner_aper_mask.to_image(
                self.imageCube[0].shape).astype(bool)

            outer_aper_mask = outerAperture.to_mask(method='exact')[0]
            outer_aper_mask = outer_aper_mask.to_image(
                self.imageCube[0].shape).astype(bool)

            backgroundMask = (~inner_aper_mask)*outer_aper_mask

            self.background_Annulus[kf] = self.metric(
                self.imageCube[kf][backgroundMask])

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

        if centering == 'Gauss':
            centers = self.centering_GaussianFit
        if centering == 'FluxWeight':
            centers = self.centering_FluxWeight

        # This starts the multiprocessing call to arms
        # pool = Pool(self.nCores)

        func = partial(
            measure_one_annular_bg,
            innerRad=innerRad,
            outerRad=outerRad,
            metric=self.metric,
            apMethod='exact'
        )

        self.background_Annulus = pool_run_func(
            func, zip(self.imageCube, centers))  # the order is very important

        # pool.close()
        # pool.join()

        self.background_Annulus = np.array(self.background_Annulus)
        self.background_df['AnnularMask'] = self.background_Annulus.copy()

    def measure_background_median_masked(self, aperRad=10, n_sig=5):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        self.background_MedianMask = np.zeros(self.nFrames)

        for kf in self.tqdm(range(self.nFrames), desc='MedianMaskedBG', leave=False, total=self.nFrames):
            aperture = CircularAperture(self.centering_FluxWeight[kf], aperRad)
            aperture = aperture.to_mask(method='exact')[0]
            aperture = aperture.to_image(self.imageCube[0].shape).astype(bool)
            backgroundMask = ~aperture

            medFrame = np.nanmedian(self.imageCube[kf][backgroundMask])
            # scale.mad(self.imageCube[kf][backgroundMask])
            madFrame = np.nanstd(self.imageCube[kf][backgroundMask])

            medianMask = abs(self.imageCube[kf] - medFrame) < n_sig*madFrame

            maskComb = medianMask*backgroundMask
            # maskComb[maskComb == 0] = False

            self.background_MedianMask[kf] = np.nanmedian(
                self.imageCube[kf][maskComb])

        self.background_df['MedianMask'] = self.background_MedianMask

    def mp_measure_background_median_masked(self, aperRad=10, n_sig=5, centering='Gauss'):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        if centering == 'Gauss':
            centers = self.centering_GaussianFit
        if centering == 'FluxWeight':
            centers = self.centering_FluxWeight

        # This starts the multiprocessing call to arms
        # pool = Pool(self.nCores)

        func = partial(measure_one_median_bg, aperRad=aperRad,
                       apMethod='exact', metric=self.metric, n_sig=n_sig)

        self.background_MedianMask = pool_run_func(
            func, zip(self.imageCube, centers))

        # pool.close()
        # pool.join()

        self.background_MedianMask = np.array(self.background_MedianMask)
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

        for kf in self.tqdm(range(self.nFrames), desc='KDE_BG', leave=False, total=self.nFrames):
            aperture = CircularAperture(self.centering_FluxWeight[kf], aperRad)
            aperture = aperture.to_mask(method='exact')[0]
            aperture = aperture.to_image(self.imageCube[0].shape).astype(bool)
            backgroundMask = ~aperture

            kdeFrame = kde.KDEUnivariate(
                self.imageCube[kf][backgroundMask].ravel())
            kdeFrame.fit()

            self.background_KDEUniv[kf] = kdeFrame.support[kdeFrame.density.argmax(
            )]

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

        if centering == 'Gauss':
            centers = self.centering_GaussianFit
        if centering == 'FluxWeight':
            centers = self.centering_FluxWeight

        self.background_KDEUniv = np.zeros(self.nFrames)

        # This starts the multiprocessing call to arms
        # pool = Pool(self.nCores)

        func = partial(
            measure_one_KDE_bg,
            aperRad=aperRad,
            apMethod='exact',
            metric=self.metric
        )

        self.background_KDEUniv = pool_run_func(
            func,
            zip(self.imageCube, centers)
        )  # the order is very important

        # pool.close()
        # pool.join()

        self.background_KDEUniv = np.array(self.background_KDEUniv)
        self.background_df['KDEUnivMask_mp'] = self.background_KDEUniv

    def measure_all_background(self, aperRad=10, n_sig=5):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        pInner = 0.2  # Percent Inner = -20%
        pOuter = 0.3  # Percent Outer = +30%

        print('Measuring Background Using Circle Mask with Multiprocessing')
        self.mp_measure_background_circle_masked(aperRad=aperRad)

        print('Measuring Background Using Annular Mask with Multiprocessing')
        self.mp_measure_background_annular_mask(
            innerRad=(1-pInner)*aperRad, outerRad=(1+pOuter)*aperRad)

        print('Measuring Background Using Median Mask with Multiprocessing')
        self.mp_measure_background_median_masked(aperRad=aperRad, n_sig=n_sig)

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

        y, x = 0, 1

        if background not in self.background_df.columns:
            raise KeyError("`background` must be in",
                           self.background_df.columns)

        centering_options = [
            'Gaussian_Fit',
            'Gaussian_Mom',
            'FluxWeighted',
            'LeastAsymmetry'
        ]
        if centering not in centering_options:
            raise ValueError(
                "`centering` must be either 'Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', or 'LeastAsymmetry'")

        if aperRad is None:
            staticRad = 70 if 'wlp' in self.fitsFilenames[0].lower() else 3
            """
            if 'wlp' in self.fitsFilenames[0].lower():
                staticRad = 70
            else:
                staticRad = 3
            """

        y_centers = self.centering_df[centering + '_Y_Centers']
        x_centers = self.centering_df[centering + '_X_Centers']
        centering_Use = np.transpose([y_centers, x_centers])

        background_Use = self.background_df[background]

        flux_key_now = f"{centering}_{background}_rad_{aperRad}"

        if flux_key_now not in self.flux_TSO_df.keys() or useTheForce:
            flux_TSO_now = np.zeros(self.nFrames)
            noise_TSO_now = np.zeros(self.nFrames)

            for kf in self.tqdm(range(self.nFrames), desc='Flux', leave=False, total=self.nFrames):
                frameNow = np.copy(self.imageCube[kf]) - background_Use[kf]
                frameNow[np.isnan(frameNow)] = np.nanmedian(frameNow)

                noiseNow = np.copy(self.noiseCube[kf])**2.
                noiseNow[np.isnan(noiseNow)] = np.nanmedian(noiseNow)

                x_center_ = centering_Use[kf][self.x]
                y_center_ = centering_Use[kf][self.y]
                aperture = CircularAperture([x_center_, y_center_], r=aperRad)

                flux_TSO_now[kf] = aperture_photometry(
                    frameNow, aperture)['aperture_sum']
                noise_TSO_now[kf] = np.sqrt(aperture_photometry(
                    noiseNow, aperture)['aperture_sum'])

            self.flux_TSO_df[flux_key_now] = flux_TSO_now
            self.noise_TSO_df[flux_key_now] = noise_TSO_now
        else:
            print(
                f'{flux_key_now} exists: '
                'if you want to overwrite, then you `useTheForce=True`'
            )

    def compute_flux_over_time_over_aperRad(
            self, aperRads=None, centering_choices=None,
            background_choices=None, useTheForce=False, verbose=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        ppm = 1e6
        start = time()
        for bgNow in background_choices:
            for ctrNow in centering_choices:
                for aperRad in aperRads:
                    if verbose:
                        print(
                            f'Working on Background {bgNow} '
                            f'with Centering {ctrNow} and AperRad {aperRad}',
                            end=" ",
                        )

                    self.compute_flux_over_time(
                        aperRad=aperRad,
                        centering=ctrNow,
                        background=bgNow,
                        useTheForce=useTheForce
                    )

                    flux_key_now = f"{ctrNow}_{bgNow}_rad_{aperRad}"

                    if verbose:
                        flux_now = self.flux_TSO_df[flux_key_now]
                        print(
                            np.nanstd(flux_now / np.nanmedian(flux_now)) * ppm
                        )

        print('Operation took: ', time()-start)

    def mp_compute_flux_over_time(
            self, aperRad=3.0, centering='GaussianFit',
            background='AnnularMask', useTheForce=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        y, x = 0, 1

        if background not in self.background_df.columns:
            raise KeyError("`background` must be in",
                           self.background_df.columns)

        if centering not in ['Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', 'LeastAsymmetry']:
            raise Exception(
                "`centering` must be either 'Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', or 'LeastAsymmetry'")

        centering_Use = np.transpose([self.centering_df[centering + '_Y_Centers'],
                                      self.centering_df[centering + '_X_Centers']])

        background_Use = self.background_df[background]

        flux_key_now = centering + '_' + background + \
            '_' + 'rad' + '_' + str(aperRad)

        if flux_key_now not in self.flux_TSO_df.keys() or useTheForce:
            # for kf in self.tqdm(range(self.nFrames), desc='Flux', leave = False, total=self.nFrames):

            # pool = Pool(self.nCores)

            func = partial(compute_flux_one_frame, aperRad=aperRad)

            fluxNow = pool_run_func(
                func, zip(self.imageCube, centering_Use, background_Use)
            )

            # pool.close()
            # pool.join()

            # fluxNow[~np.isfinite(fluxNow)] = np.nanmedian(fluxNow[np.isfinite(fluxNow)])
            # fluxNow[fluxNow < 0] = np.nanmedian(fluxNow[fluxNow > 0])

            self.flux_TSO_df[flux_key_now] = fluxNow
            self.noise_TSO_df[flux_key_now] = np.sqrt(fluxNow)

        else:
            print(
                flux_key_now + ' exists: if you want to overwrite, then you `useTheForce=True`')

    def mp_compute_flux_over_time_varRad(self, staticRad, varRad=None, centering='Gaussian_Fit', background='AnnularMask', useTheForce=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        y, x = 0, 1

        if background not in self.background_df.columns:
            raise KeyError("`background` must be in",
                           self.background_df.columns)

        centering_options = [
            'Gaussian_Fit',
            'Gaussian_Mom',
            'FluxWeighted',
            'LeastAsymmetry'
        ]
        if centering not in centering_options:
            raise KeyError(
                "`centering` must be either 'Gaussian_Fit', 'Gaussian_Mom', "
                "'FluxWeighted', or 'LeastAsymmetry'"
            )

        y_center_ = self.centering_df[centering + '_Y_Centers']
        x_center_ = self.centering_df[centering + '_X_Centers']
        centering_Use = np.transpose([y_center_, x_center_])

        background_Use = self.background_df[background]

        flux_key_now = f'{centering}_{background}_rad_{staticRad}_{varRad}'

        assert (isinstance(staticRad, (float, int))), (
            'staticRad must be either a float or an integer'
        )

        if varRad is None or varRad == 0.0:
            aperRads = [staticRad] * self.nFrames
        else:
            med_quad_rad_dist = np.nanmedian(self.quadrature_widths)
            quad_rad_dist = self.quadrature_widths.copy() - med_quad_rad_dist
            quad_rad_dist = clipOutlier(quad_rad_dist, n_sig=5)
            aperRads = staticRad + varRad*quad_rad_dist

        if flux_key_now not in self.flux_TSO_df.keys() or useTheForce:
            # for kf in self.tqdm(range(self.nFrames), desc='Flux', leave = False, total=self.nFrames):

            # pool = Pool(self.nCores)

            # func = partial(compute_flux_one_frame)

            fluxNow = pool_run_func(compute_flux_one_frame,
                                    zip(self.imageCube, centering_Use, background_Use, aperRads))

            # pool.close()
            # pool.join()

            # fluxNow[~np.isfinite(fluxNow)] = np.nanmedian(fluxNow[np.isfinite(fluxNow)])
            # fluxNow[fluxNow < 0] = np.nanmedian(fluxNow[fluxNow > 0])

            self.flux_TSO_df[flux_key_now] = fluxNow
            self.noise_TSO_df[flux_key_now] = np.sqrt(fluxNow)

        else:
            print(
                flux_key_now + ' exists: if you want to overwrite, then you `useTheForce=True`')

    def mp_compute_flux_over_time_betaRad(self, centering='Gaussian_Fit',
                                          background='AnnularMask',
                                          useQuad=False,
                                          useTheForce=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        y, x = 0, 1

        if background not in self.background_df.columns:
            raise Exception("`background` must be in",
                            self.background_df.columns)

        if centering not in ['Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', 'LeastAsymmetry']:
            raise Exception(
                "`centering` must be either 'Gaussian_Fit', 'Gaussian_Mom', 'FluxWeighted', or 'LeastAsymmetry'")

        centering_Use = np.transpose([self.centering_df[centering + '_Y_Centers'],
                                      self.centering_df[centering + '_X_Centers']])

        background_Use = self.background_df[background]

        flux_key_now = centering + '_' + background+'_' + 'rad'

        flux_key_now = flux_key_now + \
            '_quadRad_0.0_0.0' if useQuad else flux_key_now + '_betaRad_0.0_0.0'

        if flux_key_now not in self.flux_TSO_df.keys() or useTheForce:
            # for kf in self.tqdm(range(self.nFrames), desc='Flux', leave = False, total=self.nFrames):
            sig2FW = 2*np.sqrt(2*log(2))
            aperRads = sig2FW * \
                self.quadrature_widths if useQuad else np.sqrt(
                    self.effective_widths)

            # pool = Pool(self.nCores)

            # func = partial(compute_flux_one_frame)

            fluxNow = pool_run_func(compute_flux_one_frame, zip(
                self.imageCube, centering_Use, background_Use, aperRads))

            # pool.close()
            # pool.join()

            # fluxNow[~np.isfinite(fluxNow)] = np.nanmedian(fluxNow[np.isfinite(fluxNow)])
            # fluxNow[fluxNow < 0] = np.nanmedian(fluxNow[fluxNow > 0])

            self.flux_TSO_df[flux_key_now] = fluxNow
            self.noise_TSO_df[flux_key_now] = np.sqrt(fluxNow)

        else:
            print(
                flux_key_now + ' exists: if you want to overwrite, then you `useTheForce=True`')

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

        nPLDComp = nCols*nRows

        # nCols = nCols // 2 # User input assumes matrix structure, which starts at y- and x-center
        # nRows = nRows // 2 #   and moves +\- nRows/2 and nCols/2, respectively

        # nominally 15
        ycenter = int(
            ycenter) if ycenter is not None else self.imageCube.shape[1]//2-1
        # nominally 15
        xcenter = int(
            xcenter) if xcenter is not None else self.imageCube.shape[2]//2-1

        ylower = ycenter - nRows // 2     # nominally 14
        yupper = ycenter + nRows // 2 + 1  # nominally 17 (to include 16)
        xlower = xcenter - nCols // 2     # nominally 14
        xupper = xcenter + nCols // 2 + 1  # nominally 17 (to include 16)

        PLD_comps_local = self.imageCube[:, ylower:yupper, xlower:xupper].reshape(
            (self.imageCube.shape[0], nPLDComp)).T

        PLD_norm = sum(PLD_comps_local, axis=0)
        PLD_comps_local = PLD_comps_local / PLD_norm

        self.PLD_components = PLD_comps_local
        self.PLD_norm = PLD_norm

        if order > 1:
            for o in range(2, order+1):
                self.PLD_components = vstack(
                    [self.PLD_components, PLD_comps_local**o])

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
        y, x = 0, 1

        if centering == 'gaussian':
            ycenters = self.centering_GaussianFit.T[y]
            xcenters = self.centering_GaussianFit.T[x]
        else:
            print('Only Gaussian Centering is supported at the moment')
            print('Continuting with Gaussian Centering')
            ycenters = self.centering_GaussianFit.T[y]
            xcenters = self.centering_GaussianFit.T[x]

        try:
            self.inliers_Phots = self.inliers_Phots
        except:
            self.inliers_Phots = {}

        for flux_key_now in self.flux_TSO_df.keys():

            phots = self.flux_TSO_df[flux_key_now]

            if flux_key_now not in self.inliers_Phots.keys() or useTheForce:
                self.inliers_Phots[flux_key_now] = DBScan_Segmented_Flux(
                    phots, ycenters, xcenters, dbsClean=dbsClean)
            else:
                print(
                    flux_key_now + ' exists: if you want to overwrite, then you `useTheForce=True`')

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
            ycenters = self.centering_GaussianFit.T[y]
            xcenters = self.centering_GaussianFit.T[x]
        else:
            print('Only Gaussian Centering is supported at the moment')
            print('Continuting with Gaussian Centering')
            ycenters = self.centering_GaussianFit.T[y]
            xcenters = self.centering_GaussianFit.T[x]

        try:
            self.inliers_Phots = self.inliers_Phots
        except:
            self.inliers_Phots = {}

        # This starts the multiprocessing call to arms
        # pool = Pool(self.nCores)

        func = partial(DBScan_Flux, ycenters=ycenters,
                       xcenters=xcenters, dbsClean=dbsClean)

        # the order is very important
        inliersMP = pool_run_func(func, zip(self.flux_TSO_df.values.T))

        # pool.close()
        # pool.join()

        for k_mp, flux_key_now in enumerate(self.flux_TSO_df.keys()):
            self.inliers_Phots[flux_key_now] = inliersMP[k_mp]

    def mp_DBScan_PLD_All(self, dbsClean=0):
        raise Exception(
            'This Function is Not Working; please use lame_`DBScan_PLD_all`'
        )
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        if not hasattr(self, "inliers_PLD"):
            self.inliers_PLD = np.ones(self.PLD_components.shape, dtype=bool)

        # This starts the multiprocessing call to arms
        # pool = Pool(self.nCores)

        func = partial(DBScan_Segmented_Flux, dbsClean=dbsClean)

        # the order is very important
        inliersMP = pool_run_func(func, zip(self.PLD_components.T))

        # pool.close()
        # pool.join()

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

        if not hasattr(self, "inliers_PLD"):
            self.inliers_PLD = np.ones(self.PLD_components.shape, dtype=bool)

        for kPLD, PLDnow in enumerate(self.PLD_components):
            self.inliers_PLD[kPLD] = DBScan_PLD(PLDnow, dbsClean=dbsClean)
