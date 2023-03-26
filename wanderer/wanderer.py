import joblib
import numpy as np
# import matplotlib.pyplot as plt
import os
import pandas as pd
# import scipy as sp
# import sys

# import image_registration as ir

from astropy.io import fits
# from astropy.modeling import models, fitting
from datetime import datetime
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
# from sklearn.cluster import DBSCAN

# from sklearn.preprocessing import StandardScaler
# from skimage.filters import gaussian as gaussianFilter
# from socket import gethostname
from statsmodels.robust import scale
from statsmodels.nonparametric import kde
from time import time, localtime  # , sleep
from tqdm import tqdm, tqdm_notebook

# from skimage.filters import gaussian as gaussianFilter
# import everything that `Wanderer` needs to operate
# from .utils import *
from .utils import (
    # actr,
    clip_outlier,
    command_line_inputs,
    compute_flux_one_frame,
    create_aper_mask,
    dbscan_flux,
    dbscan_pld,
    dbscan_segmented_flux,
    fit_gauss,
    fitgaussian,
    fit_one_center,
    flux_weighted_centroid,
    gaussian,
    get_julian_date_from_header,
    grab_dir_and_filenames,
    lmfit_one_center,
    measure_one_annular_bg,
    measure_one_circle_bg,
    measure_one_median_bg,
    measure_one_kde_bg,
    moments,
    pool_run_func,
    WandererCLI
)


class Wanderer(object):
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
            self, fits_file_dir='./', filetype='slp.fits', telescope=None,
            yguess=None, xguess=None, pix_rad=5, method='mean', num_cores=None,
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

        self.fits_file_dir = fits_file_dir
        self.fits_names = glob(f'{self.fits_file_dir}/*{self.filetype}')
        # self.fits_names = glob(self.fits_file_dir + '/*' + self.filetype)
        self.n_slope_files = len(self.fits_names)

        if telescope is None:
            raise ValueError(
                "Please specify `telescope` as either 'JWST' "
                "or 'Spitzer' or 'HST'"
            )

        self.telescope = telescope

        if self.telescope == 'Spitzer':
            fitsdir_split = self.fits_file_dir.replace('raw', 'cal').split('/')
            for _ in range(4):
                fitsdir_split.pop()

            # self.cal_dir = ''
            # for thing in fitsdir_split:
            #     self.cal_dir = self.cal_dir + thing + '/'
            #
            # self.permBadPixels = fits.open(
            #     self.cal_dir + 'nov14_ch1_bcd_pmask_subarray.fits'
            # )

        if self.n_slope_files == 0:
            print(
                'Pipeline found no Files in ' +
                self.fits_file_dir + ' of type /*' +
                filetype
            )
            exit(-1)

        self.centering_df = pd.DataFrame()
        self.background_df = pd.DataFrame()
        self.flux_tso_df = pd.DataFrame()
        self.noise_tso_df = pd.DataFrame()

        self.yguess = yguess
        if self.yguess is None:
            self.yguess = self.image_cube.shape[self.y]//2

        self.xguess = xguess
        if self.xguess is None:
            self.xguess = self.image_cube.shape[self.x]//2

        self.pix_rad = pix_rad
        self.num_cores = cpu_count()//2 if num_cores is None else int(num_cores)

        # TODO: Use timestamp reformatting methods
        tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, \
            tm_isdst = localtime()
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
        testfits = fits.open(self.fits_names[0])[0]

        self.n_frames = self.n_slope_files
        self.image_cube = np.zeros(
            (self.n_frames, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.noise_cube = np.zeros(
            (self.n_frames, testfits.data[0].shape[0], testfits.data[0].shape[1]))
        self.time_cube = np.zeros(self.n_frames)

        self.bad_pixel_masks = None

        del testfits

        progress_fits_filenames = self.tqdm(
            enumerate(self.fits_names),
            desc='JWST Load File',
            leave=False,
            total=self.n_slope_files
        )

        for kf, fname in progress_fits_filenames:
            fits_now = fits.open(fname)

            self.image_cube[kf] = fits_now[0].data[0]
            self.noise_cube[kf] = fits_now[0].data[1]

            # re-write these 4 lines into `get_julian_date_from_header`
            start_jd, end_jd = get_julian_date_from_header(fits_now[0].header)
            timeSpan = (end_jd - start_jd) * self.day2sec / self.n_frames

            self.time_cube[kf] = start_jd + timeSpan * (kf + 0.5)
            self.time_cube[kf] = self.time_cube[kf] / self.day2sec - 2450000.

            del fits_now[0].data

            fits_now.close()

            del fits_now

    def spitzer_load_fits_file(self, output_units='electrons', remove_nans=True):
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
        nframes_per_file = 64

        testfits = fits.open(self.fits_names[0])[0]
        testheader = testfits.header

        bcd_shape = testfits.data[0].shape
        # bcd_shape = testfits.data.shape

        self.n_frames = self.n_slope_files * nframes_per_file
        self.image_cube = np.zeros((self.n_frames, bcd_shape[0], bcd_shape[1]))
        self.noise_cube = np.zeros((self.n_frames, bcd_shape[0], bcd_shape[1]))
        self.time_cube = np.zeros(self.n_frames)

        self.bad_pixel_masks = None

        del testfits

        # Converts DN/s to microJy per pixel
        #   1) exp_time * gain / flux_conv converts MJ/sr to electrons
        #   2) as2sr * MJ2mJ * testheader['PXSCAL1'] * testheader['PXSCAL2']
        #       converts MJ/sr to muJ/pixel

        # for key, val in testheader.items():
        #     print(f'{key} = {val}')

        if output_units == 'electrons':
            flux_conv = testheader['FLUXCONV']
            exp_time = testheader['EXPTIME']
            gain = testheader['GAIN']
            flux_conversion = exp_time*gain / flux_conv

        elif output_units == 'muJ_per_Pixel':
            as2sr = arcsec**2.0  # steradians per square arcsecond
            MJ2mJ = 1e12  # mircro-Janskeys per Mega-Jansky
            # converts MJ
            flux_conversion = abs(
                as2sr * MJ2mJ * testheader['PXSCAL1'] * testheader['PXSCAL2']
            )
        else:
            raise ValueError(
                "`output_units` must be either 'electrons' or 'muJ_per_Pixel'"
            )

        print('Loading Spitzer Data')

        progress_fits_filenames = self.tqdm(
            enumerate(self.fits_names),
            desc='Spitzer Load File',
            leave=False,
            total=self.n_slope_files
        )
        for kfile, fname in progress_fits_filenames:
            bcd_now = fits.open(fname)
            bunc_now = fits.open(fname.replace('bcd.fits', 'bunc.fits'))

            for iframe in range(nframes_per_file):
                idx_ = kfile * nframes_per_file + iframe
                header_ = bcd_now[0].header
                bmjd_obs_ = header_['BMJD_OBS']
                et_obs_ = header_['ET_OBS']
                utcs_obs_ = header_['UTCS_OBS']
                frametime_ = float(header_['FRAMTIME'])

                # Convert from exposure time to UTC
                et_to_utc = (et_obs_ - utcs_obs_) * sec2day

                # Convert from exposure time to frame time in UTC
                frametime_adjust = et_to_utc + iframe * frametime_ * sec2day

                # Same time as BMJD with UTC correction
                self.time_cube[idx_] = bmjd_obs_ + frametime_adjust

                self.image_cube[idx_] = bcd_now[0].data[iframe] * \
                    flux_conversion
                self.noise_cube[idx_] = bunc_now[0].data[iframe] * \
                    flux_conversion

            def delete_fits_data(fits_data):
                del fits_data[0].data
                fits_data.close()
                del fits_data

            delete_fits_data(bcd_now)
            delete_fits_data(bunc_now)
            # del bcd_now[0].data
            # bcd_now.close()
            # del bcd_now

            # del bunc_now[0].data
            # bunc_now.close()
            # del bunc_now

        if remove_nans:
            # Set NaNs to Median
            # where_is_nan_ = np.where(np.isnan(self.image_cube))
            is_nan_ = np.isnan(self.image_cube)
            self.image_cube[is_nan_] = np.nanmedian(self.image_cube)

    def hst_load_fits_file(self, fits_now):
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
            self, savefiledir=None, save_name_header=None,
            save_file_type='.pickle.save'):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        if save_name_header is None:
            raise ValueError(
                '`save_name_header` should be the beginning '
                'of each save file name'
            )

        if savefiledir is None:
            savefiledir = './'

        print('Loading from Master Files')
        file_path_base = os.path.join(savefiledir, save_name_header)
        file_path_template = file_path_base + '{0}' + save_file_type

        print(f"Loading {file_path_template.format('_centering_dataframe')}")
        self.centering_df = joblib.load(
            file_path_template.format('_centering_dataframe')
        )

        print(f"Loading {file_path_template.format('_background_dataframe')}")
        self.background_df = joblib.load(
            file_path_template.format('_background_dataframe')
        )

        print(f"Loading {file_path_template.format('_flux_tso_dataframe')}")
        self.flux_tso_df = joblib.load(
            file_path_template.format('_flux_tso_dataframe')
        )

        noise_file_path = file_path_template.format('_noise_tso_dataframe')
        if os.path.exists(noise_file_path):
            print(f"Loading {noise_file_path}")
            self.noise_tso_df = joblib.load(noise_file_path)
        else:
            print(f"Does Not Exist: {noise_file_path}")
            self.noise_tso_df = None

        print(f"Loading {file_path_template.format('_image_cube_array')}")
        self.image_cube = joblib.load(
            file_path_template.format('_image_cube_array')
        )

        print(f"Loading {file_path_template.format('_noise_cube_array')}")
        self.noise_cube = joblib.load(
            file_path_template.format('_noise_cube_array')
        )

        print(f"Loading {file_path_template.format('_time_cube_array')}")
        self.time_cube = joblib.load(
            file_path_template.format('_time_cube_array')
        )

        print(
            f"Loading {file_path_template.format('_image_bad_pix_cube_array')}"
        )
        self.bad_pixel_masks = joblib.load(
            file_path_template.format('_image_bad_pix_cube_array')
        )

        print(f"Loading {file_path_template.format('_save_dict')}")
        self.save_dict = joblib.load(
            file_path_template.format('_save_dict')
        )

        print('n_frames', 'n_frames' in self.save_dict.keys())
        print(
            'Assigning Parts of `self.save_dict` to individual data structures'
        )

        for key in self.save_dict.keys():
            exec("self." + key + " = self.save_dict['" + key + "']")

    def save_collection(self, file_path_template):
        dump_dict = {
            '_centering_dataframe': self.centering_df,
            '_background_dataframe': self.background_df,
            '_flux_tso_dataframe': self.flux_tso_df,
            '_image_cube_array': self.image_cube,
            '_noise_cube_array': self.noise_cube,
            '_time_cube_array': self.time_cube,
            '_image_bad_pix_cube_array': self.bad_pixel_masks,
            '_noise_tso_dataframe': self.noise_tso_df,
            # '_save_dict': self.save_dict,
        }

        for filename_, df_ in dump_dict.items():
            # joblib.dump(df_, file_path_template.format(filename_))
            df_.to_csv(file_path_template.format(filename_))

        joblib.dump(
            self.save_dict,
            file_path_template.format(_save_dict)
        )

    def save_data_to_save_files(
            self, savefiledir=None, save_name_header=None,
            save_file_type='.joblib.save', save_master=True, save_time=True):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        if save_name_header is None:
            raise ValueError(
                '`save_name_header` should be the beginning '
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

        save_file_type_bak = date_string + save_file_type
        self.initiate_save_dict()

        if save_master:
            print('\nSaving to Master File -- Overwriting Previous Master')

            file_path_base = os.path.join(savefiledir, save_name_header)
            file_path_template = file_path_base + '{0}' + save_file_type

            self.save_collection(file_path_template)

        if save_time:
            print('Saving to New TimeStamped File -- These Tend to Pile Up!')

            file_path_base = os.path.join(
                savefiledir, 'TimeStamped', save_name_header
            )
            file_path_template = file_path_base + '{0}' + save_file_type_bak
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
            'background_annulus',
            'background_circle_mask',
            'background_gaussian_fit',
            'background_kde_univ',
            'background_median_mask',
            'centering_fluxweight',
            'centering_gaussian_fit',
            'centering_least_asym',
            'effective_widths',
            'fits_file_dir',
            'fits_names',
            'heights_gaussian_fit',
            'inliers_phots',
            'inliers_pld',
            'inliers_master',
            'method',
            'pix_rad',
            'n_frames',
            'pld_components',
            'pld_norm',
            'quadrature_widths',
            'yguess',
            'xguess',
            'widths_gaussian_fit',
            'aor',
            'planet_name',
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

                print("\tself." + thing)  # +" does not yet exist")

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

        temp = Wanderer(
            fits_file_dir=self.fits_file_dir,
            filetype=self.filetype,
            telescope=self.telescope,
            yguess=self.yguess,
            xguess=self.xguess,
            pix_rad=self.pix_rad,
            method=self.method,
            num_cores=self.num_cores
        )

        temp.centering_df = self.centering_df
        temp.background_df = self.background_df
        temp.flux_tso_df = self.flux_tso_df
        temp.noise_tso_df = self.noise_tso_df

        temp.image_cube = self.image_cube
        temp.noise_cube = self.noise_cube
        temp.time_cube = self.time_cube

        temp.bad_pixel_masks = self.bad_pixel_masks

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
        self.imagecube_median = np.nanmedian(self.image_cube, axis=0)
        self.imagecube_mad = scale.mad(self.image_cube, axis=0)

        self.bad_pixel_masks = abs(self.image_cube - self.imagecube_median)
        self.bad_pixel_masks = self.bad_pixel_masks > n_sig*self.imagecube_mad

        # print(
        #   "There are " + str(np.sum(self.bad_pixel_masks)) + " 'Hot' Pixels"
        # )
        print(f"There are {str(np.sum(self.bad_pixel_masks))} 'Hot' Pixels")

        # self.image_cube[self.bad_pixel_masks] = nan

    def fit_gaussian_centering(
            self, method='la', initc='fw', sub_array=False, print_compare=False):
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

        yinds0, xinds0 = np.indices(self.image_cube[0].shape)

        ylower = np.int32(self.yguess - self.pix_rad)
        yupper = np.int32(self.yguess + self.pix_rad)
        xlower = np.int32(self.xguess - self.pix_rad)
        xupper = np.int32(self.xguess + self.pix_rad)

        # ylower, xlower, yupper, xupper = np.int32(
        #     [ylower, xlower, yupper, xupper]
        # )

        yinds = yinds0[ylower:yupper, xlower:xupper]
        xinds = xinds0[ylower:yupper, xlower:xupper]

        self.centering_gaussian_fit = np.zeros((self.image_cube.shape[0], 2))
        self.widths_gaussian_fit = np.zeros((self.image_cube.shape[0], 2))

        self.heights_gaussian_fit = np.zeros(self.image_cube.shape[0])
        # self.rotation_gaussian_fit = np.zeros(self.image_cube.shape[0])
        self.background_gaussian_fit = np.zeros(self.image_cube.shape[0])

        progress_kframe = self.tqdm(
            range(self.n_frames),
            desc='GaussFit',
            leave=False,
            total=self.n_frames
        )
        for kf in progress_kframe:
            subFrame_now = self.image_cube[kf][ylower:yupper, xlower:xupper]
            # subFrame_now[np.isnan(subFrame_now)] = np.nanmedian(~np.isnan(subFrame_now))
            subFrame_now[np.isnan(subFrame_now)] = np.nanmedian(subFrame_now)

            cmom = np.array(moments(subFrame_now))  # H, Xc, Yc, Xs, Ys, O

            if method == 'aperture_photometry':
                if initc == 'fluxweighted' and self.centering_fluxweight.sum():
                    fwc_ = self.centering_fluxweight[kf]
                    fwc_[self.y] = fwc_[self.y] - ylower
                    fwc_[self.x] = fwc_[self.x] - xlower
                    gaussI = np.hstack([cmom[0], fwc_, cmom[3:]])
                if initc == 'cm':
                    gaussI = np.hstack([cmom[0], cmom[1], cmom[2], cmom[3:]])

                # H, Xc, Yc, Xs, Ys, Th, O
                p_gauss = fit_gauss(subFrame_now, xinds, yinds, gaussI)

            if method == 'la':
                # , xinds, yinds, np.copy(cmom)) # H, Xc, Yc, Xs, Ys, Th, O
                p_gauss = fitgaussian(subFrame_now)

            self.centering_gaussian_fit[kf][self.x] = p_gauss[1] + xlower
            self.centering_gaussian_fit[kf][self.y] = p_gauss[2] + ylower

            self.widths_gaussian_fit[kf][self.x] = p_gauss[3]
            self.widths_gaussian_fit[kf][self.y] = p_gauss[4]

            self.heights_gaussian_fit[kf] = p_gauss[0]
            self.background_gaussian_fit[kf] = p_gauss[5]

            del p_gauss, cmom

        self.centering_df = pd.DataFrame()
        self.centering_df['gaussian_fit_ycenters'] = self.centering_gaussian_fit.T[self.y]
        self.centering_df['gaussian_fit_xcenters'] = self.centering_gaussian_fit.T[self.x]
        self.centering_df['gaussian_mom_ycenters'] = self.centering_gaussian_fit.T[self.y]
        self.centering_df['gaussian_mom_xcenters'] = self.centering_gaussian_fit.T[self.x]

        self.centering_df['gaussian_fit_y_widths'] = self.widths_gaussian_fit.T[self.y]
        self.centering_df['gaussian_fit_x_widths'] = self.widths_gaussian_fit.T[self.x]

        self.centering_df['gaussian_fit_heights'] = self.heights_gaussian_fit
        self.centering_df['gaussian_fit_offset'] = self.background_gaussian_fit

    def mp_lmfit_gaussian_centering(
            self, yguess=15, xguess=15, sub_array_size=10, init_params=None,
            useMoments=False, num_cores=cpu_count()-1, center_range=None,
            width_range=None, n_sig=6.1, method='leastsq', recheck_method=None,
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
        if np.isnan(self.image_cube).any():
            is_nan_ = np.isnan(self.image_cube)
            self.image_cube[is_nan_] = np.nanmedian(self.image_cube)

        imageSize = self.image_cube.shape[1]

        nparams = 6
        if init_params is None:
            useMoments = True
            init_params = moments(self.image_cube[0])

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

        yy0, xx0 = np.indices(self.image_cube[0].shape)

        pix_rad = sub_array_size//2
        ylower = self.yguess - self.pix_rad
        yupper = self.yguess + self.pix_rad
        xlower = self.xguess - self.pix_rad
        xupper = self.xguess + self.pix_rad

        ylower, xlower, yupper, xupper = np.int32(
            [ylower, xlower, yupper, xupper])

        yy = yy0[ylower:yupper, xlower:xupper]
        xx = xx0[ylower:yupper, xlower:xupper]

        # pool = Pool(num_cores)

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

        gaussian_centers = pool_run_func(func, zip(self.image_cube))

        # pool.close()
        # pool.join()

        print(
            'Finished with Fitting Centers. Now assigning to instance values.'
        )

        self.centering_gaussian_fit = np.zeros((self.image_cube.shape[0], 2))
        self.widths_gaussian_fit = np.zeros((self.image_cube.shape[0], 2))

        self.heights_gaussian_fit = np.zeros(self.image_cube.shape[0])
        self.background_gaussian_fit = np.zeros(self.image_cube.shape[0])

        gaussian_centers = np.array(gaussian_centers)

        # ['center_y']
        self.centering_gaussian_fit.T[self.y] = gaussian_centers.T[0]
        # ['center_x']
        self.centering_gaussian_fit.T[self.x] = gaussian_centers.T[1]

        # ['width_y']
        self.widths_gaussian_fit.T[self.y] = gaussian_centers.T[2]
        # ['width_x']
        self.widths_gaussian_fit.T[self.x] = gaussian_centers.T[3]

        self.heights_gaussian_fit[:] = gaussian_centers.T[4]  # ['height']
        self.background_gaussian_fit[:] = gaussian_centers.T[5]  # ['offset']

        if verbose:
            print('Rechecking corner cases:')

        medY, medX = np.nanmedian(self.centering_gaussian_fit, axis=0)
        stdX, stdY = np.nanstd(self.centering_gaussian_fit, axis=0)

        outliers = (((self.centering_gaussian_fit.T[y] - medY)/stdY)**2 + (
            (self.centering_gaussian_fit.T[x] - medX)/stdX)**2) > n_sig

        if recheck_method is not None and isinstance(recheck_method, str):
            for kf in np.where(outliers)[0]:
                if verbose:
                    # print('    Corner Case:\t{}\tPreviousSolution={}'.format(
                    #     kf, self.centering_gaussian_fit[kf]), end="\t")
                    print(
                        f'    Corner Case:\t{kf}'
                        f'\tPreviousSolution={self.centering_gaussian_fit[kf]}',
                        end="\t"
                    )

                p_gauss = lmfit_one_center(
                    self.image_cube[kf],
                    yy=yy,
                    xx=xx,
                    gfit_model=gfit_model,
                    lmfit_init_params=lmfit_init_params,
                    yupper=yupper,
                    ylower=ylower,
                    xupper=xupper,
                    xlower=xlower,
                    method=recheck_method
                )

                # ['center_y']
                self.centering_gaussian_fit[kf][self.y] = p_gauss[0]
                # ['center_x']
                self.centering_gaussian_fit[kf][self.x] = p_gauss[1]

                # ['width_y']
                self.widths_gaussian_fit[kf][self.y] = p_gauss[2]
                # ['width_x']
                self.widths_gaussian_fit[kf][self.x] = p_gauss[3]

                self.heights_gaussian_fit[kf] = p_gauss[4]  # ['height']

                self.background_gaussian_fit[kf] = p_gauss[5]  # ['offset']

                if verbose:
                    print(f'NewSolution={self.centering_gaussian_fit[kf]}')

        elif median_crop:
            print('Setting Gaussian Centerintg Outliers to the Median')
            y_gaussball = ((self.centering_gaussian_fit.T[y] - medY)/stdY)**2
            x_gaussball = ((self.centering_gaussian_fit.T[x] - medX)/stdX)**2

            inliers = (y_gaussball+x_gaussball) <= n_sig
            medY, medX = np.nanmedian(
                self.centering_gaussian_fit[inliers], axis=0)

            self.centering_gaussian_fit.T[y][outliers] = medY
            self.centering_gaussian_fit.T[x][outliers] = medX

        try:
            self.centering_df = self.centering_df  # check if it exists
        except Exception as err:
            print(f"Option X2 Fit Failed as {err}")
            self.centering_df = pd.DataFrame()  # create it if it does not exist

        ycenter = self.centering_gaussian_fit.T[self.y]
        xcenter = self.centering_gaussian_fit.T[self.x]
        self.centering_df['gaussian_fit_ycenters'] = ycenter
        self.centering_df['gaussian_fit_xcenters'] = xcenter

        y_width = self.widths_gaussian_fit.T[self.y]
        x_width = self.widths_gaussian_fit.T[self.x]
        self.centering_df['gaussian_fit_y_widths'] = y_width
        self.centering_df['gaussian_fit_x_widths'] = x_width

        self.centering_df['gaussian_fit_heights'] = self.heights_gaussian_fit
        self.centering_df['gaussian_fit_offset'] = self.background_gaussian_fit

    def assign_gaussian_centering(self, p_gauss, xlower, kf, ylower):

        y, x = self.y, self.x
        self.centering_gaussian_fit[kf][x] = p_gauss[1] + xlower
        self.centering_gaussian_fit[kf][y] = p_gauss[2] + ylower

        self.widths_gaussian_fit[kf][x] = p_gauss[3]
        self.widths_gaussian_fit[kf][y] = p_gauss[4]

        self.heights_gaussian_fit[kf] = p_gauss[0]
        self.background_gaussian_fit[kf] = p_gauss[5]

    def mp_fit_gaussian_centering(
            self, n_sig=False, method='la', initc='fw',
            sub_array=False, print_compare=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        if np.isnan(self.image_cube).any():
            is_nan_ = np.isnan(self.image_cube)
            self.image_cube[is_nan_] = np.nanmedian(self.image_cube)

        y, x = self.y, self.x

        # yinds0, xinds0 = np.indices(self.image_cube[0].shape)

        ylower = np.int32(self.yguess - self.pix_rad)
        yupper = np.int32(self.yguess + self.pix_rad)
        xlower = np.int32(self.xguess - self.pix_rad)
        xupper = np.int32(self.xguess + self.pix_rad)

        # ylower, xlower, yupper, xupper = np.int32(
        #     [ylower, xlower, yupper, xupper])

        # yinds = yinds0[ylower:yupper, xlower:xupper]
        # xinds = xinds0[ylower:yupper, xlower:xupper]

        self.centering_gaussian_fit = np.zeros((self.image_cube.shape[0], 2))
        self.widths_gaussian_fit = np.zeros((self.image_cube.shape[0], 2))

        self.heights_gaussian_fit = np.zeros(self.image_cube.shape[0])
        # self.rotation_gaussian_fit = np.zeros(self.image_cube.shape[0])
        self.background_gaussian_fit = np.zeros(self.image_cube.shape[0])

        # Gaussian fit centering
        # This starts the multiprocessing call to arms
        # pool = Pool(self.num_cores)

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
        gaussian_centers = pool_run_func(func, zip(self.image_cube))

        # pool.close()
        # pool.join()

        print(
            'Finished with Fitting Centers. Now assigning to instance values.'
        )
        for kf, p_gauss in enumerate(gaussian_centers):
            self.assign_gaussian_centering(self, p_gauss, xlower, kf, ylower)
            """ # TODO: confirm this is correct
            self.centering_gaussian_fit[kf][self.x] = p_gauss[1] + xlower
            self.centering_gaussian_fit[kf][self.y] = p_gauss[2] + ylower

            self.widths_gaussian_fit[kf][self.x] = p_gauss[3]
            self.widths_gaussian_fit[kf][self.y] = p_gauss[4]

            self.heights_gaussian_fit[kf] = p_gauss[0]
            self.background_gaussian_fit[kf] = p_gauss[5]
            """

        ycenter = self.centering_gaussian_fit.T[y]
        xcenter = self.centering_gaussian_fit.T[x]

        y_width = self.widths_gaussian_fit.T[y]
        x_width = self.widths_gaussian_fit.T[x]
        self.centering_df = pd.DataFrame()
        self.centering_df['gaussian_fit_ycenters'] = ycenter
        self.centering_df['gaussian_fit_xcenters'] = xcenter
        self.centering_df['gaussian_mom_ycenters'] = ycenter
        self.centering_df['gaussian_mom_xcenters'] = xcenter

        self.centering_df['gaussian_fit_y_widths'] = y_width
        self.centering_df['gaussian_fit_x_widths'] = x_width

        self.centering_df['gaussian_fit_heights'] = self.heights_gaussian_fit
        self.centering_df['gaussian_fit_offset'] = self.background_gaussian_fit

        # self.centering_df['gaussian_fit_Rotation'] = self.rotation_gaussian_fit

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

        # yinds0, xinds0 = np.indices(self.image_cube[0].shape)

        ylower = np.int32(self.yguess - self.pix_rad)
        yupper = np.int32(self.yguess + self.pix_rad)
        xlower = np.int32(self.xguess - self.pix_rad)
        xupper = np.int32(self.xguess + self.pix_rad)

        # yinds = yinds0[ylower:yupper, xlower:xupper]
        # xinds = xinds0[ylower:yupper, xlower:xupper]

        nFWCParams = 2  # Xc, Yc
        self.centering_fluxweight = np.zeros((self.n_frames, nFWCParams))
        # print(self.image_cube.shape)
        progress_flux_wght_center = self.tqdm(
            range(self.n_frames),
            desc='Flux Weighted Centering',
            leave=False,
            total=self.n_frames
        )

        for kf in progress_flux_wght_center:
            subFrame_now = self.image_cube[kf][ylower:yupper, xlower:xupper]
            subFrame_now[np.isnan(subFrame_now)] = np.nanmedian(
                ~np.isnan(subFrame_now))

            self.centering_fluxweight[kf] = flux_weighted_centroid(
                self.image_cube[kf],
                self.yguess,
                self.xguess,
                b_size=7
            )
            self.centering_fluxweight[kf] = self.centering_fluxweight[kf][::-1]

        self.centering_fluxweight[:, 0] = clip_outlier(
            self.centering_fluxweight.T[0]
        )
        self.centering_fluxweight[:, 1] = clip_outlier(
            self.centering_fluxweight.T[1]
        )

        ycenter_ = self.centering_fluxweight.T[self.y]
        xcenter_ = self.centering_fluxweight.T[self.x]
        self.centering_df['fluxweighted_ycenters'] = ycenter_
        self.centering_df['fluxweighted_xcenters'] = xcenter_

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

        # yinds0, xinds0 = np.indices(self.image_cube[0].shape)

        ylower = np.int32(self.yguess - self.pix_rad)
        yupper = np.int32(self.yguess + self.pix_rad)
        xlower = np.int32(self.xguess - self.pix_rad)
        xupper = np.int32(self.xguess + self.pix_rad)

        # ylower, xlower, yupper, xupper = np.int32(
        #     [ylower, xlower, yupper, xupper])

        # yinds = yinds0[ylower:yupper, xlower:xupper]
        # xinds = xinds0[ylower:yupper, xlower:xupper]

        # nFWCParams = 2  # Xc, Yc
        # self.centering_fluxweight = np.zeros((self.n_frames, nFWCParams))

        # This starts the multiprocessing call to arms
        # pool = Pool(self.num_cores)

        func = partial(
            fit_one_center,
            n_sig=n_sig,
            method='fwc',
            ylower=ylower,
            yupper=yupper,
            xlower=xlower,
            xupper=xupper,
            b_size=7
        )

        # the order is very important
        fwc_centers = pool_run_func(func, zip(self.image_cube))

        # pool.close()
        # pool.join()

        self.centering_fluxweight = np.array(fwc_centers)

        ycenter_ = self.centering_fluxweight.T[self.y]
        xcenter_ = self.centering_fluxweight.T[self.x]
        self.centering_df['fluxweighted_ycenters'] = ycenter_
        self.centering_df['fluxweighted_xcenters'] = xcenter_

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
        raise NotImplementedError(
            'fit_least_asymmetry_centering is not longer operational'
        )
        y, x = 0, 1

        # yinds0, xinds0 = np.indices(self.image_cube[0].shape)

        ylower = np.int32(self.yguess - self.pix_rad)
        yupper = np.int32(self.yguess + self.pix_rad)
        xlower = np.int32(self.xguess - self.pix_rad)
        xupper = np.int32(self.xguess + self.pix_rad)

        # ylower, xlower, yupper, xupper = np.int32(
        #     [ylower, xlower, yupper, xupper])

        # yinds = yinds0[ylower:yupper+1, xlower:xupper+1]
        # xinds = xinds0[ylower:yupper+1, xlower:xupper+1]

        nAsymParams = 2  # Xc, Yc
        self.centering_least_asym = np.zeros((self.n_frames, nAsymParams))

        progress_frame = self.tqdm(
            range(self.n_frames),
            desc='Asym',
            leave=False,
            total=self.n_frames
        )
        for kf in progress_frame:
            """
            The following is a sequence of error handling and reattempts to
                center fit using the least_asymmetry algorithm

            The least_asymmetry algorithm returns a RunError if the center
                is not found in the image
            Our experience shows that these failures are strongly correlated
                with the existence of a cosmic ray hit nearby the PSF.

            Option 1: We run `actr` with default settings
                -- developed for Spitzer exoplanet lightcurves
            Option 2: We assume that there is a deformation in the PSF
                and square the image array (preserving signs)
            Option 3: We assume that there is a cosmic ray hit nearby
                the PSF and shrink the asym_rad by half
            Option 4: We assume that there is a deformation in the PSF AND
                (or caused by) a cosmic ray hit nearby the PSF; so we square
                the image array (preserving signs)
                AND shrink the asym_rad by half
            Option 5: We assume that there is a deformation in the PSF AND
                (or caused by) a cosmic ray hit nearby the PSF; so we square
                the image array (preserving signs) AND shrink the asym_rad by
                half BUT this time we have to get crazy and shrink the
                asym_size to 2 (reduces accuracy dramatically)
            """

            fitFailed = False  # "Benefit of the Doubt"

            # Option 1: We run `actr` with default settings
            #   that were developed for Spitzer exoplanet lightcurves
            kf, yguess, xguess = np.int32([kf, self.yguess, self.xguess])

            try:
                center_asym = actr(
                    self.image_cube[kf],
                    [yguess, xguess],
                    asym_rad=8,
                    asym_size=5,
                    maxcounts=2,
                    method='gaus',
                    half_pix=False,
                    resize=False,
                    weights=False)[0]
            except Exception as err:
                print(f"Option 1 Fit Failed as {err}")
                fitFailed = True

            # Option 2: We assume that there is a deformation in the PSF
            #   and square the image array (preserving signs)
            if fitFailed:
                try:
                    center_asym = actr(
                        np.sign(self.image_cube[kf])*self.image_cube[kf]**2,
                        [yguess, xguess],
                        asym_rad=8,
                        asym_size=5,
                        maxcounts=2,
                        method='gaus',
                        half_pix=False,
                        resize=False,
                        weights=False
                    )[0]
                    fitFailed = False
                except Exception as err:
                    print(f"Option 2 Fit Failed as {err}")

            # Option 3: We assume that there is a cosmic ray hit
            #   nearby the PSF and shrink the asym_rad by half
            if fitFailed:
                try:
                    center_asym = actr(
                        self.image_cube[kf],
                        [yguess, xguess],
                        asym_rad=4,
                        asym_size=5,
                        maxcounts=2,
                        method='gaus',
                        half_pix=False,
                        resize=False,
                        weights=False
                    )[0]
                    fitFailed = False
                except Exception as err:
                    print(f"Option 3 Fit Failed as {err}")

            """
            Option 4: We assume that there is a deformation in the PSF AND
                (or caused by) a cosmic ray hit nearby the PSF; so we square
                the image array (preserving signs) AND shrink the asym_rad by
                half
            """
            if fitFailed:
                try:
                    center_asym = actr(
                        np.sign(self.image_cube[kf])*self.image_cube[kf]**2,
                        [yguess, xguess],
                        asym_rad=4,
                        asym_size=5,
                        maxcounts=2,
                        method='gaus',
                        half_pix=False,
                        resize=False,
                        weights=False
                    )[0]
                    fitFailed = False
                except Exception as err:
                    print(f"Option 4 Fit Failed as {err}")

            # Option 5: We assume that there is a deformation in the PSF
            #   AND (or caused by) a cosmic ray hit nearby the PSF; so we
            #       square the image array (preserving signs) AND shrink
            #       the asym_rad by half
            #   BUT this time we have to get crazy and shrink the asym_size
            #       to 3 (reduces accuracy dramatically)
            if fitFailed:
                try:
                    center_asym = actr(
                        np.sign(self.image_cube[kf])*self.image_cube[kf]**2,
                        [yguess, xguess],
                        asym_rad=4,
                        asym_size=3,
                        maxcounts=2,
                        method='gaus',
                        half_pix=False,
                        resize=False,
                        weights=False
                    )[0]
                    fitFailed = False
                except Exception as err:
                    print(f"Option 5 Fit Failed as {err}")

            if fitFailed:
                # I ran out of options -- literally
                print('Least Asymmetry FAILED: and returned `RuntimeError`')

            try:
                # This will work if the fit was successful
                self.centering_least_asym[kf] = center_asym[::-1]
            except Exception as err:
                print(f"Option X1 Fit Failed as {err}")
                print('Least Asymmetry FAILED: and returned `NaN`')
                fitFailed = True

            if fitFailed:
                print(
                    f'Least Asymmetry FAILED: Setting self.centering_least_asym'
                    f'[{kf}] to Initial Guess: [{self.yguess},{self.xguess}]'
                )

                self.centering_least_asym[kf] = np.array([yguess, xguess])

        ycenter_ = self.centering_least_asym.T[self.y]
        xcenter_ = self.centering_least_asym.T[self.x]
        self.centering_df['least_asym_ycenters'] = ycenter_
        self.centering_df['least_asym_xcenters'] = xcenter_

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

        # yinds0, xinds0 = np.indices(self.image_cube[0].shape)

        ylower = np.int32(self.yguess - self.pix_rad)
        yupper = np.int32(self.yguess + self.pix_rad)
        xlower = np.int32(self.xguess - self.pix_rad)
        xupper = np.int32(self.xguess + self.pix_rad)

        # ylower, xlower, yupper, xupper = np.int32(
        #     [ylower, xlower, yupper, xupper])

        yguess, xguess = np.int32([self.yguess, self.xguess])

        # yinds = yinds0[ylower:yupper+1, xlower:xupper+1]
        # xinds = xinds0[ylower:yupper+1, xlower:xupper+1]

        # nAsymParams = 2  # Xc, Yc
        # self.centering_least_asym = np.zeros((self.n_frames, nAsymParams))
        # for kf in self.tqdm(range(self.n_frames), desc='Asym',
        #   leave = False, total=self.n_frames):
        # This starts the multiprocessing call to arms
        # pool = Pool(self.num_cores)

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

        # the order is very important
        self.centering_least_asym = pool_run_func(
            func,
            zip(
                self.image_cube,
                [[yguess, xguess]]*self.nframes
            )
        )

        # pool.close()
        # pool.join()

        self.centering_least_asym = np.array(self.centering_least_asym[0])

        ycenter_ = self.centering_least_asym.T[self.y]
        xcenter_ = self.centering_least_asym.T[self.x]
        self.centering_df['least_asym_ycenters'] = ycenter_
        self.centering_df['least_asym_xcenters'] = xcenter_

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

        print('Fit for flux weighted Centers\n')
        self.fit_flux_weighted_centering()

        print('Fit for Least Asymmetry Centers\n')
        self.fit_least_asymmetry_centering()

    def measure_effective_width_subframe(self, pix_rad=None):
        pix_rad = self.pix_rad if pix_rad is None else pix_rad
        midFrame = self.image_cube.shape[1]//2
        lower = midFrame - pix_rad
        upper = midFrame + pix_rad

        image_view = self.image_cube[:, lower:upper, lower:upper]

        image_sum_sq = image_view.sum(axis=(1, 2))**2.
        image_sq_sum = (image_view**2).sum(axis=(1, 2))
        self.effective_widths = image_sum_sq / image_sq_sum

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
            self.measure_effective_width_subframe(pix_rad=self.pix_rad)
            """
            midFrame = self.image_cube.shape[1]//2
            lower = midFrame - self.pix_rad
            upper = midFrame + self.pix_rad

            image_view = self.image_cube[:, lower:upper, lower:upper]

            image_sum_sq = image_view.sum(axis=(1, 2))**2.
            image_sq_sum = (image_view**2).sum(axis=(1, 2))
            self.effective_widths = image_sum_sq / image_sq_sum
            """
        else:
            image_sum_sq = self.image_cube.sum(axis=(1, 2))**2.
            image_sq_sum = (self.image_cube**2).sum(axis=(1, 2))
            self.effective_widths = image_sum_sq / image_sq_sum

        self.centering_df['effective_widths'] = self.effective_widths

        x_widths = self.centering_df['gaussian_fit_x_widths']
        y_widths = self.centering_df['gaussian_fit_y_widths']

        self.quadrature_widths = np.sqrt(x_widths**2 + y_widths**2)
        self.centering_df['quadrature_widths'] = self.quadrature_widths

    def measure_background_circle_masked(
            self, aper_rad=10, centering='fluxweight'):
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
            Assigning all np.zeros in the mask to NaNs because the
                `mean` and `median` functions are set to `nanmean` functions,
                which will skip all NaNs
        """

        self.background_circle_mask = np.zeros(self.n_frames)

        progress_frame = self.tqdm(
            range(self.n_frames),
            desc='CircleBG',
            leave=False,
            total=self.n_frames
        )
        for kf in progress_frame:
            aperture = create_aper_mask(
                centering=self.centering_fluxweight[kf],
                aper_rad=aper_rad,
                image_shape=self.image_cube[0].shape,
                method='exact'
            )

            background_mask = ~aperture

            """
            aperture = CircularAperture(self.centering_fluxweight[kf], aper_rad)

            # list of Aperture_mask objects (one for each position)
            aper_mask = aperture.to_mask(method='exact')

            if isinstance(aper_mask, (list, tuple, np.array)):
                aper_mask = aper_mask[0]

            # background_mask = abs(aperture.get_fractions(
            #   np.ones(self.image_cube[0].shape))-1)
            # background_mask = ~aperture
            # background_mask = ~aper_mask.to_image(
            #     self.image_cube[0].shape
            # ).astype(bool)

            # background_mask = ~background_mask  # [background_mask == 0] = False
            """
            self.background_circle_mask[kf] = self.metric(
                self.image_cube[kf][background_mask]
            )

        self.background_df['Circle_mask'] = self.background_circle_mask.copy()

    def mp_measure_background_circle_masked(
            self, aper_rad=10, centering='Gauss'):
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
            Assigning all np.zeros in the mask to NaNs because the
                `mean` and `median` functions are set to `nanmean` functions,
                which will skip all NaNs
        """

        if centering == 'Gauss':
            centers = self.centering_gaussian_fit
        elif centering == 'fluxweight':
            centers = self.centering_fluxweight

        # This starts the multiprocessing call to arms
        # pool = Pool(self.num_cores)

        func = partial(
            measure_one_circle_bg,
            aper_rad=aper_rad,
            metric=self.metric,
            aper_method='exact'
        )

        self.background_circle_mask = pool_run_func(
            func, zip(self.image_cube, centers)
        )

        # pool.close()
        # pool.join()

        self.background_circle_mask = np.array(self.background_circle_mask)
        self.background_df['Circle_mask'] = self.background_circle_mask.copy()

    def measure_background_annular_mask(self, inner_rad=8, outer_rad=13):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        self.background_annulus = np.zeros(self.n_frames)

        progress_frame = self.tqdm(
            range(self.n_frames),
            desc='Annular BG',
            leave=False,
            total=self.n_frames
        )
        for kf in progress_frame:
            inner_aper_mask = create_aper_mask(
                centering=self.centering_fluxweight[kf],
                aper_rad=inner_rad,
                image_shape=self.image_cube[0].shape,
                method='exact'
            )

            outer_aper_mask = create_aper_mask(
                centering=self.centering_fluxweight[kf],
                aper_rad=outer_rad,
                image_shape=self.image_cube[0].shape,
                method='exact'
            )
            """
            innerAperture = CircularAperture(
                self.centering_fluxweight[kf],
                inner_rad
            )

            inner_aper_mask = innerAperture.to_mask(method='exact')

            if isinstance(inner_aper_mask, (list, tuple, np.array)):
                inner_aper_mask = inner_aper_mask[0]

            inner_aper_mask = inner_aper_mask.to_image(
                self.image_cube[0].shape
            ).astype(bool)
            """
            """
            outerAperture = CircularAperture(
                self.centering_fluxweight[kf],
                outer_rad
            )

            outer_aper_mask = outerAperture.to_mask(method='exact')

            if isinstance(outer_aper_mask, (list, tuple, np.array)):
                outer_aper_mask = outer_aper_mask[0]

            outer_aper_mask = outer_aper_mask.to_image(
                self.image_cube[0].shape
            ).astype(bool)
            """
            background_mask = (~inner_aper_mask)*outer_aper_mask

            self.background_annulus[kf] = self.metric(
                self.image_cube[kf][background_mask]
            )

        self.background_df['annular_mask'] = self.background_annulus.copy()

    def mp_measure_background_annular_mask(
            self, inner_rad=8, outer_rad=13, method='exact', centering='Gauss'):
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
            centers = self.centering_gaussian_fit
        elif centering == 'fluxweight':
            centers = self.centering_fluxweight

        # This starts the multiprocessing call to arms
        # pool = Pool(self.num_cores)

        func = partial(
            measure_one_annular_bg,
            inner_rad=inner_rad,
            outer_rad=outer_rad,
            metric=self.metric,
            aper_method='exact'
        )

        # the order is very important
        self.background_annulus = pool_run_func(
            func, zip(self.image_cube, centers)
        )

        # pool.close()
        # pool.join()

        self.background_annulus = np.array(self.background_annulus)
        self.background_df['annular_mask'] = self.background_annulus.copy()

    def measure_background_median_masked(self, aper_rad=10, n_sig=5):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        self.background_median_mask = np.zeros(self.n_frames)

        # the order is very important
        progress_frames = self.tqdm(
            range(self.n_frames),
            desc='Median Masked BG',
            leave=False,
            total=self.n_frames
        )
        for kf in progress_frames:
            aperture = create_aper_mask(
                centering=self.centering_fluxweight[kf],
                aper_rad=aper_rad,
                image_shape=self.image_cube[0].shape,
                method='exact'
            )
            """
            aperture = CircularAperture(self.centering_fluxweight[kf], aper_rad)
            aperture = aperture.to_mask(method='exact')

            if isinstance(aperture, (list, tuple, np.ndarray)):
                aperture = aperture[0]

            aperture = aperture.to_image(self.image_cube[0].shape).astype(bool)
            """
            background_mask = ~aperture

            medFrame = np.nanmedian(self.image_cube[kf][background_mask])
            # scale.mad(self.image_cube[kf][background_mask])
            madFrame = np.nanstd(self.image_cube[kf][background_mask])

            median_mask = abs(self.image_cube[kf] - medFrame) < n_sig*madFrame

            maskComb = median_mask*background_mask
            # maskComb[maskComb == 0] = False

            self.background_median_mask[kf] = np.nanmedian(
                self.image_cube[kf][maskComb]
            )

        self.background_df['median_mask'] = self.background_median_mask

    def mp_measure_background_median_masked(
            self, aper_rad=10, n_sig=5, centering='Gauss'):
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
            centers = self.centering_gaussian_fit
        elif centering == 'fluxweight':
            centers = self.centering_fluxweight

        # This starts the multiprocessing call to arms
        # pool = Pool(self.num_cores)

        func = partial(
            measure_one_median_bg,
            aper_rad=aper_rad,
            aper_method='exact',
            metric=self.metric,
            n_sig=n_sig
        )

        self.background_median_mask = pool_run_func(
            func, zip(self.image_cube, centers)
        )

        # pool.close()
        # pool.join()

        self.background_median_mask = np.array(self.background_median_mask)
        self.background_df['median_mask'] = self.background_median_mask

    def measure_background_kde_mode(self, aper_rad=10):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.
        """

        self.background_kde_univ = np.zeros(self.n_frames)

        progress_frames = self.tqdm(
            range(self.n_frames),
            desc='KDE Background',
            leave=False,
            total=self.n_frames
        )
        for kf in progress_frames:
            aperture = create_aper_mask(
                centering=self.centering_fluxweight[kf],
                aper_rad=aper_rad,
                image_shape=self.image_cube[0].shape,
                method='exact'
            )
            """
            aperture = CircularAperture(self.centering_fluxweight[kf], aper_rad)
            aperture = aperture.to_mask(method='exact')

            if isinstance(aperture, (list, tuple, np.ndarray)):
                aperture = aperture[0]

            aperture = aperture.to_image(self.image_cube[0].shape).astype(bool)
            """
            background_mask = ~aperture

            kdeFrame = kde.KDEUnivariate(
                self.image_cube[kf][background_mask].ravel()
            )
            kdeFrame.fit()

            density_argmax = kdeFrame.density.argmax()
            self.background_kde_univ[kf] = kdeFrame.support[density_argmax]

        self.background_df['kde_univ_mask'] = self.background_kde_univ

    def mp_measure_background_kde_mode(self, aper_rad=10, centering='Gauss'):
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
            centers = self.centering_gaussian_fit
        elif centering == 'fluxweight':
            centers = self.centering_fluxweight

        self.background_kde_univ = np.zeros(self.n_frames)

        # This starts the multiprocessing call to arms
        # pool = Pool(self.num_cores)

        func = partial(
            measure_one_kde_bg,
            aper_rad=aper_rad,
            aper_method='exact',
            metric=self.metric
        )

        self.background_kde_univ = pool_run_func(
            func,
            zip(self.image_cube, centers)
        )  # the order is very important

        # pool.close()
        # pool.join()

        self.background_kde_univ = np.array(self.background_kde_univ)
        self.background_df['kde_univ_mask_mp'] = self.background_kde_univ

    def measure_all_background(self, aper_rad=10, n_sig=5):
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
        p_outer = 0.3  # Percent Outer = +30%

        print('Measuring Background Using Circle Mask with Multiprocessing')
        self.mp_measure_background_circle_masked(aper_rad=aper_rad)

        print('Measuring Background Using Annular Mask with Multiprocessing')
        self.mp_measure_background_annular_mask(
            inner_rad=(1-p_inner)*aper_rad, outer_rad=(1+p_outer)*aper_rad)

        print('Measuring Background Using Median Mask with Multiprocessing')
        self.mp_measure_background_median_masked(
            aper_rad=aper_rad, n_sig=n_sig)

        print('Measuring Background Using KDE Mode with Multiprocessing')
        self.mp_measure_background_kde_mode(aper_rad=aper_rad)

    def helper_flux_over_time(self, aper_rad, flux_key, centers, backgrounds):
        flux_tso_now = np.zeros(self.n_frames)
        noise_tso_now = np.zeros(self.n_frames)

        progress_frames = self.tqdm(
            range(self.n_frames),
            desc='flux',
            leave=False,
            total=self.n_frames
        )

        for kf in progress_frames:
            frame_now = np.copy(self.image_cube[kf]) - backgrounds[kf]
            frame_now[np.isnan(frame_now)] = np.nanmedian(frame_now)

            noise_now = np.copy(self.noise_cube[kf])**2.
            noise_now[np.isnan(noise_now)] = np.nanmedian(noise_now)

            xcenter_ = centers[kf][self.x]
            ycenter_ = centers[kf][self.y]
            aperture = CircularAperture([xcenter_, ycenter_], r=aper_rad)

            flux_tso_now[kf] = aperture_photometry(frame_now, aperture)
            flux_tso_now[kf] = flux_tso_now[kf]['aperture_sum']

            noise_tso_now[kf] = aperture_photometry(noise_now, aperture)
            noise_tso_now[kf] = np.sqrt(noise_tso_now[kf]['aperture_sum'])

        self.flux_tso_df[flux_key] = flux_tso_now
        self.noise_tso_df[flux_key] = noise_tso_now

    def compute_flux_over_time(
            self, aper_rad=None, centering='gaussian_fit',
            background='annular_mask', use_the_force=False):
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
            raise KeyError(
                f"`background` must be in {self.background_df.columns}",
            )

        centering_options = [
            'gaussian_fit',
            'gaussian_mom',
            'fluxweighted',
            'least_asym'
        ]
        if centering not in centering_options:
            raise ValueError(
                "`centering` must be either 'gaussian_fit', "
                "'gaussian_mom', 'fluxweighted', or 'least_asym'"
            )

        if aper_rad is None:
            static_rad = 70 if 'wlp' in self.fits_names[0].lower() else 3
            """
            if 'wlp' in self.fits_names[0].lower():
                static_rad = 70
            else:
                static_rad = 3
            """

        ycenters = self.centering_df[centering + '_ycenters']
        xcenters = self.centering_df[centering + '_xcenters']
        centering_Use = np.transpose([ycenters, xcenters])

        background_Use = self.background_df[background]

        flux_key_now = f"{centering}_{background}_rad_{aper_rad}"

        if flux_key_now not in self.flux_tso_df.keys() or use_the_force:
            self.helper_flux_over_time(
                aper_rad=aper_rad,
                flux_key=flux_key_now,
                centers=centering_Use,
                backgrounds=background_Use
            )
            """
            flux_tso_now = np.zeros(self.n_frames)
            noise_tso_now = np.zeros(self.n_frames)

            progress_frames = self.tqdm(
                range(self.n_frames),
                desc='flux',
                leave=False,
                total=self.n_frames
            )
            for kf in progress_frames:
                frame_now = np.copy(self.image_cube[kf]) - background_Use[kf]
                frame_now[np.isnan(frame_now)] = np.nanmedian(frame_now)

                noise_now = np.copy(self.noise_cube[kf])**2.
                noise_now[np.isnan(noise_now)] = np.nanmedian(noise_now)

                xcenter_ = centering_Use[kf][self.x]
                ycenter_ = centering_Use[kf][self.y]
                aperture = CircularAperture([xcenter_, ycenter_], r=aper_rad)

                flux_tso_now[kf] = aperture_photometry(
                    frame_now, aperture
                )['aperture_sum']
                noise_tso_now[kf] = np.sqrt(
                    aperture_photometry(
                        noise_now,
                        aperture
                    )['aperture_sum']
                )

            self.flux_tso_df[flux_key_now] = flux_tso_now
            self.noise_tso_df[flux_key_now] = noise_tso_now
            """
        else:
            print(
                f'{flux_key_now} exists: '
                'if you want to overwrite, then you `use_the_force=True`'
            )

    def compute_flux_over_time_over_aper_rad(
            self, aper_rads=None, centering_choices=None,
            background_choices=None, use_the_force=False, verbose=False):
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
        for bg_now in background_choices:
            for ctr_now in centering_choices:
                for aper_rad in aper_rads:
                    if verbose:
                        print(
                            f'Working on Background {bg_now} '
                            f'with Centering {ctr_now} and Aper_rad {aper_rad}',
                            end=" ",
                        )

                    self.compute_flux_over_time(
                        aper_rad=aper_rad,
                        centering=ctr_now,
                        background=bg_now,
                        use_the_force=use_the_force
                    )

                    flux_key_now = f"{ctr_now}_{bg_now}_rad_{aper_rad}"

                    if verbose:
                        flux_now = self.flux_tso_df[flux_key_now]
                        print(
                            np.nanstd(flux_now / np.nanmedian(flux_now)) * ppm
                        )

        print('Operation took: ', time()-start)

    def mp_compute_flux_over_time(
            self, aper_rad=3.0, centering='gaussian_fit',
            background='annular_mask', use_the_force=False):
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
            raise KeyError(
                f"`background` must be in {self.background_df.columns}"
            )

        centering_options = [
            'gaussian_fit', 'gaussian_mom', 'fluxweighted', 'least_asym'
        ]
        if centering not in centering_options:
            raise ValueError(
                "`centering` must be either 'gaussian_fit', 'gaussian_mom', "
                "'fluxweighted', or 'least_asym'"
            )

        ycenter_ = self.centering_df[centering + '_ycenters']
        xcenter_ = self.centering_df[centering + '_xcenters']
        centering_Use = np.transpose([ycenter_, xcenter_])

        background_Use = self.background_df[background]

        flux_key_now = f"{centering}_{background}_rad_{aper_rad}"

        if flux_key_now not in self.flux_tso_df.keys() or use_the_force:
            # for kf in self.tqdm(range(self.n_frames), desc='flux', leave = False, total=self.n_frames):

            # pool = Pool(self.num_cores)

            func = partial(compute_flux_one_frame, aper_rad=aper_rad)

            flux_now = pool_run_func(
                func, zip(self.image_cube, centering_Use, background_Use)
            )

            # pool.close()
            # pool.join()

            # flux_now[~np.isfinite(flux_now)] = np.nanmedian(flux_now[np.isfinite(flux_now)])
            # flux_now[flux_now < 0] = np.nanmedian(flux_now[flux_now > 0])

            self.flux_tso_df[flux_key_now] = flux_now
            self.noise_tso_df[flux_key_now] = np.sqrt(flux_now)

        else:
            print(
                f'{flux_key_now} exists: '
                'if you want to overwrite, then you `use_the_force=True`'
            )

    def mp_compute_flux_over_time_var_rad(self, static_rad, var_rad=None, centering='gaussian_fit', background='annular_mask', use_the_force=False):
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
            raise KeyError(
                f"`background` must be in {self.background_df.columns}"
            )

        centering_options = [
            'gaussian_fit',
            'gaussian_mom',
            'fluxweighted',
            'least_asym'
        ]
        if centering not in centering_options:
            raise KeyError(
                "`centering` must be either 'gaussian_fit', 'gaussian_mom', "
                "'fluxweighted', or 'least_asym'"
            )

        ycenter_ = self.centering_df[centering + '_ycenters']
        xcenter_ = self.centering_df[centering + '_xcenters']
        centering_Use = np.transpose([ycenter_, xcenter_])

        background_Use = self.background_df[background]

        flux_key_now = f'{centering}_{background}_rad_{static_rad}_{var_rad}'

        assert (isinstance(static_rad, (float, int))), (
            'static_rad must be either a float or an integer'
        )

        if var_rad is None or var_rad == 0.0:
            aper_rads = [static_rad] * self.n_frames
        else:
            med_quad_rad_dist = np.nanmedian(self.quadrature_widths)
            quad_rad_dist = self.quadrature_widths.copy() - med_quad_rad_dist
            quad_rad_dist = clip_outlier(quad_rad_dist, n_sig=5)
            aper_rads = static_rad + var_rad*quad_rad_dist

        if flux_key_now not in self.flux_tso_df.keys() or use_the_force:
            # for kf in self.tqdm(range(self.n_frames), desc='flux', leave = False, total=self.n_frames):

            # pool = Pool(self.num_cores)

            # func = partial(compute_flux_one_frame)

            flux_now = pool_run_func(
                compute_flux_one_frame,
                zip(self.image_cube, centering_Use, background_Use, aper_rads)
            )

            # pool.close()
            # pool.join()

            # flux_now[~np.isfinite(flux_now)] = np.nanmedian(flux_now[np.isfinite(flux_now)])
            # flux_now[flux_now < 0] = np.nanmedian(flux_now[flux_now > 0])

            self.flux_tso_df[flux_key_now] = flux_now
            self.noise_tso_df[flux_key_now] = np.sqrt(flux_now)

        else:
            print(
                f'{flux_key_now} exists: '
                'if you want to overwrite, then you `use_the_force=True`'
            )

    def mp_compute_flux_over_time_beta_rad(
            self, centering='gaussian_fit', background='annular_mask',
            useQuad=False, use_the_force=False):
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
            raise Exception(
                f"`background` must be in {self.background_df.columns}"
            )

        centering_options = [
            'gaussian_fit', 'gaussian_mom', 'fluxweighted', 'least_asym'
        ]
        if centering not in centering_options:
            raise ValueError(
                "`centering` must be either 'gaussian_fit', 'gaussian_mom', "
                "'fluxweighted', or 'least_asym'"
            )

        ycenter_ = self.centering_df[centering + '_ycenters']
        xcenter_ = self.centering_df[centering + '_xcenters']
        centering_Use = np.transpose([ycenter_, xcenter_])

        background_Use = self.background_df[background]

        flux_key_now = centering + '_' + background+'_' + 'rad'

        flux_key_now = flux_key_now + \
            '_quad_rad_0.0_0.0' if useQuad else flux_key_now + '_beta_rad_0.0_0.0'

        if flux_key_now not in self.flux_tso_df.keys() or use_the_force:
            # for kf in self.tqdm(range(self.n_frames), desc='flux', leave = False, total=self.n_frames):
            sig2FW = 2*np.sqrt(2*np.log(2))

            aper_rads = np.sqrt(self.effective_widths)

            if useQuad:
                aper_rads = sig2FW * self.quadrature_widths

            # pool = Pool(self.num_cores)

            # func = partial(compute_flux_one_frame)

            flux_now = pool_run_func(
                compute_flux_one_frame,
                zip(self.image_cube, centering_Use, background_Use, aper_rads)
            )

            # pool.close()
            # pool.join()

            # flux_now[~np.isfinite(flux_now)] = np.nanmedian(flux_now[np.isfinite(flux_now)])
            # flux_now[flux_now < 0] = np.nanmedian(flux_now[flux_now > 0])

            self.flux_tso_df[flux_key_now] = flux_now
            self.noise_tso_df[flux_key_now] = np.sqrt(flux_now)

        else:
            print(
                f'{flux_key_now} exists: '
                'if you want to overwrite, then you `use_the_force=True`'
            )

    def extract_pld_components(
            self, ycenter=None, xcenter=None, nCols=3, nRows=3, order=1):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        n_pld_comp = nCols*nRows

        # nCols = nCols // 2 # User input assumes matrix structure, which starts at y- and x-center
        # nRows = nRows // 2 #   and moves +\- nRows/2 and nCols/2, respectively

        img_shape = self.image_cube.shape[1:]
        # nominally 15
        ycenter = int(ycenter) if ycenter is not None else img_shape[0]//2-1
        """
        if ycenter is not None:
            ycenter = int(ycenter)
        else:
            ycenter = img_shape[0] // 2 - 1
        """

        # nominally 15
        xcenter = int(xcenter) if xcenter is not None else img_shape[1]//2-1
        """
        if xcenter is not None:
            xcenter = int(xcenter)
        else:
            xcenter = img_shape[1] // 2 - 1
        """
        ylower = ycenter - nRows // 2     # nominally 14
        yupper = ycenter + nRows // 2 + 1  # nominally 17 (to include 16)
        xlower = xcenter - nCols // 2     # nominally 14
        xupper = xcenter + nCols // 2 + 1  # nominally 17 (to include 16)

        image_sub_cube = self.image_cube[:, ylower:yupper, xlower:xupper]
        new_shape = self.image_cube.shape[0], n_pld_comp
        pld_comps_local = image_sub_cube.reshape(new_shape).T

        pld_norm = np.sum(pld_comps_local, axis=0)
        pld_comps_local = pld_comps_local / pld_norm

        self.pld_components = pld_comps_local
        self.pld_norm = pld_norm

        if order > 1:
            for ord_ in range(2, order+1):
                self.pld_components = np.vstack(
                    [self.pld_components, pld_comps_local**ord_]
                )

    def dbscan_flux_all(
            self, centering='gaussian', dbs_clean=0, use_the_force=False):
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

        if centering != 'gaussian':
            # ycenters = self.centering_gaussian_fit.T[y]
            # xcenters = self.centering_gaussian_fit.T[x]
            print('Only Gaussian Centering is supported at the moment')
            print('Continuting with Gaussian Centering')
        # else:
        #     print('Only Gaussian Centering is supported at the moment')
        #     print('Continuting with Gaussian Centering')
        #     ycenters = self.centering_gaussian_fit.T[y]
        #     xcenters = self.centering_gaussian_fit.T[x]

        ycenters = self.centering_gaussian_fit.T[y]
        xcenters = self.centering_gaussian_fit.T[x]

        if not hasattr(self, 'inliers_phots'):
            self.inliers_phots = {}

        for flux_key_now in self.flux_tso_df.keys():

            phots = self.flux_tso_df[flux_key_now]

            if flux_key_now not in self.inliers_phots.keys() or use_the_force:
                self.inliers_phots[flux_key_now] = dbscan_segmented_flux(
                    phots, ycenters, xcenters, dbs_clean=dbs_clean)
            else:
                print(
                    f'{flux_key_now} exists: '
                    'if you want to overwrite, then you `use_the_force=True`'
                )

    def mp_dbscan_flux_all(self, centering='gaussian', dbs_clean=0):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        if centering != 'gaussian':
            print('Only Gaussian Centering is supported at the moment')
            print('Continuting with Gaussian Centering')
        #     ycenters = self.centering_gaussian_fit.T[y]
        #     xcenters = self.centering_gaussian_fit.T[x]
        # else:
        #     print('Only Gaussian Centering is supported at the moment')
        #     print('Continuting with Gaussian Centering')
        #     ycenters = self.centering_gaussian_fit.T[y]
        #     xcenters = self.centering_gaussian_fit.T[x]

        ycenters = self.centering_gaussian_fit.T[self.y]
        xcenters = self.centering_gaussian_fit.T[self.x]

        if not hasattr(self, 'inliers_phots'):
            self.inliers_phots = {}

        # This starts the multiprocessing call to arms
        # pool = Pool(self.num_cores)

        func = partial(
            dbscan_flux,
            ycenters=ycenters,
            xcenters=xcenters,
            dbs_clean=dbs_clean
        )

        # the order is very important
        inliers_mp = pool_run_func(func, zip(self.flux_tso_df.values.T))

        # pool.close()
        # pool.join()

        for k_mp, flux_key_now in enumerate(self.flux_tso_df.keys()):
            self.inliers_phots[flux_key_now] = inliers_mp[k_mp]

    def mp_dbscan_pld_all(self, dbs_clean=0):
        raise NotImplementedError(
            'This Function is Not Working; please use lame_`dbscan_pld_all`'
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

        if not hasattr(self, "inliers_pld"):
            self.inliers_pld = np.ones(self.pld_components.shape, dtype=bool)

        # This starts the multiprocessing call to arms
        # pool = Pool(self.num_cores)

        func = partial(dbscan_segmented_flux, dbs_clean=dbs_clean)

        # the order is very important
        inliers_mp = pool_run_func(func, zip(self.pld_components.T))

        # pool.close()
        # pool.join()

        for k_mp, inlier in enumerate(inliers_mp):
            self.inliers_pld[k_mp] = inlier

    def mp_dbscan_pld_all(self, dbs_clean=0):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """

        if not hasattr(self, "inliers_pld"):
            self.inliers_pld = np.ones(self.pld_components.shape, dtype=bool)

        for kpld, pldnow in enumerate(self.pld_components):
            self.inliers_pld[kpld] = dbscan_pld(pldnow, dbs_clean=dbs_clean)


def load_wanderer_instance_from_file(
        planet_name=None, channel=None, aor_dir=None, shell=True,
        check_defaults=True):

    if shell:
        data_config = command_line_inputs(check_defaults=check_defaults)
    else:
        data_config = WandererCLI()
        # data_config = default_inputs(check_defaults=check_defaults)

    # data_config.planet_name = 'hatp26b'
    # data_config.channel = 'ch2'
    # data_config.aor_dir = 'r42621184'
    # print(planet_name, channel, aor_dir)

    data_config.planet_name = planet_name or data_config.planet_name
    data_config.channel = channel or data_config.channel
    data_config.aor_dir = aor_dir or data_config.aor_dir

    # planets_dir = data_config.planets_dir
    # load_sub_dir = data_config.save_sub_dir
    # data_sub_dir = data_config.data_sub_dir
    # data_tail_dir = data_config.data_tail_dir
    # fits_format = data_config.fits_format
    # unc_format = data_config.unc_format
    # load_file_type = data_config.save_file_type
    # method = data_config.method
    # telescope = data_config.telescope
    # # output_units = data_config.output_units
    data_dir = data_config.data_dir or 'aordirs'
    # num_cores = data_config.num_cores
    # verbose = data_config.verbose

    startFull = time()
    print(f'Found {data_config.num_cores} cores to process')

    data_config.data_dir = data_config.data_dir or 'aordirs'
    fits_file_dir, fitsFilenames, uncsFilenames = grab_dir_and_filenames(
        data_config=data_config,
        fits_format=data_config.fits_format,
        unc_format=data_config.unc_format
    )

    loadfiledir = os.path.join(
        data_config.planets_dir,
        data_config.save_sub_dir,
        data_config.channel,
        data_config.aor_dir
    )

    header_test = fits.getheader(fitsFilenames[0])

    print(
        f'\n\nAORLABEL:\t{header_test["AORLABEL"]}'+'\n'
        f'Num Fits Files:\t{len(fitsFilenames)}'+'\n'
        f'Num Unc Files:\t{len(uncsFilenames)}\n\n'
    )

    if data_config.verbose:
        print(fitsFilenames)
    if data_config.verbose:
        print(uncsFilenames)

    # Necessary Constants Spitzer
    ppm = 1e6
    y, x = 0, 1

    yguess, xguess = 15., 15.   # Specific to Spitzer circa 2010 and beyond
    # Specific to Spitzer Basic Calibrated Data
    filetype = f'{data_config.fits_format}.fits'

    print('Initialize an instance of `Wanderer`\n')
    wanderer = Wanderer(
        fits_file_dir=fits_file_dir,
        filetype=filetype,
        telescope=data_config.telescope,
        yguess=yguess,
        xguess=xguess,
        method=data_config.method,
        num_cores=data_config.num_cores
    )

    wanderer.AOR = data_config.aor_dir
    wanderer.planet_name = data_config.planet_name
    wanderer.channel = data_config.channel

    print(
        'Loading `wanderer` to a set of pickles for various '
        'Image Cubes and the Storage Dictionary'
    )

    load_name_header = f'{data_config.planet_name}_{data_config.aor_dir}_median'

    path_to_files = os.path.join(
        data_config.planets_dir,
        # data_config.planet_name,
        data_config.save_sub_dir
    )
    if not os.path.exists(path_to_files):
        raise ValueError()

    # if not os.path.exists(loadfiledir):
    #     print(f'Creating {loadfiledir}')
    #     os.mkdir(loadfiledir)

    load_path = os.path.join(
        loadfiledir,
        f'{load_name_header}_STRUCTURE{data_config.save_file_type}'
    )

    print()
    print(f'Loading to {load_path}')
    print()

    wanderer.load_data_from_save_files(
        savefiledir=loadfiledir,
        save_name_header=load_name_header,
        save_file_type=data_config.save_file_type
    )

    print(f'Entire Pipeline took {time() - startFull} seconds')

    return wanderer, data_config
