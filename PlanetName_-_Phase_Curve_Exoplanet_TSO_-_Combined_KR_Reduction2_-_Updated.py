# import batman
import corner
import sys
# import datetime
# import emcee3
import emcee
# import jdcal
from os import environ, path
import pywt
# import spiderman as sp

from batman    import TransitModel, TransitParams
from spiderman import ModelParams as sp_ModelParams

from functools import partial
from pylab     import *
from pandas    import DataFrame

from scipy.spatial           import cKDTree
from sklearn.externals       import joblib
from sklearn.model_selection import train_test_split
from sklearn.decomposition   import PCA, FastICA

from astropy.io import fits
from glob import glob


from tqdm import tqdm, tqdm_notebook

# from scipy.optimize    import leastsq, minimize
from scipy.interpolate import CubicSpline
# from scipy.signal      import medfilt
# from scipy.stats       import binned_statistic

from lmfit import Parameters, Minimizer, report_errors

from statsmodels.robust.scale import mad
from sklearn.preprocessing    import scale

from time import time

# from sys import argv

# from photutils import CircularAperture, CircularAnnulus, EllipticalAperture
# from photutils import aperture_photometry

# from statsmodels.robust import scale
from datetime import datetime

from exoparams import PlanetParams

from numpy import cos, pi, abs

# from sklearn.svm import SVR
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

from multiprocessing import cpu_count, Pool

from astropy.constants import R_sun, au

from ExoplanetTSO_Auxiliary import wanderer

from spitzer_helper_functions import clipOutliers, bin_array, b2inc, deltaphase_eclipse
from spitzer_helper_functions import extract_PLD_components, de_median, savePickleOut, spiderman_lmfit_model
from spitzer_helper_functions import phase_cos_curve, phase_cos_sin_curve, inc2b, batman_lmfit_model
# from spitzer_helper_functions import pixel_level_decorrelation_instrument_profile

stdScaler = StandardScaler()
stime = time()

from matplotlib import rcParams, pyplot as plt
rcParams["savefig.dpi"] = 200
rcParams["figure.dpi"] = 200

print("\n**DONE LOADING LIBRARIES AND DEFINING FUNCTIONS**\n")

# # List of Function Definitions
def batman_lmfit_model(period, tCenter, inc, aprs, edepth, tdepth, ecc, omega, times, u1=None, u2=None,
                       ldtype = 'uniform', transittype="primary", bm_params=None):
    
    if tdepth is not 0.0 or edepth is not 0.0:
        if bm_params is None:
            bm_params = TransitParams() # object to store transit parameters
        
        bm_params.per       = period  # orbital period
        bm_params.t0        = tCenter # time of inferior conjunction
        bm_params.a         = aprs    # semi-major axis (in units of stellar radii)
        bm_params.fp        = edepth  # f
        bm_params.tdepth    = tdepth  # from Fraine et al. 2014s
        bm_params.rp        = sqrt(tdepth) # planet radius (in units of stellar radii)
        bm_params.ecc       = ecc     # eccentricity
        bm_params.w         = omega   # longitude of periastron (in degrees)
        bm_params.inc       = inc     # orbital inclination (in degrees)
        bm_params.limb_dark = ldtype  # limb darkening model # NEED TO FIX THIS
        bm_params.u         = []      # limb darkening coefficients # NEED TO FIX THIS
        
        if u1 is not None and ldtype is not 'uniform':
            bm_params.u.append(u1)
        elif u1 is not None and ldtype is 'uniform':
            raise ValueError('If you set `u1`, you must also set `ldtype` to either linear or quadratic')
        if u2 is not None and ldtype is 'quadratic':
            bm_params.u.append(u2)
        elif u2 is not None and ldtype is not 'quadratic':
            raise ValueError('If you set `u2`, you must also set `ldtype` quadratic')
        
        bm_params.delta_phase = deltaphase_eclipse(bm_params.ecc, bm_params.w) if bm_params.ecc is not 0.0 else 0.5
        bm_params.t_secondary = bm_params.t0 + bm_params.per*bm_params.delta_phase
        
        m_eclipse = TransitModel(bm_params, times, transittype=transittype).light_curve(bm_params)
    else:
        return ones(times.size)
    
    return m_eclipse

def spiderman_lmfit_model(period, tCenter, inc, aprs, tdepth, ecc, omega, times, u1, u2, xi, T_night, delta_T, 
                         nspiders=1000, nCores = cpu_count(),
                         spider_params=None, n_layers=20, stellar_radius = 1.0, stellar_temperature=5500.,
                         wave_low = 4.0, wave_hi = 5.0, brightness_model='zhang', if_eclipse=False):
    
    if tdepth is not 0.0:
        # if spider_params is None:
        #     spider_params                = sp_ModelParams(brightness_model=brightness_model)
        #     spider_params.n_layers       = n_layers
        #     spider_params.stellar_radius = stellar_radius
        #     spider_params.T_s            = stellar_temperature
        #     spider_params.l1             = wave_low
        #     spider_params.l2             = wave_hi
        #     spider_params.eclipse        = if_eclipse
        
        a_au  = aprs * spider_params.stellar_radius * R_sun.value / au.value
        # inc   = b2inc(bImpact, aprs, ecc, omega)*180/pi
        
        spider_params.t0      = tCenter      # Central time of PRIMARY transit [days]
        spider_params.per     = period       # Period [days]
        spider_params.a_abs   = a_au         # The absolute value of the semi-major axis [AU]
        spider_params.inc     = inc # orbital inclination (in degrees)
        spider_params.ecc     = ecc          # Eccentricity
        spider_params.w       = omega        # Argument of periastron
        spider_params.rp      = sqrt(tdepth) # planet radius (in units of stellar radii)
        spider_params.a       = aprs         # Semi-major axis scaled by stellar radius
        spider_params.p_u1    = u1           # Planetary limb darkening parameter
        spider_params.p_u2    = u2           # Planetary limb darkening parameter
        
        spider_params.xi      = xi           # Ratio of radiative to advective timescale             
        spider_params.T_n     = T_night      # Temperature of nightside
        spider_params.delta_T = delta_T      # Day-night temperature contrast
        
        nskips            = times.size // nspiders
        times_subsampled  = times[::nskips]
        spider_lightcurve = spider_params.lightcurve(times_subsampled)
        spider_lightcurve = CubicSpline(times_subsampled, spider_lightcurve)
        
        return spider_lightcurve(times)
    else:
        return ones(times.size)
    
    raise Exception('Something Weird happened')

def find_qhull_one_point(point, x0, y0, np0, inds):
    # ind         = kdtree.query(kdtree.data[point],sm_num+1)[1][1:]
    ind = inds[point]
    dx  = x0[ind] - x0[point]
    dy  = y0[ind] - y0[point]
    
    if np0.sum() != 0.0:
        dnp         = np0[ind] - np0[point]
    
    sigx  = np.std(dx )
    sigy  = np.std(dy )
    
    if dnp.sum() != 0.0:
        signp   = np.std(dnp)
        gw_temp = np.exp(-dx**2./(2.0*sigx**2.)) * np.exp(-dy**2./(2.*sigy**2.)) * np.exp(-dnp**2./(2.*signp**2.))
    else:
        gw_temp = np.exp(-dx**2./(2.0*sigx**2.)) * np.exp(-dy**2./(2.*sigy**2.))
    
    gw_sum  = gw_temp.sum()
    
    return gw_temp / gw_sum

def mp_find_nbr_qhull(xpos, ypos, npix = None, inds = None, n_nbr = 100, returnInds=False,
                      a = 1.0, b = 0.7, c = 1.0, expansion = 1000., nCores=cpu_count()):
    '''
        Python Implimentation of N. Lewis method, described in Lewis etal 2012, Knutson etal 2012, Fraine etal 2013
        
        Taken from N. Lewis IDL code:
            
            Construct a 3D surface (or 2D if only using x and y) from the data
            using the qhull.pro routine.  Save the connectivity information for
            each data point that was used to construct the Delaunay triangles (DT)
            that form the grid.  The connectivity information allows us to only
            deal with a sub set of data points in determining nearest neighbors
            that are either directly connected to the point of interest or
            connected through a neighboring point
        
        Python Version:
            J. Fraine    first edition, direct translation from IDL 12.05.12
    '''
    #The surface fitting performs better if the data is scattered about zero
    x0  = (xpos - np.median(xpos))/a
    y0  = (ypos - np.median(ypos))/b
    
    if npix is not None and bool(c):
        np0 = np.sqrt(npix)
        np0 = (np0 - np.median(np0))/c
        features  = np.transpose((y0, x0, np0))
    else:
        features  = np.transpose((y0, x0))
        
        if np.sum(np0) == 0.0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because Noise Pixels are Zero')
        if c == 0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because c == 0')
    
    if inds is None:
        kdtree    = cKDTree(features * expansion) #Multiplying `features` by 1000.0 avoids precision problems
        inds      = kdtree.query(kdtree.data, n_nbr+1)[1][:,1:]
        
        print('WARNING: Because `inds` was not provided, we must now compute and return it here')
        returnInds= True
    
    n, k  = inds.shape
    # k   = sm_num                           # This is the number of nearest neighbors you want
    # n   = x0.size                          # This is the number of data points you have
    # gw  = np.zeros((k,n), dtype=np.float64) # This is the gaussian weight for each data point determined from the nearest neighbors
    
    func  = partial(find_qhull_one_point, x0=x0, y0=y0, np0=np0, inds=inds)
    
    pool  = Pool(nCores)
    
    gw_pool = pool.starmap(func, zip(range(n)))
    
    pool.close()
    pool.join()
    
    if returnInds:
        return np.array(gw_pool), inds
    else:
        return np.array(gw_pool)

def find_nbr_qhull(xpos, ypos, npix, sm_num = 100, a = 1.0, b = 0.7, c = 1.0, print_space = 10000.):
    '''
        Python Implimentation of N. Lewis method, described in Lewis etal 2012, Knutson etal 2012, Fraine etal 2013
        
        Taken from N. Lewis IDL code:
            
            Construct a 3D surface (or 2D if only using x and y) from the data
            using the qhull.pro routine.  Save the connectivity information for
            each data point that was used to construct the Delaunay triangles (DT)
            that form the grid.  The connectivity information allows us to only
            deal with a sub set of data points in determining nearest neighbors
            that are either directly connected to the point of interest or
            connected through a neighboring point
        
        Python Version:
            J. Fraine    first edition, direct translation from IDL 12.05.12
    '''
    from scipy import spatial
    #The surface fitting performs better if the data is scattered about zero
    
    npix    = np.sqrt(npix)
    
    x0  = (xpos - np.median(xpos))/a
    y0  = (ypos - np.median(ypos))/b
    
    if np.sum(npix) != 0.0 and c != 0:
        np0 = (npix - np.median(npix))/c
    else:
        if np.sum(npix) == 0.0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because Noise Pixels are Zero')
        if c == 0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because c == 0')
    
    k            = sm_num                           # This is the number of nearest neighbors you want
    n            = x0.size                          # This is the number of data points you have
    nearest      = np.zeros((k,n),dtype=np.int64)   # This stores the nearest neighbors for each data point
    
    #Multiplying by 1000.0 avoids precision problems
    if npix.sum() != 0.0 and c != 0:
        kdtree  = cKDTree(np.transpose((y0*1000., x0*1000., np0*1000.)))
    else:
        kdtree  = cKDTree(np.transpose((y0*1000., x0*1000.)))
    
    gw  = np.zeros((k,n),dtype=np.float64) # This is the gaussian weight for each data point determined from the nearest neighbors
    
    start   = time()
    for point in tqdm_notebook(range(n),total=n):
        ind         = kdtree.query(kdtree.data[point],sm_num+1)[1][1:]
        dx          = x0[ind] - x0[point]
        dy          = y0[ind] - y0[point]
        
        if npix.sum() != 0.0 and c != 0:
            dnp         = np0[ind] - np0[point]
        
        sigx        = np.std(dx )
        sigy        = np.std(dy )
        if npix.sum() != 0.0 and c != 0:
            signp       = np.std(dnp)
        if npix.sum() != 0.0 and c != 0:
            # gw_temp     = np.exp(-dx**2./(2.0*sigx**2.)) * np.exp(-dy**2./(2.*sigy**2.))  * np.exp(-dnp**2./(2.*signp**2.))
            gw_temp     = np.exp(-dx**2./(2.0*sigx**2.) - dy**2./(2.*sigy**2.) - dnp**2./(2.*signp**2.))
        else:
            # gw_temp     = np.exp(-dx**2./(2.0*sigx**2.)) * np.exp(-dy**2./(2.*sigy**2.))
            gw_temp     = np.exp(-dx**2./(2.0*sigx**2.) - dy**2./(2.*sigy**2.))
        
        # gw_sum      = gw_temp.sum()
        if gw_temp.sum():
            gw[:,point] = gw_temp/gw_temp.sum() 
        else:
            gw[:,point] = 0.0
        
        #if (gw_sum == 0.0) or ~np.isfinite(gw_sum):
        #    raise Exception('(gw_sum == 0.0) or ~isfinite(gw_temp))')
        
        nearest[:,point]  = ind
    
    return gw.transpose(), nearest.transpose() # nearest  == nbr_ind.transpose()

def pixel_level_decorrelation_instrument_profile(times, input_features, 
                              pld1_l, pld2_l, pld3_l, pld4_l, pld5_l, pld6_l, pld7_l, pld8_l, pld9_l,
                              pld1_q, pld2_q, pld3_q, pld4_q, pld5_q, pld6_q, pld7_q, pld8_q, pld9_q):# ,
                              # intcpt=1.0, slope=0.0, crvtur=0.0):
    
    feature_coeffs   = [pld1_l, pld2_l, pld3_l, pld4_l, pld5_l, pld6_l, pld7_l, pld8_l, pld9_l,                         pld1_q, pld2_q, pld3_q, pld4_q, pld5_q, pld6_q, pld7_q, pld8_q, pld9_q]
    
    instrumental  = dot(input_features, feature_coeffs) # This is now quadratic
    
    return instrumental

def clipOutlier2D(arr2D, nSig=10):
    arr2D     = arr2D.copy()
    medArr2D  = median(arr2D,axis=0)
    sclArr2D  = np.sqrt(((mad(arr2D)**2.).sum()))
    outliers  = abs(arr2D - medArr2D) >  nSig*sclArr2D
    inliers   = abs(arr2D - medArr2D) <= nSig*sclArr2D
    arr2D[outliers] = median(arr2D[inliers],axis=0)
    return arr2D

def plot_MAD_AperRads(instance, minRad=None, maxRad=None, varRadFlag=True):
    color_cycle = cycler(rcParams['axes.prop_cycle']).by_key()['color']
    
    quad_width= instance.quadrature_widths.values
    vrad_dist = quad_width - np.median(quad_width)
    vrad_dist = clipOutlier2D(vrad_dist, nSig=5)
    vrad_dist_med = np.median(vrad_dist)
    
    betaColor = 7
    quadColor = 8
    
    ax = figure().add_subplot(111)
    
    for key in instance.flux_TSO_df.keys():
        staticRad = float(key.split('_')[-2])
        varRad    = float(key.split('_')[-1])
        aperRad   = staticRad + varRad*vrad_dist_med
        colorNow  = color_cycle[int(varRad*4)]
        
        if 'betaRad' in key:
            aperRad    = median(sqrt(instance.effective_widths))
            colorNow  = color_cycle[int(betaColor)]
        
        if 'quadRad' in key:
            aperRad   = 2*sqrt(2*log(2))*median(instance.quadrature_widths.values)
            colorNow  = color_cycle[int(quadColor)]
        
        if minRad is not  None and maxRad is not None:
            if aperRad > minRad and aperRad < maxRad:
                ax.scatter(aperRad, mad(np.diff(instance.flux_TSO_df[key])), color=colorNow, zorder=int(varRad*4))
        else:
            ax.scatter(aperRad, mad(np.diff(instance.flux_TSO_df[key])), color=colorNow, zorder=int(varRad*4))
    
    for varRad in [0.,0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        colorNow  = color_cycle[int(varRad*4)]
        ax.scatter([],[], color=colorNow, label=varRad)
    
    ax.scatter([],[],color=color_cycle[int(betaColor)], label='Beta')
    ax.scatter([],[],color=color_cycle[int(quadColor)], label='Quad')
    
    ax.set_xlabel('StaticRad + Average(varRad)')
    ax.set_ylabel('MAD( Diff ( Flux ) )')
    ax.legend(loc=0,fontsize=10)
    
    ax.set_title('' if not hasattr(instance, '__name__') else instance.__name__)

def plot_model_over_reduced_lc_half_PLDsq_universal(times, phots, phots_err, input_features, in_eclipse, fit_values,
                                                    spider_params = None, refTime=0.0, figsize=None, nbins=200, 
                                                    plotRawData=False, alpha=1.0, phase_method='fourier', 
                                                    baseline='PLD',ax = None, returnAx = False, 
                                                    model_color=0, data_color=1):
    
    color_cycle = cycler(rcParams['axes.prop_cycle']).by_key()['color']
    model_color = color_cycle[model_color] if isinstance(model_color, int) else model_color
    data_color  = color_cycle[data_color]  if isinstance(data_color , int) else data_color 
    
    # print('model:{}\tdata:{}'.format(model_color, data_color))
    
    clean_model = model_selector(fit_values, times, input_features, in_eclipse, spider_params=spider_params, 
                                 phase_method=phase_method, baseline=baseline, model_only=True)
    
    model_params_keys = np.sort(list(fit_values.keys()))
    
    feature_coeffs  = [fit_values[key].value for key in model_params_keys if 'pld' in key and '_l' in key] +                       [fit_values[key].value for key in model_params_keys if 'pld' in key and '_q' in key]
    
    # Check if line fit parameters exist, then allocate them; or use defaults
    intcpt  = fit_values['intcpt'] if 'intcpt' in fit_values.keys() else 1.0
    slope   = fit_values['slope']  if 'slope'  in fit_values.keys() else 0.0
    crvtur  = fit_values['crvtur'] if 'crvtur' in fit_values.keys() else 0.0
    
    instrumental = intcpt * np.ones(times.size)
    if slope  != 0.0:
        instrumental += slope * (times-times.mean())
    if crvtur != 0.0:
        instrumental += crvtur * (times-times.mean())**2.
    
    if baseline == 'PLD':
        instrumental += pixel_level_decorrelation_instrument_profile(times, input_features, 
                        *feature_coeffs)#, intcpt=intcpt, slope=slope, crvtur=crvtur)
    
    if baseline == 'KRData':
        phots_now, gk, nbr_ind = input_features
        instrumental *= np.sum((phots_now/clean_model)[nbr_ind]*gk,axis=1)
    
    # clean_model  = whole_model / instrumental 
    
    binsize = phots.size // nbins
    
    reduced = phots/instrumental
    diff_dm = 0.0#np.median(reduced - clean_model)
    
    print('Diff DM: {:0f}'.format(diff_dm*ppm))
    
    bin_flux    , _ = bin_array((reduced - diff_dm), binsize=binsize)
    bin_flux_err, _ = bin_array(phots_err, binsize=binsize)
    
    bin_time    , _  = bin_array(times, binsize=binsize)
    
    bin_flux_err     = bin_flux_err/sqrt(binsize)
    
    if ax is None:
        fig= figure() if figsize is None else figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    if plotRawData:
        ax.plot(times, phots/ instrumental, '.', ms=1, label='data', alpha=alpha)
    
    # ingress   = np.where(in_eclipse)[0].min()
    # egress    = np.where(in_eclipse)[0].max()
    # IT_mean   = np.mean([clean_model[ingress:egress]])
    # gap       = IT_mean - 1.0
    gap = 0
    print('Gap: {:0f}'.format(gap*ppm))
    
    ax.errorbar(bin_time-refTime, bin_flux    - gap, bin_flux_err, color=color_cycle[0],                     fmt='o', label='binned data', ms=1, zorder=0, lw=1,alpha=alpha)
    ax.plot(times-refTime, clean_model - gap, color=color_cycle[1],                 lw=2,label='best fit model', zorder=1)
    
    ax.legend(loc=0)
    
    # ax.axhline(1.0, ls='--', lw=0.5)
    
    if returnAx:
        return ax

def plot_model_over_reduced_lc_full_PLDsq_universal(times_0, times_1, phots_0, phots_1, phots_err_0, phots_err_1, 
                                                    input_features_0, input_features_1, in_eclipse_0, in_eclipse_1, 
                                                    fit_values, refTime=0.0, planetName='', figsize=None, 
                                                    nbins=200, alpha=1.0, plotRawData=False, 
                                                    phase_method='fourier', baseline='PLD', 
                                                    ax = None, returnAx = False, model_color=0, data_color=1):
    
    none_to_inf  = lambda x, sign=1: sign*np.inf if x is None else x
    
    _params_names = ['period', 'tCenter', 'inc', 'aprs', 'edepth', 'tdepth', 'ecc', 'omega', 'u1', 'u2', 
                     'amp1', 'amp2', 'amp3', 'amp4', 'amp5', 'amp6', 'amp7', 'amp8', # 'mean', 
                     'pld1_l', 'pld2_l', 'pld3_l', 'pld4_l', 'pld5_l', 'pld6_l', 'pld7_l', 'pld8_l', 'pld9_l', 
                     'pld1_q', 'pld2_q', 'pld3_q', 'pld4_q', 'pld5_q', 'pld6_q', 'pld7_q', 'pld8_q', 'pld9_q']#, 
                     # 'intcpt', 'slope', 'crvtur']
    
    fit_values_0 = Parameters()
    fit_values_1 = Parameters()
    
    for pname in _params_names:
        if 'pld' not in pname:
            valu = fit_values[pname].value
            vary = fit_values[pname].vary
            rmin = none_to_inf(fit_values[pname].min, -1)
            rmax = none_to_inf(fit_values[pname].max,  1)
            
            fit_values_0.add(pname, valu, vary, rmin, rmax)
            fit_values_1.add(pname, valu, vary, rmin, rmax)
        else:
            pname_0 = pname + '_1'
            valu = fit_values[pname_0].value
            vary = fit_values[pname_0].vary
            rmin = none_to_inf(fit_values[pname_0].min, -1)
            rmax = none_to_inf(fit_values[pname_0].max,  1)
            
            fit_values_0.add(pname, valu, vary, rmin, rmax)
            
            pname_1 = pname + '_2'
            valu = fit_values[pname_1].value
            vary = fit_values[pname_1].vary
            rmin = none_to_inf(fit_values[pname_1].min, -1)
            rmax = none_to_inf(fit_values[pname_1].max,  1)
            fit_values_1.add(pname, valu, vary, rmin, rmax)
    
    fit_values_0['mean'] = fit_values['mean_0']
    fit_values_1['mean'] = fit_values['mean_1']
    
    # nPeriods = int((times_0.mean() - fit_values_0['tCenter'])/fit_values_0['period'])
    
    ax_0 = plot_model_over_reduced_lc_half_PLDsq_universal(times_0, phots_0, phots_err_0,
                                                           input_features_0, in_eclipse_0, 
                                                           fit_values_0, refTime=refTime, 
                                                           figsize=figsize, nbins=nbins, plotRawData=plotRawData,
                                                           phase_method=phase_method, baseline=baseline,
                                                           ax = ax, returnAx = True, alpha=alpha, 
                                                           model_color=model_color, data_color=data_color)
    
    ax_0.set_title(planetName + ' E-depth: {:.0f}'.format(fit_values['edepth'].value*ppm))
    
    return plot_model_over_reduced_lc_half_PLDsq_universal(times_1, phots_1, phots_err_1,
                                                           input_features_1, in_eclipse_1, 
                                                           fit_values_1, refTime=refTime, alpha=alpha,  
                                                           figsize=figsize, nbins=nbins, plotRawData=plotRawData,
                                                           phase_method=phase_method, baseline=baseline, 
                                                           ax = ax_0, returnAx = returnAx,
                                                           model_color=model_color, data_color=data_color)

# **EMCEE Setup**
def create_prior(params):
    """
    emccee uses a uniform prior for every variable.
    Here we create a functions which checks the bounds
    and returns np.inf if a value is outside of its
    allowed range. WARNING: A uniform prior may not be
    what you want!
    """
    none_to_inf  = lambda x, sign=1: sign*np.inf if x is None else x
    lower_bounds = np.array([none_to_inf(i.min, -1) for i in params.values() if i.vary])
    upper_bounds = np.array([none_to_inf(i.max,  1) for i in params.values() if i.vary])
    
    def bounds_prior(values):
        values = np.asarray(values)
        is_ok = np.all((lower_bounds < values) & (values < upper_bounds))
        return 0 if is_ok else -np.inf
    
    return bounds_prior

def create_lnliklihood(mini, sigma=None):
    """create a normal-likihood from the residuals"""
    def lnprob(vals, sigma=sigma):
        for v, p in zip(vals, [p for p in mini.params.values() if p.vary]):
            p.value = v
        
        # residuals = mini.residual
        
        if not sigma:
            # sigma is either the RMS estimate or it will
            # be part of the sampling.
            sigma = vals[-1]
        
        val = -0.5*np.sum(np.log(2*np.pi*sigma**2) + (partial_residuals(mini.params)/sigma)**2)
        
        return val
    
    return lnprob

def starting_guess(mini, estimate_sigma=True):
    """
    Use best a fit as a starting point for the samplers.
    If no sigmas are given, it is assumed that
    all points have the same uncertainty which will
    be also part of the sampled parameters.
    """
    vals = [i.value for i in mini.params.values() if i.vary]
    
    if estimate_sigma:
        vals.append(mini.residual.std())
    
    return vals

def create_all(mini, sigma=None):
    """
    creates the log-poposterior function from a minimizer.
    sigma should is either None or an array with the
    1-sigma uncertainties of each residual point. If None,
    sigma will be assumed the same for all residuals and
    is added to the sampled parameters.
    """
    sigma_given = not sigma is None
    
    lnprior = create_prior(mini.params)
    lnprob  = create_lnliklihood(mini, sigma=sigma)
    guess   = starting_guess(mini, not sigma_given)
    
    if sigma_given:
        func = lambda x: lnprior(x[:]) + lnprob(x)
    else:
        func = lambda x: lnprior(x[:-1]) + lnprob(x)
    
    return func, guess

def model_selector(model_params, times, input_features, in_eclipse, 
                   spider_params=None, phase_method='fourier', 
                   baseline = 'PLD', simple=False, model_only=False):
    """Identifies the phase curve + noise model to use from `method`, allocates 
        then model specific params and returns the model
    
    Args:
        model_params    : set of model parmaeters  -- named to avoid global / local confusion
        times           : temporal array to use with model
        input_features  : basis vectors for linear fitting to noise array
        in_eclipse      : boolean array to identify the eclipse location from initial guess
        method          : string to identify if residuals should be taken over 
                            `fourier` or `spiderman` phase curve models
    
    Returns:
        Returns the current model, which is phase_curve x transit x eclipse x normalization
    """
    
    # Grab list of pld1..9 linear, then add list of pld1..9 quadratic
    model_params_keys      = np.sort(list(model_params.keys()))
    
    feature_coeffs  = [model_params[key].value for key in model_params_keys if 'pld' in key and '_l' in key] + \
                      [model_params[key].value for key in model_params_keys if 'pld' in key and '_q' in key]
    
    # Check if line fit parameters exist, then allocate them; or use defaults
    intcpt  = model_params['intcpt'].value if 'intcpt' in model_params.keys() else 1.0
    slope   = model_params['slope'].value  if 'slope'  in model_params.keys() else 0.0
    crvtur  = model_params['crvtur'].value if 'crvtur' in model_params.keys() else 0.0
    
    # Transit Parameters
    period  = model_params['period'].value
    tCenter = model_params['tCenter'].value
    inc     = model_params['inc'].value
    aprs    = model_params['aprs'].value
    edepth  = model_params['edepth'].value
    tdepth  = model_params['tdepth'].value
    ecc     = model_params['ecc'].value
    omega   = model_params['omega'].value
    u1      = model_params['u1'].value
    u2      = model_params['u2'].value
    
    delta_phase = deltaphase_eclipse(ecc, omega) if ecc is not 0.0 else 0.5
    t_secondary = tCenter + period*delta_phase
    
    # Create phase curve model -- including PLD noise parameters
    if phase_method == 'spiderman':
        # Siderman Phase Curve Parameters
        xi      = model_params['xi'].value
        T_night = model_params['T_night'].value
        delta_T = model_params['delta_T'].value
        mean    = model_params['mean'].value
        
        spider_model  = partial(spiderman_lmfit_model, period=period, tCenter=tCenter, inc=inc, aprs=aprs, 
                                           tdepth=tdepth, ecc=ecc, omega=omega, u1=u1, u2=u2, 
                                           spider_params=spider_params, 
                                           xi=xi, T_night=T_night, delta_T=delta_T, if_eclipse=False)
        
        phase_curve   = spider_model(times=times) + mean
    
    if phase_method == 'fourier':
        mean = model_params['mean'].value
        amp1 = model_params['amp1'].value
        amp2 = model_params['amp2'].value
        amp3 = model_params['amp3'].value
        amp4 = model_params['amp4'].value
        # amp5 = model_params['amp5'].value
        # amp6 = model_params['amp6'].value
        # amp7 = model_params['amp7'].value
        # amp8 = model_params['amp8'].value
        
        angphase       = 2 * pi / period * (times-t_secondary)
        
        phase_curve   = mean + amp1*cos(angphase)   \
                             + amp2*sin(angphase)   \
                             + amp3*cos(2*angphase) \
                             + amp4*sin(2*angphase) # \
                             # + amp5*cos(4*angphase) \
                             # + amp6*sin(4*angphase) \
                             # + amp7*cos(6*angphase) \
                             # + amp8*sin(6*angphase)
    
    if phase_method == 'kbs':
        mean = model_params['mean'].value
        amp1 = model_params['amp1'].value
        amp2 = model_params['amp2'].value
        
        angphase      = 2 * pi / period * (times - t_secondary - amp2)
        phase_curve   = mean + amp1*cos(angphase)
    
    if phase_method in ['transit', 'eclipse', 'None', None]:
        phase_curve   = 1.0
    
    instrumental = intcpt * np.ones(times.size)
    if slope  != 0.0:
        instrumental += slope * (times-times.mean())
    if crvtur != 0.0:
        instrumental += crvtur * (times-times.mean())**2.
    
    batman_model  = partial(batman_lmfit_model, period=period, tCenter=tCenter, inc=inc, aprs=aprs  , 
                                       times=times, edepth=edepth, tdepth=tdepth  , ecc=ecc, omega=omega)
    
    if phase_method is 'eclipse':
        transit = 1.0
    else:
        transit = batman_model(transittype="primary"  , ldtype='quadratic', u1=u1, u2=u2)
    
    if phase_method is 'transit':
        eclipse = 1.0
    else:
        eclipse = batman_model(transittype="secondary", ldtype='uniform') - edepth
    
    # eclipse_1     = (eclipse == 1 - edepth)#*()
    # eclipse_2     = (eclipse == 1 - edepth)#*()
    # in_eclipse    = eclipse != 1# - edepth
    #if in_eclipse.any():
    #    # midpoint = int(np.where(in_eclipse)[0].mean())
    #    phase_curve[eclipse_1] = 1.0#phase_curve[in_eclipse]#np.mean([phase_curve[in_eclipse]])#eclipse[in_eclipse]#
    
    try:
        phase_curve[eclipse == 1.0 - edepth] = np.mean(phase_curve[eclipse == 1.0 - edepth])
    except:
        pass # `phase_curve` is not iterable because `phase_method` is in ['transit', 'eclipse', 'None', None]
    
    clean_model = transit * eclipse * phase_curve
    
    if baseline == 'PLD':
        instrumental += pixel_level_decorrelation_instrument_profile(times, input_features, 
                        *feature_coeffs)#, intcpt=intcpt, slope=slope, crvtur=crvtur)
    
    if baseline == 'KRData':
        phots_now, gk, nbr_ind = input_features
        instrumental *= np.sum((phots_now/clean_model)[nbr_ind]*gk,axis=1)
    
    where_nan = np.where(np.isnan(instrumental))[0]
    
    # print(instrumental[where_nan-1])
    # print(instrumental[where_nan-0])
    # print(instrumental[where_nan+1])
    # print([instrumental[where_nan-1], instrumental[where_nan+1]])
    # print(np.array([instrumental[where_nan-1], instrumental[where_nan+1]]).mean(axis=0))
    
    instrumental[where_nan] = np.array([instrumental[where_nan-1], instrumental[where_nan+1]]).mean(axis=0)
    
    if model_only:
        return clean_model
    else:
        return clean_model * instrumental

def residuals_func(model_params, data, times, input_features, in_eclipse, simple=True, 
                   spider_params=None, phase_method='fourier', baseline='PLD'):
    """Returns residuals of current model estimate with input `data`
    
    Args:
        model_params    : set of model parmaeters  -- named to avoid global / local confusion
        data            : data to be fit
        times           : temporal array to use with model
        input_features  : basis vectors for linear fitting to noise array
        in_eclipse      : boolean array to identify the eclipse location from initial guess
        method          : string to identify if residuals should be taken over 
                            `fourier` or `spiderman` phase curve models
    
    Returns:
        Returns the array of residuals, that is the data - model
    """
    # Return residuals between this model and the `data` allocated above
    model_now = model_selector(model_params, times, input_features, in_eclipse, 
                               spider_params, phase_method, baseline, simple=simple)
    
    return model_now - data

def load_aor_from_phot_pipeline(planet_dir_name, channel, AORNow, PLD_order=1):
    loadfiledir         = environ['HOME']+'/Research/Planets/PhaseCurves/'+planet_dir_name+'/saveFiles/' + channel + '/' 
    loadFileNameHeader  = planet_dir_name+'_'+ AORNow +'_Median'
    loadFileType        = '.pickle.save'
    
    print()
    print('Loading from ' + loadfiledir + loadFileNameHeader + loadFileType)
    print()
    
    instance = wanderer(fitsFileDir=loadfitsdir, filetype=filetype, telescope='Spitzer', 
                                                yguess=yguess, xguess=xguess, method='median', nCores=cpu_count())
    
    instance.load_data_from_save_files(savefiledir=loadfiledir,
                                       saveFileNameHeader=loadFileNameHeader,
                                       saveFileType='.pickle.save')
    
    if PLD_order > 1: instance.extract_PLD_components(order=PLD_order)
    
    timeCubeLocal       = instance.timeCube
    photsLocal          = instance.flux_TSO_df.values
    PLDFeatureLocal     = instance.PLD_components.T
    
    try:
        inliers_Phots_local = instance.inliers_Phots.values()
    except:
        inliers_Phots_local = np.ones(photsLocal.shape)
    
    try:
        inliers_PLD_local = instance.inliers_PLD.values()
    except:
        inliers_PLD_local = np.ones(PLDFeatureLocal.shape)
    
    return timeCubeLocal, photsLocal, PLDFeatureLocal, inliers_Phots_local, inliers_PLD_local, instance

# **Load Planet Params**
ppm             = 1e6
y,x             = 0,1

yguess, xguess  = 15., 15.   # Specific to Spitzer circa 2010 and beyond
filetype        = 'bcd.fits' # Specific to Spitzer Basic Calibrated Data

# **Starting Position**
planet_params_name = 'Planet-Name b'  # the exact name that is called by `exoparams` via exoplanets.org
planet_dir_name    = 'planet_name'# directory inside `basedir` that the data is stored

try:
    channel = sys.argv[1]
except:
    channel = 'ch2'

if   'ch1' in channel:
    spitzer_waves_low  = 3.0
    spitzer_waves_high = 4.0
elif 'ch2' in channel:
    spitzer_waves_low  = 4.0
    spitzer_waves_high = 5.0
else:
    raise Exception("`channel` must be either `'ch1'` or `'ch2'`")

planet_params = PlanetParams(planet_params_name)

iPeriod   = planet_params.per.value
iTCenter  = planet_params.tt.value-2400000.5
# iBImpact  = planet_params.b.value
iApRs     = planet_params.ar.value
iInc      = planet_params.i.value
# iRsAp     = 1.0/planet_params.ar.value
iEdepth   = 3000/ppm # blind guess
iTdepth   = planet_params.depth.value
iEcc      = planet_params.ecc.value
iOmega    = planet_params.om.value*pi/180

stellar_radius = planet_params.rstar.value
stellar_temp   = planet_params.teff.value

# # Pixel Level Decorrelation
planetDirectory = '/Research/Planets/PhaseCurves/'

dataSub = 'bcd/'

dataDir     = environ['HOME'] + planetDirectory + planet_dir_name + '/data/raw/' + channel + '/big/'
print(dataDir + '*')
AORs = []
for dirNow in glob(dataDir + '*'):
    print(dirNow)
    AORs.append(dirNow.split('/')[-1])

fileExt = '*bcd.fits'
uncsExt = '*bunc.fits'

# This is required, but irrelevent for existing saved data
iAOR        = 1
AORNow      = AORs[iAOR]
loadfitsdir = dataDir + AORNow + '/' + channel + '/' + dataSub
print(loadfitsdir)

timeCubeStack       = {}
photsStack          = {}
PLDFeatureStack     = {}
inliers_PhotsStack0 = {}
inliers_PLDStack    = {}
instanceStack       = {}

for AORNow in AORs:
    thingy = load_aor_from_phot_pipeline(planet_dir_name, channel + '/', AORNow, PLD_order=2)
    
    timesNow, photsNow, PLDFeaturesNow, inliers_PhotsNow, inliers_PLDNow, instanceNow = thingy
    
    del thingy
    
    timeCubeStack[AORNow]       = timesNow
    photsStack[AORNow]          = photsNow
    PLDFeatureStack[AORNow]     = PLDFeaturesNow
    inliers_PhotsStack0[AORNow] = inliers_PhotsNow
    inliers_PLDStack[AORNow]    = inliers_PLDNow
    instanceStack[AORNow]       = instanceNow

try:
    aor_0, aor_1 = timeCubeStack.keys();
except:
    aor_0, aor_1, aor_2 = timeCubeStack.keys();
    print('NEED TO ADJUST FOR 3 AORs')

if timeCubeStack[aor_0].min() > timeCubeStack[aor_1].min():
    print('Reversing AOR Labels to be Time Ordered')
    aor_0, aor_1 = aor_1, aor_0

# i_u1, i_u2 = 0.1, 0.1
# xi_0, T_night_0, delta_T_0 = 1e-5, 750.0, 500.0

i_u1, i_u2, iEdepth = 0.1, 0.0, 2000/ppm
xi_0, T_night_0, delta_T_0 = 1e-5, 750.0, 500.0

initialParams_KRData_half_Fourier_PhaseCurve = Parameters()

initialParams_KRData_half_Fourier_PhaseCurve.add_many(
    # Planetary Parameters
    ('period' , iPeriod  , False),
    ('tCenter', iTCenter , True , iTCenter-0.02, iTCenter+0.02),
    ('inc'    , iInc     , False, 0.0 ,  90.),
    ('aprs'   , iApRs    , False, 0.0 , 100.),
    ('edepth' , iEdepth  , True , 0.0 , 1.0 ),
    ('tdepth' , iTdepth  , True , 0.0 , 1.0 ),
    ('ecc'    , 0.0      , False, 0.0 , 1.0 ),
    ('omega'  , iOmega   , False, 0.0 , 360.),
    ('u1'     , i_u1     , True , 0.0 , 1.0 ),
    ('u2'     , i_u2     , False, 0.0 , 1.0 ),
    ('amp1'   , 1e-3     , True ),
    # ('amp2'   , 1e-3     , True ), # for non-KBS only
    ('amp2'   , 1e-4     , True , 0.0, 0.25*iPeriod ), # for KBS only
    ('amp3'   , 1e-4     , False),
    ('amp4'   , 1e-4     , False),
    ('amp5'   , 0.0      , False),
    ('amp6'   , 0.0      , False),
    ('amp7'   , 0.0      , False),
    ('amp8'   , 0.0      , False),
    ('mean'   , 1.0      , True),
    ('xi'     , xi_0     , False, 0.0, inf),
    ('delta_T', delta_T_0, False, 0.0, inf),
    ('T_night', T_night_0, False, 0.0, inf),
    # Out of transit linear baselines
    ('intcpt' , 1.0      , True ),
    ('slope'  , 0.0      , True ),
    ('crvtur' , 0.0      , False)
)

init_eclipse_params             = TransitParams()


# iTCenter  = planet_params.tt.value-2400000.5+443.315*planet_params.per.value

init_eclipse_params.per         = iPeriod
init_eclipse_params.t0          = iTCenter
init_eclipse_params.inc         = iInc
init_eclipse_params.a           = iApRs
init_eclipse_params.fp          = iEdepth
init_eclipse_params.rp          = sqrt(iTdepth)
init_eclipse_params.ecc         = iEcc
init_eclipse_params.w           = iOmega

init_eclipse_params.delta_phase = deltaphase_eclipse(init_eclipse_params.ecc, init_eclipse_params.w) if init_eclipse_params.ecc is not 0.0 else 0.5
init_eclipse_params.t_secondary = init_eclipse_params.t0 + init_eclipse_params.per*init_eclipse_params.delta_phase

init_eclipse_params.limb_dark   = 'uniform'
init_eclipse_params.u           = []

# staticRad   = '2.0'
# varRad      = '0.0'
# phot_select = np.where(instance_0.flux_TSO_df.keys() == 'Gaussian_Fit_AnnularMask_rad_'+staticRad+'_'+varRad)[0][0]

skip           = 1000
nSig           = 6
only_plot_once = True
phase_method   = 'kbs'
baseline       = 'KRData'

instance_0  = instanceStack[aor_0]
instance_1  = instanceStack[aor_1]

instance_0.__name__ = aor_0
instance_1.__name__ = aor_1

for flux_key_now in tqdm(instance_0.flux_TSO_df.keys(), total = len(instance_0.flux_TSO_df.keys()), desc='AperRad'):
    fits_result_save_name = 'lmfit_save_files/' + planet_params_name.replace(' ','_') + '_full_phase_curve_' + channel + \
                            '_observations_'    + phase_method + '_' + baseline + '_' + flux_key_now + '.pickle.save'
    
    print('Running '   + flux_key_now)
    print('Saving to ' + fits_result_save_name)
    
    phot_select   = np.where(instance_0.flux_TSO_df.keys() == flux_key_now)[0][0]
    
    inliersMaster        = {}
    inliersMaster[aor_0] = array(list(inliers_PhotsStack0[aor_0])).all(axis=0)
    inliersMaster[aor_1] = array(list(inliers_PhotsStack0[aor_1])).all(axis=0)
    
    inliersMaster[aor_0] = inliersMaster[aor_0] * inliers_PLDStack[aor_0].all(axis=1)
    inliersMaster[aor_1] = inliersMaster[aor_1] * inliers_PLDStack[aor_1].all(axis=1)
    
    for aorNow in inliersMaster.keys():
        if inliersMaster[aorNow].all():
            print('Working on AOR {} and flux_key {}'.format(aorNow, flux_key_now))
            cy_now, cx_now        = instanceStack[aorNow].centering_GaussianFit.T
            phots_now             = photsStack[aorNow][:,phot_select]
            phots_clipped         = clipOutlier2D(phots_now, nSig=nSig)
            cy_clipped, cx_clipped= clipOutlier2D(transpose([cy_now, cx_now]),nSig=nSig).T
            arr2D_clipped         = transpose([phots_clipped, cy_clipped, cx_clipped])
            inliersMaster[aorNow] = (phots_clipped == phots_now)*(cy_clipped==cy_now)*(cx_clipped==cx_now)
    
    times_0 = timeCubeStack[aor_0]#[inliersMaster[aor_0]]
    times_1 = timeCubeStack[aor_1]#[inliersMaster[aor_1]]
    
    PLDfeatures_0 = PLDFeatureStack[aor_0]#[inliersMaster[aor_0]]
    PLDfeatures_1 = PLDFeatureStack[aor_1]#[inliersMaster[aor_1]]
    
    phots_0 = photsStack[aor_0][:, phot_select]#[inliersMaster[aor_0], phot_select]
    phots_1 = photsStack[aor_1][:, phot_select]#[inliersMaster[aor_1], phot_select]
    
    phots_0_err = np.sqrt(abs(phots_0))
    phots_1_err = np.sqrt(abs(phots_1))
    
    # **Setup Initial Parameters**
    init_1st_eclipse_model          = TransitModel(init_eclipse_params, times_0, transittype="secondary")
    init_1st_eclipse_lightcurve     = init_1st_eclipse_model.light_curve(init_eclipse_params)
    
    init_1st_transit_model          = TransitModel(init_eclipse_params, times_0, transittype="primary")
    init_1st_transit_lightcurve     = init_1st_transit_model.light_curve(init_eclipse_params)
    
    in_1st_eclipse = init_1st_eclipse_lightcurve == 1
    
    # **Choose to Skip First ~30 Minutes or Not**
    SKIP_30_MIN = True
    if SKIP_30_MIN:
        nskip   = 1000 # with 2s integrations, 1000 frames ~ 33 minutes
        inliersMaster[aor_0][:nskip] = False
    
    ypos_0, xpos_0  = clipOutlier2D(transpose([instance_0.centering_GaussianFit.T[y][inliersMaster[aor_0]], instance_0.centering_GaussianFit.T[x][inliersMaster[aor_0]]])).T
    ypos_1, xpos_1  = clipOutlier2D(transpose([instance_1.centering_GaussianFit.T[y][inliersMaster[aor_1]], instance_1.centering_GaussianFit.T[x][inliersMaster[aor_1]]])).T
    
    npix_0          = sqrt(instance_0.effective_widths[inliersMaster[aor_0]])
    npix_1          = sqrt(instance_1.effective_widths[inliersMaster[aor_1]])
    
    if False and flux_key_now == 'Gaussian_Fit_AnnularMask_rad_2.0_0.0':
        # only_plot_once = False
        med_phots_0 = median(phots_0[inliersMaster[aor_0]])
        std_phots_0 = std(phots_0[inliersMaster[aor_0]])
        
        y_centers_stack = hstack([instance_0.centering_GaussianFit.T[y][inliersMaster[aor_0]], instance_1.centering_GaussianFit.T[y][inliersMaster[aor_1]]])
        x_centers_stack = hstack([instance_0.centering_GaussianFit.T[x][inliersMaster[aor_0]], instance_1.centering_GaussianFit.T[x][inliersMaster[aor_1]]])
        
        fig = figure(figsize=(20,20))
        ax  = fig.add_subplot(111)
        ax.hist2d(x_centers_stack, y_centers_stack, bins=50, cmap=plt.cm.jet);#, range=wasp77_range_ch1);
        fig.savefig('figure_results/' + channel + '/centering_hist2d_multiAOR_{}_{}.png'.format(channel.replace('/',''), flux_key_now))
        
        ax.clear()
        ax.plot(*(transpose([instance_0.centering_GaussianFit.T[x][inliersMaster[aor_0]], instance_0.centering_GaussianFit.T[y][inliersMaster[aor_0]]])).T,'.',ms=1,alpha=0.25);
        ax.plot(*(transpose([instance_1.centering_GaussianFit.T[x][inliersMaster[aor_1]], instance_1.centering_GaussianFit.T[y][inliersMaster[aor_1]]])).T,'.',ms=1,alpha=0.25);
        fig.savefig('figure_results/' + channel + '/centering_scatterplot_multiAOR_{}_{}.png'.format(channel.replace('/',''), flux_key_now))
        
        ax.clear()
        ax.plot(photsStack[aor_0][:,phot_select]);
        ax.plot(np.arange(phots_0.shape[0])[skip:], phots_0[skip:],'.',ms=2);
        ax.plot(np.arange(phots_0.shape[0])[:skip], phots_0[:skip],'.',ms=2);
        
        ax.plot(np.arange(init_1st_eclipse_lightcurve.size), init_1st_transit_lightcurve*med_phots_0);
        ax.plot(np.arange(init_1st_eclipse_lightcurve.size), init_1st_eclipse_lightcurve*med_phots_0);
        ax.plot(np.arange(init_1st_eclipse_lightcurve.size)[in_1st_eclipse],      init_1st_eclipse_lightcurve[in_1st_eclipse]*med_phots_0);
        ax.set_ylim(med_phots_0 - 5*std_phots_0, med_phots_0 + 5*std_phots_0)
        
        fig.savefig('figure_results/' + channel + '/initial_photometry_multiAOR_{}_{}.png'.format(channel.replace('/',''), flux_key_now))
        plt.close('all')
        
        fig = figure(figsize=(30,10));
        ax1 = fig.add_subplot(1,3,1);
        ax2 = fig.add_subplot(1,3,2);
        ax3 = fig.add_subplot(1,3,3);
        ax1.plot(xpos_0, ypos_0,'.',ms=1,alpha=0.25);
        ax2.plot(xpos_0, npix_0,'.',ms=1,alpha=0.25);
        ax3.plot(ypos_0, npix_0,'.',ms=1,alpha=0.25);
        ax1.plot(xpos_1, ypos_1,'.',ms=1,alpha=0.25);
        ax2.plot(xpos_1, npix_1,'.',ms=1,alpha=0.25);
        ax3.plot(ypos_1, npix_1,'.',ms=1,alpha=0.25);
        fig.savefig('figure_results/' + channel + '/xpos_ypos_npix_scatter_plot_{}_{}.png'.format(channel.replace('/',''), flux_key_now))
    
    xpos_c  = np.hstack([xpos_0, xpos_1])
    ypos_c  = np.hstack([ypos_0, ypos_1])
    npix_c  = np.hstack([npix_0, npix_1])
    
    gw_c, nbr_c = mp_find_nbr_qhull(xpos_c, ypos_c, npix_c, a = 1.0, b = 0.7, c = 1.0)
    
    color_cycle = cycler(rcParams['axes.prop_cycle']).by_key()['color']
    
    med_eclipse_1 = median(phots_0[in_1st_eclipse == 1.0])
    
    #
    # # **SPIDERMAN Fitting Procedure**
    # # **Fit Spiderman Model with PLD to 1st AOR**
    # spider_params_hoststar                = sp_ModelParams(brightness_model='zhang')
    # spider_params_hoststar.n_layers       = 20
    # spider_params_hoststar.stellar_radius = stellar_radius
    # spider_params_hoststar.T_s            = stellar_temp
    # spider_params_hoststar.l1             = spitzer_waves_low  # Spitzer IRAC-1 or IRAC-2
    # spider_params_hoststar.l2             = spitzer_waves_high # Spitzer IRAC-1 or IRAC-2
    #
    # stellar_a_au  = iApRs * stellar_radius * R_sun.value / au.value
    #
    # spider_params_hoststar.t0      = iTCenter       # Central time of PRIMARY transit [days]
    # spider_params_hoststar.per     = iPeriod        # Period [days]
    # spider_params_hoststar.a_abs   = stellar_a_au   # The absolute value of the semi-major axis [AU]
    # spider_params_hoststar.inc     = iInc          # orbital inclination (in degrees)
    # spider_params_hoststar.ecc     = iEcc          # Eccentricity
    # spider_params_hoststar.w       = iOmega        # Argument of periastron
    # spider_params_hoststar.rp      = sqrt(iTdepth) # planet radius (in units of stellar radii)
    # spider_params_hoststar.a       = iApRs         # Semi-major axis scaled by stellar radius
    # spider_params_hoststar.p_u1    = i_u1          # Planetary limb darkening parameter
    # spider_params_hoststar.p_u2    = i_u2          # Planetary limb darkening parameter
    #
    # spider_params_hoststar.eclipse = False
    # spider_params_hoststar.xi      = 0.0           # Ratio of radiative to advective timescale
    # spider_params_hoststar.T_n     = 1000.0        # Temperature of nightside
    # spider_params_hoststar.delta_T = 500.0         # Day-night temperature contrast
    
    # **Global Fitting Procedure**
    # **Test KR-Data on Whole System**
    phots_c = np.hstack([phots_0[inliersMaster[aor_0]], phots_1[inliersMaster[aor_1]]])
    times_c = np.hstack([times_0[inliersMaster[aor_0]], times_1[inliersMaster[aor_1]]])
    
    partial_residuals = partial(residuals_func, 
                                data           = phots_c / median(phots_c),
                                times          = times_c, 
                                input_features = [phots_c / median(phots_c), gw_c, nbr_c],
                                in_eclipse     = None,
                                # spider_params  = spider_params_hoststar,
                                phase_method   = phase_method,
                                baseline       = baseline)
    
    mini  = Minimizer(partial_residuals, initialParams_KRData_half_Fourier_PhaseCurve)
    
    start = time()
    fitResults_KRData_half_Fourier_PhaseCurve = mini.leastsq()
    print("Full phase curve fitting operation took {} seconds".format(time()-start))
    
    print(report_errors(fitResults_KRData_half_Fourier_PhaseCurve.params))
    
    var_i_care = ['edepth', 'ecc', 'omega', 'amp1', 'amp2', 'amp3', 'amp4', 'amp5', 'amp6', 'amp7', 'amp8']
    ppm_or_not = [ppm, 1.0, 1.0, ppm, ppm, ppm,ppm, ppm, ppm, ppm, ppm]
    unit_list  = ['ppm', ' ', 'deg', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm']
    
    for varname in var_i_care:
        if varname in fitResults_KRData_half_Fourier_PhaseCurve.var_names:
            print('{}:\t{:.0f}\tppm'.format(varname, fitResults_KRData_half_Fourier_PhaseCurve.params[varname]*ppm))
    
    print('mean-1:\t{:.0f}\tppm'.format((fitResults_KRData_half_Fourier_PhaseCurve.params['mean']-1)*ppm))
    
    fig = figure(figsize=(20,20))
    ax  = fig.add_subplot(111)
    ax  = plot_model_over_reduced_lc_half_PLDsq_universal(times_c, 
                                                    phots_c / median(phots_c), 
                                                    sqrt(phots_c) / median(phots_c), 
                                                    # input_feature_set_c, 
                                                    [phots_c / median(phots_c), gw_c, nbr_c], 
                                                    in_1st_eclipse, 
                                                    # initialParams_KRData_half_Fourier_PhaseCurve, 
                                                    fitResults_KRData_half_Fourier_PhaseCurve.params,
                                                    # spider_params=spider_params_hoststar,
                                                    figsize=None, nbins=10000, plotRawData=False,
                                                    phase_method=phase_method, baseline=baseline,
                                                    ax = ax, returnAx = True)
    
    _, _, _, _, staticRad, varRad = flux_key_now.split('_')
    
    ax.set_title('{} - {} - {} - {}: {:.0f} ppm'.format(planet_params_name, channel.upper(), staticRad, varRad, fitResults_KRData_half_Fourier_PhaseCurve.params['edepth']*ppm));
    
    fig_save_name = planet_params_name.replace(' ','_') + '_full_phase_curve_' + channel + '_observations_' + phase_method + '_' + baseline + '_' + flux_key_now + '.png'
    
    axhline(1.0, ls='--')
    print('Saving figure to '  + 'figure_results/' + fig_save_name)
    fig.savefig('figure_results/' + channel + '/' + fig_save_name)
    
    ax.set_ylim(0.999,1.003)
    
    fig_save_name= planet_params_name.replace(' ','_') + '_top_full_phase_curve_' + channel + '_observations_' + phase_method + '_' + baseline + '_' + flux_key_now + '.png'
    
    print('Saving Figure to '  + 'figure_results/' + fig_save_name)
    fig.savefig('figure_results/' + channel + '/' + fig_save_name)
    
    joblib.dump(fitResults_KRData_half_Fourier_PhaseCurve, fits_result_save_name)
    
    print(flux_key_now + ' completed.')