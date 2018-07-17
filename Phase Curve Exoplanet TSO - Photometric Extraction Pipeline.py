from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument('-pn', '--planet_name', required=True, type=str, help=)
ap.add_argument('-c', '--channel', required=True, type=str, help=)
ap.add_argument('-ad', '--aor_dir', required=True, type=str, help=)
ap.add_argument('-pd', '--planets_dir', required=False, type=str, default='/Research/Planets/', help=)
ap.add_argument('-ds', '--data_sub_dir', required=False, type=str, default='/data/raw/', help=)
ap.add_argument('-dt', '--data_tail_dir', required=False, type=str, default='/big/', help=)
ap.add_argument('-ff', '--fits_format', required=False, type=str, default='bcd', help=)
ap.add_argument('-m', '--method', required=False, type=str, default='median', help=)
ap.add_argument('-t', '--telescope', required=False, type=str, default='Spitzer', help=)
ap.add_argument('-ou', '--outputUnits', required=False, type=str, default='electrons', help=)
ap.add_argument('-d', '--data_dir', required=False, type=str, default='', help='Set location of all `bcd` and `bunc` files: bypass previous setup')
args = vars(ap.parse_args())

planetName = args['planet_name']
channel = args['channel']
AORNow = args['aor_dir']
planetDirectory = args['planets_dir']
data_sub_dir = args['data_sub_dir']
data_tail_dir = args['data_tail_dir']
fits_format = args['fits_format']
method = args['method']
telescope = args['telescope']
outputUnits = args['outputUnits']

# from astroML.plotting          import hist
from astropy.io                import fits
from astropy.modeling          import models, fitting
from datetime                  import datetime
# from image_registration        import cross_correlation_shifts
from glob                      import glob
from functools                 import partial
# from matplotlib.ticker         import MaxNLocator
# from matplotlib                import style
from os                        import listdir
# from least_asymmetry.asym      import actr, moments, fitgaussian
from multiprocessing           import cpu_count, Pool
from numpy                     import min as npmin, max as npmax, zeros, arange, sum, float, isnan, hstack
from numpy                     import int32 as npint, round as npround, nansum as sum, nanstd as std
from os                        import environ, path, mkdir
from pandas                    import DataFrame, read_csv, read_pickle, scatter_matrix
from photutils                 import CircularAperture, CircularAnnulus, aperture_photometry, findstars
from pylab                     import sort, linspace, indices, median, mean, std, empty, transpose, ceil
from pylab                     import concatenate, pi, sqrt, ones, diag, inf, isnan, isfinite, array, nanmax
# from pylab                     import gcf, ion, figure, plot, imshow, scatter, legend, rcParams
# from seaborn                   import *
from scipy.special             import erf
from scipy                     import stats
from sklearn.externals         import joblib
from sklearn.preprocessing     import StandardScaler
from socket                    import gethostname
from statsmodels.robust        import scale
from statsmodels.nonparametric import kde
from sys                       import exit
from time                      import time, localtime
from tqdm                      import tqdm_notebook

from numpy                     import zeros, nanmedian as median, nanmean as mean, nan
from sys                       import exit
from sklearn.externals         import joblib

import numpy as np

startFull = time()


# **Master Class for Exoplanet Time Series Observation Photometry**

# In[6]:


from ExoplanetTSO_Auxiliary import wanderer


# In[7]:


def clipOutlier2D(arr2D, nSig=10):
    arr2D     = arr2D.copy()
    medArr2D  = median(arr2D,axis=0)
    sclArr2D  = np.sqrt(((scale.mad(arr2D)**2.).sum()))
    outliers  = abs(arr2D - medArr2D) >  nSig*sclArr2D
    inliers   = abs(arr2D - medArr2D) <= nSig*sclArr2D
    arr2D[outliers] = median(arr2D[inliers],axis=0)
    return arr2D


# In[8]:


# rcParams['figure.dpi'] = 150
# rcParams['image.interpolation'] = 'None'
# rcParams['image.cmap']          = 'Blues_r'
# rcParams['axes.grid']           = False


# As an example, Spitzer data is expected to be store in the directory structure:
# 
# `$HOME/PLANET_DIRECTORY/PLANETNAME/data/raw/AORDIR/CHANNEL/bcd/`
# 
# EXAMPLE:
# 
# 1. On a Linux machine
# 2. With user `tempuser`,
# 3. And all Spitzer data is store in `Research/Planets`
# 4. The planet named `Happy-5b`
# 5. Observed during AOR r11235813
# 6. In CH2 (4.5 microns)
# 
# The `loadfitsdir` should read as: `/home/tempuser/Research/Planets/HAPPY5/data/raw/r11235813/ch2/bcd/`

# In[28]:


from os import environ

# channel = 'ch2/'

dataSub = fits_format+'/'

dataDir     = environ['HOME'] + planetDirectory + planetName + data_sub_dir + channel + data_tail_dir

# AORs = []
# for dirNow in glob(dataDir + '/*'):
#     AORs.append(dirNow.split('/')[-1])

fileExt = '*{}.fits'.format(fits_format)
uncsExt = '*bunc.fits'

print('Current Data Dir'.format(dataDir))


# In[31]:


# len(AORs)


# In[35]:


# iAOR        = 0
loadfitsdir = dataDir + AORNow + '/' + channel + '/' + dataSub
print(loadfitsdir)


# In[36]:


nCores = cpu_count()


# In[37]:


fitsFilenames = glob(loadfitsdir + fileExt);print(len(fitsFilenames))
uncsFilenames = glob(loadfitsdir + uncsExt);print(len(uncsFilenames))

if len(fitsFilenames) == 0: raise ValueError('There are NO `bcd.fits` files in the directory {}'.format(loadfitsdir))
if len(uncsFilenames) == 0: raise ValueError('There are NO `bunc.fits` files in the directory {}'.format(loadfitsdir))


do_db_scan = len(fitsFilenames*64) < 1e5

header_test = fits.getheader(fitsFilenames[0])
print('\n\nAORLABEL:\t{}\nNum Fits Files:\t{}\nNum Unc Files:\t{}\n\n'.format(header_test['AORLABEL'], len(fitsFilenames), len(uncsFilenames)))

if verbose == 10: print(fitsFilenames)
if verbose == 10: print(uncsFilenames)
# # Load ExoplanetTSO Class

# Necessary Constants Spitzer
# ---

# In[39]:

ppm             = 1e6
y,x             = 0,1

yguess, xguess  = 15., 15.   # Specific to Spitzer circa 2010 and beyond
filetype        = 'bcd.fits' # Specific to Spitzer Basic Calibrated Data

print('Initialize an instance of `wanderer` as `example_wanderer_median`\n')
example_wanderer_median = wanderer(fitsFileDir=loadfitsdir, filetype=filetype, telescope=telescope, 
                                            yguess=yguess, xguess=xguess, method=method, nCores=nCores)

example_wanderer_median.AOR        = AORNow
example_wanderer_median.planetName = planetName
example_wanderer_median.channel    = channel


# In[41]:


print('Load Data From Fits Files in ' + loadfitsdir + '\n')
# exarymple_wanderer_median.load_data_from_fits_files()
example_wanderer_median.spitzer_load_fits_file(outputUnits=outputUnits)#(outputUnits='muJ_per_Pixel')


# In[42]:


print('Skipping Load Data From Save Files in ' + loadfitsdir + '\n')
# example_wanderer_median.load_data_from_save_files(savefiledir='./SaveFiles/', \
# saveFileNameHeader='Example_Wanderer_Median_', saveFileType='.pickle.save')


# **Double check for NaNs**

# In[43]:


example_wanderer_median.imageCube[np.where(isnan(example_wanderer_median.imageCube))] = np.nanmedian(example_wanderer_median.imageCube)


# **Identifier Strong Outliers**

# In[44]:


print('Find, flag, and NaN the "Bad Pixels" Outliers' + '\n')
example_wanderer_median.find_bad_pixels()


# In[45]:


print('Fit for All Centers: Flux Weighted, Gaussian Fitting, Gaussian Moments, Least Asymmetry' + '\n')
# example_wanderer_median.fit_gaussian_centering()
example_wanderer_median.fit_flux_weighted_centering()
# example_wanderer_median.fit_least_asymmetry_centering()
# example_wanderer_median.fit_all_centering() # calling this calls least_asymmetry, which does not work :(

start = time()
example_wanderer_median.mp_lmfit_gaussian_centering(subArraySize=6, recheckMethod=None, median_crop=False)
print('Operation took {} seconds with {} cores'.format(time()-start, example_wanderer_median.nCores))
#
# nSig       = 10.1
# medY       = median(example_wanderer_median.centering_GaussianFit.T[y])
# medX       = median(example_wanderer_median.centering_GaussianFit.T[x])
# stdY       = std(example_wanderer_median.centering_GaussianFit.T[y])
# stdX       = std(example_wanderer_median.centering_GaussianFit.T[x])
#
# ySig = 4
# xSig = 4
# outliers   = (((example_wanderer_median.centering_GaussianFit.T[y] - medY)/(ySig*stdY))**2 \
#             + ((example_wanderer_median.centering_GaussianFit.T[x] - medX)/(xSig*stdX))**2) > 1


# In[70]:


# ax = figure().add_subplot(111)
# cx, cy = example_wanderer_median.centering_GaussianFit.T[x],example_wanderer_median.centering_GaussianFit.T[y]
# ax.plot(cx,cy,'.',ms=1)
# ax.plot(cx[outliers],cy[outliers],'.',ms=1)
# # ax.plot(median(cx), median(cy),'ro',ms=1)
# ax.set_xlim(medX-nSig*stdX,medX+nSig*stdX)
# ax.set_ylim(medY-nSig*stdY,medY+nSig*stdY)


# In[71]:

if do_db_scan:
    from sklearn.cluster import DBSCAN

    dbs     = DBSCAN(n_jobs=-1, eps=0.2, leaf_size=10)
    dbsPred = dbs.fit_predict(example_wanderer_median.centering_GaussianFit)
    
    dbs_options = [k for k in range(-1,100) if (dbsPred==k).sum()]
else:
    dbsPred = None
    dbs_options = []


# fig = figure(figsize=(6,6))
# ax  = fig.add_subplot(111)
#
# medGaussCenters   = median(example_wanderer_median.centering_GaussianFit,axis=0)
# sclGaussCenters   = scale.mad(example_wanderer_median.centering_GaussianFit)
# sclGaussCenterAvg = np.sqrt(((sclGaussCenters**2.).sum()))
#
# yctrs = example_wanderer_median.centering_GaussianFit.T[y]
# xctrs = example_wanderer_median.centering_GaussianFit.T[x]
#
# nSigmas         = 5
# for nSig in linspace(1,10,10):
#     CircularAperture(medGaussCenters[::-1],nSig*sclGaussCenterAvg).plot(ax=ax)
#
# for dbsOpt in dbs_options:
#     ax.plot(xctrs[dbsPred==dbsOpt], yctrs[dbsPred==dbsOpt],'.',zorder=0, ms=1)


# In[80]:


npix = 3

stillOutliers = np.where(abs(example_wanderer_median.centering_GaussianFit - medGaussCenters) > 4*sclGaussCenterAvg)[0]
print('There are {} outliers remaining'.format(len(stillOutliers)))
# for o in stillOutliers:
#     figure()
#     imshow(example_wanderer_median.imageCube[o][16-npix:16+npix+1,16-npix:16+npix+1])

# In[81]:

if do_db_scan:
    dbsClean  = 0
    dbsKeep   = (dbsPred == dbsClean)

# **TEST**
# 
# Try column-wise background subtraction (and row-wise) to model the read pattern

# In[82]:

nCores = example_wanderer_median.nCores
start = time()
example_wanderer_median.mp_measure_background_circle_masked()
print('CircleBG took {} seconds with {} cores'.format(time() - start, nCores))
start = time()
example_wanderer_median.mp_measure_background_annular_mask()
print('AnnularBG took {} seconds with {} cores'.format(time() - start, nCores))
start = time()
example_wanderer_median.mp_measure_background_KDE_Mode()
print('KDEUnivBG took {} seconds with {} cores'.format(time() - start, nCores))
start = time()
example_wanderer_median.mp_measure_background_median_masked()
print('MedianBG took {} seconds with {} cores'.format(time() - start, nCores))

# fig = figure(figsize=(20,10))
# ax  = fig.add_subplot(111)
# ax.plot(example_wanderer_median.timeCube, example_wanderer_median.background_CircleMask,'.',alpha=0.2)
# ax.plot(example_wanderer_median.timeCube, example_wanderer_median.background_Annulus,'.',alpha=0.2)
# ax.plot(example_wanderer_median.timeCube, example_wanderer_median.background_MedianMask,'.',alpha=0.2)
# ax.plot(example_wanderer_median.timeCube, example_wanderer_median.background_KDEUniv,'.',alpha=0.2)
# ax.axvline(example_wanderer_median.timeCube.min()-.01+0.02)
# ax.set_ylim(-25,100)
# # ax.set_xlim(example_wanderer_median.timeCube.min()-.01,example_wanderer_median.timeCube.min() + .05)

example_wanderer_median.measure_effective_width()
print(example_wanderer_median.effective_widths.mean(), sqrt(example_wanderer_median.effective_widths).mean())

#
# vrad_dist = example_wanderer_median.quadrature_widths - np.median(example_wanderer_median.quadrature_widths)
# vrad_dist = clipOutlier2D(vrad_dist, nSig=5)
# ax = figure().add_subplot(111)
# ax.hist(vrad_dist, bins=example_wanderer_median.nFrames//100);
# ax.hist(0.75*vrad_dist, bins=example_wanderer_median.nFrames//100);
# ax.hist(0.5*vrad_dist, bins=example_wanderer_median.nFrames//100);
# ax.hist(0.25*vrad_dist, bins=example_wanderer_median.nFrames//100);
# # ax.set_xlim(-.25,.25);
#

# In[88]:


print('Pipeline took {} seconds thus far'.format(time() - startFull))


# In[89]:


print('Iterating over Background Techniques, Centering Techniques, Aperture Radii' + '\n')
centering_choices  = ['Gaussian_Fit']#, 'Gaussian_Mom', 'FluxWeighted']#, 'LeastAsymmetry']
background_choices = ['AnnularMask']#example_wanderer_median.background_df.columns
staticRads         = np.arange(1, 6,0.5)#[1.0 ]# aperRads = np.arange(1, 6,0.5)
varRads            = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50]#[None]# 

vrad_dist = example_wanderer_median.quadrature_widths - np.median(example_wanderer_median.quadrature_widths)
vrad_dist = clipOutlier2D(vrad_dist, nSig=5)

for staticRad in tqdm_notebook(staticRads, total=len(staticRads), desc='Static'):
    for varRad in tqdm_notebook(varRads, total=len(varRads), desc='Variable'):
        startMPFlux = time()
        example_wanderer_median.mp_compute_flux_over_time_varRad(staticRad, varRad, centering_choices[0], background_choices[0], useTheForce=True)
        
        # print('Flux Measurements took {} seconds for sRad {} and vRad {}'.format(time()-startMPFlux,staticRad,varRad))


# **Create Beta Variable Radius**
# Gaussian_Fit_AnnularMask_rad_betaRad_0.0_0.0

example_wanderer_median.mp_compute_flux_over_time_betaRad()

print('Entire Pipeline took {} seconds'.format(time() - startFull))

if do_db_scan: example_wanderer_median.mp_DBScan_Flux_All()

inlier_master = array(list(example_wanderer_median.inliers_Phots.values())).mean(axis=0) == 1.0



example_wanderer_median.extract_PLD_components()

if do_db_scan: example_wanderer_median.mp_DBScan_PLD_All()

print('Saving `example_wanderer_median` to a set of pickles for various Image Cubes and the Storage Dictionary')

savefiledir         = environ['HOME']+'/Research/Planets/'+planetName+'/ExtracedData/' + channel 
saveFileNameHeader  = planetName+'_'+ AORNow +'_Median'
saveFileType        = '.joblib.save'

if not path.exists(environ['HOME']+'/Research/Planets/'+planetName+'/ExtracedData/'):
    mkdir(environ['HOME']+'/Research/Planets/'+planetName+'/ExtracedData/')

if not path.exists(savefiledir):
    print('Creating ' + savefiledir)
    mkdir(savefiledir)

print()
print('Saving to ' + savefiledir + saveFileNameHeader + saveFileType)
print()

example_wanderer_median.save_data_to_save_files(savefiledir=savefiledir, saveFileNameHeader=saveFileNameHeader, saveFileType=saveFileType)

#
# quad_width= example_wanderer_median.quadrature_widths.values
# vrad_dist = quad_width - np.median(quad_width)
# vrad_dist = clipOutlier2D(vrad_dist, nSig=5)
# vrad_dist_med = np.median(vrad_dist)
#
# color_cycle = rcParams['axes.prop_cycle'].by_key()['color']
#
# ax = figure().add_subplot(111)
# for key in example_wanderer_median.flux_TSO_df.keys():
#     staticRad = float(key.split('_')[-2])
#     varRad    = float(key.split('_')[-1])
#     aperRad   = staticRad + varRad*vrad_dist_med
#     colorNow  = color_cycle[int(varRad*4)]
#     # if aperRad > 1.5 and aperRad < 3.5:
#     ax.scatter(aperRad, scale.mad(np.diff(example_wanderer_median.flux_TSO_df[key])),                    color=colorNow, zorder=int(varRad*4))
#
# for varRad in [0.,0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
#     colorNow  = color_cycle[int(varRad*4)]
#     ax.scatter([],[], color=colorNow, label=varRad)
#
# ax.set_xlabel('StaticRad + Average(varRad)')
# ax.set_ylabel('MAD( Diff ( Flux ) )')
# ax.legend(loc=0)
# ax.set_ylim(292.5,294)
# ax.set_xlim(2.9, 3.1)
# gcf().savefig('')

# In[131]:


print('Entire Pipeline took {} seconds'.format(time() - startFull))

