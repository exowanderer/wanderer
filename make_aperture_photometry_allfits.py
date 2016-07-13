%cpaste
%colors Linux
from astroML.plotting   import hist
from astropy.modeling   import models, fitting
from astropy.io         import fits
from astropy.modeling   import models, fitting
from datetime           import datetime
from image_registration import cross_correlation_shifts
from matplotlib.ticker  import MaxNLocator
from matplotlib         import style
from matplotlib         import pyplot as plt
from os                 import listdir
from photutils          import CircularAperture, aperture_photometry
from pylab              import ion, gcf, sort, linspace, indices, median, mean, std, empty, concatenate, pi, sqrt, ones, diag, inf, rcParams, array, get_current_fig_manager, twinx
from numpy              import min as npmin, max as npmax, zeros, empty, ones, where, arange
from scipy.special      import erf
from scipy import stats
from sklearn.externals import joblib
# from seaborn            import *
from socket             import gethostname
from statsmodels.robust import scale
from sys                import exit, stdout
from time               import time

ion()
style.use('fivethirtyeight')

''' UNCOMMENT TO LOAD PICKLEOUT '''
'''
pickleDir                       = 'pickleSaves/'
pickleFileName                  = 'NIRCam_CV3_DefaultFlags_InitialRunThrough.pkl'

pickleOut                       = joblib.load(pickleDir + pickleFileName) 

for key in pickleOut.keys():
    exec("{0:s} = pickleOut['{1:s}']".format(key,key))
'''

def fit_ols_statsmodels(ydata,xdata):
    import statsmodels.api as sm # version 0.5 
    X           = sm.add_constant(xdata)
    result      = sm.OLS(ydata, X).fit()
    theta_hat   = result.params
    Sigma       = result.cov_params()
    return result, Sigma, sigma_hat

def print_arbitrary(arb_string):
    ### dumps `arb_string` to stdout ###
    print >> stdout, str(arb_string), "\r",
    stdout.flush()

rcParams['axes.grid']           = False
rcParams['lines.markersize']    = 3

nRamps  = 20    # assumed for all WLP integrations

redfitsdir  = 'reduced_orig_flags/redfits/'
if 'surtr' in gethostname().lower():
    redfitsdir  = 'reduced_orig_flags/redfits/'
elif 'sydney' in gethostname().lower():
    redfitsdir  = 'reduced_orig_flags/redfits/'
elif 'aerith' in gethostname().lower():
    redfitsdir  = 'reduced_orig_flags/redfits/'
else:
    exit('Who Am I?? Where Am I??')

redfitsdirfiles = listdir(redfitsdir)
redfitsfiles    = {}
for fname in redfitsdirfiles:
    if not '.fits' in fname:
        redfitsfiles[fname] = listdir(redfitsdir + fname)

# wlp_set             = {}
# psf_set             = {}
OpenWLPfits         = False
OpenPSFfits         = True
# redPSFfitshdudict   = {}
# redWLPfitshdudict   = {}

time_fits       = {}
diff_fits       = {}
# origfitsfiles   = {}
# for key0 in redfitshdudict.keys():
#     if not 'wlp' in key0.lower():
#         diff_fits[key0]   = {}
#         for key1 in redfitshdudict[key0].keys():
#             print_arbitrary(key0 + '; ' + key1)
#             diff_fits[key0][key1] = redfitshdudict[key0][key1].data[-1] - redfitshdudict[key0][key1].data[0]

def get_julian_date_from_header(header):
    from jd import julian_date
    fitsDate    = hduNow[0].header['DATE-OBS']
    startTimeStr= hduNow[0].header['TIME-OBS']
    endTimeStr  = hduNow[0].header['TIME-END']

    yy,mm,dd    = fitsDate.split('-')

    hh1,mn1,ss1 = array(startTimeStr.split(':')).astype(float)
    hh2,mn2,ss2 = array(endTimeStr.split(':')).astype(float)

    # hh1_dec     = hh1*3600. + mn1*60. + ss1
    # hh2_dec     = hh2*3600. + mn2*60. + ss2

    # ### Modify the global fits time stamp to represent the individual file time stamps
    # hh1_dec     = hh1_dec * float(k) / float(nFits)
    # hh2_dec     = hh2_dec * float(k) / float(nFits)

    # hh1         = hh1_dec % 3600
    # hh2         = hh2_dec % 3600

    # mn1         = (hh1_dec - hh1) % 60
    # mn2         = (hh2_dec - hh2) % 60

    # ss1         = (hh1_dec - hh1 - mn1*60)
    # ss2         = (hh2_dec - hh2 - mn2*60)

    startDate   = julian_date(yy,mm,dd,hh1,mn1,ss1)
    endDate     = julian_date(yy,mm,dd,hh2,mn2,ss2)

    return startDate, endDate

day2sec = 86400.
for key0 in sort(redfitsfiles.keys()):
    if not 'wlp' in key0.lower() and OpenPSFfits:
        diff_fits[key0] = {}
        time_fits[key0] = {}
        nGroupsBig      = len(redfitsfiles[key0])
        for k, fname in enumerate(sort(redfitsfiles[key0])):
            key1                            = fname.split('_')[-1].split('.')[0]
            hduNow                          = fits.open(redfitsdir + key0 + '/' + fname)
            startJD, endJD                  = get_julian_date_from_header(hduNow[0].header)
            diff_fits[key0][key1]           = hduNow[0].data[-1] - hduNow[0].data[0]
            timeSpan                        = (endJD - startJD)*day2sec/nGroupsBig
            #timeFitsNow                     = 0.5*(endJD + startJD)
            # timeFitsWholeDay                = int(timeFits)
            # timeFitsFracDay                 = timeFits - timeFitsWholeDay
            # timeFitsFracDay                 = timeFitsFracDay * float(k) / float(nFits)
            # time_fits[key0][key1]           = float(timeFitsWholeDay) + timeFitsFracDay
            time_fits[key0][key1]           = startJD  + timeSpan*(k+0.5) / day2sec
            del hduNow[0].data
            hduNow.close()
            del hduNow
            print 'Opening ' + fname + ' in directory ' + redfitsdir + key0 + '/'
    #
    elif 'wlp' in key0.lower() and OpenWLPfits:
        redWLPfitshdudict[key0] = {}
        for fname in sort(redfitsfiles[key0]):
            key1                            = fname.split('_')[-1].split('.')[0]
            hduNow                          = fits.open(redfitsdir + key0 + '/' + fname)
            diff_fits[key0][key1]           = hduNow[0].data[-1] - hduNow[0].data[0]
            startJD, endJD                  = get_julian_date_from_header(hduNow[0].header)
            time_fits[key0][key1]           = 0.5*(endJD + startJD)
            del hduNow[0].data
            hduNow.close()
            del hduNow
            print 'Opening ' + fname + ' in directory ' + redfitsdir + key0 + '/'
--
# for key in redPSFfitshdudict.keys():
#     redfitshdudict[key] = redPSFfitshdudict[key]

# for key in redWLPfitshdudict.keys():
#     redfitshdudict[key] = redWLPfitshdudict[key]

# origfitsfiles   = {}
# for key in sort(redfitshdudict.keys()):
#     if not key in origfitsfiles.keys():
#         origfitsfiles[key] = fits.open(redfitsdir + key + '.fits')[0]

print 'Creating Summation Differential Fits'

# diff_fits = {}
# for key0 in redfitshdudict.keys():
#     if not 'wlp' in key0.lower():
#         diff_fits[key0]   = {}
#         for key1 in redfitshdudict[key0].keys():
#             print_arbitrary(key0 + '; ' + key1)
#             diff_fits[key0][key1] = redfitshdudict[key0][key1].data[-1] - redfitshdudict[key0][key1].data[0]

### Sky Background and Gaussian Fitting ###
print 'Computing Sky Background and Gaussian Fitting'
%cpaste
y,x     = 0,1
amp0    = 1.0

key0    = diff_fits.keys()[0]
key1    = diff_fits[key0].keys()[0]
xmean0  = diff_fits[key0][key1].shape[x]/2
ymean0  = diff_fits[key0][key1].shape[y]/2
xstd0   = 1.0
ystd0   = 1.0
theta0  = 0.0
nSig    = 10.0

nRads   = 100
minRad  = 0.1
maxRad  = 10
radii   = linspace(minRad, maxRad, nRads)
fit_lvmq= fitting.LevMarLSQFitter()
maxOutSt= 0

midFrame        = 160
radSubFrame     = 20
xlower, xupper  = midFrame - radSubFrame, midFrame + radSubFrame
ylower, yupper  = midFrame - radSubFrame, midFrame + radSubFrame
subFrameLower   = array([ylower, xlower])
imgCenter       = array([xmean0, ymean0])
gaussParamStr   = ['amp', 'xc', 'yc', 'xs', 'ys', 'th']

gaussfits   = {}
skyBG       = {}
crossCorr   = {}
phot_arr_GS = {}
phot_arr_CC = {}

yinds,xinds = indices(diff_fits[key0][key1].shape)
gaussOrig   = models.Gaussian2D(amplitude=amp0, x_mean=xmean0, y_mean=ymean0, x_stddev=xstd0, y_stddev=ystd0, theta=theta0)
gaussOrigArr= gaussOrig(xinds,yinds)

### Sky Background Subtraction, Gaussian center, and Cross Correlation Centering  ###
for key0 in sort(diff_fits.keys()):
    if not 'wlp' in key0.lower():
        gaussfits[key0]     = {}
        skyBG[key0]         = {}
        crossCorr[key0]     = {}
        for key1 in sort(diff_fits[key0].keys()):
            med0                    = median(diff_fits[key0][key1].ravel())
            mad0                    = scale.mad(diff_fits[key0][key1].ravel())
            skyBG[key0][key1]       = median(diff_fits[key0][key1].ravel()[abs(diff_fits[key0][key1].ravel() - med0) < nSig*mad0])
            crossCorr[key0][key1]   = cross_correlation_shifts(gaussOrigArr / gaussOrigArr.max() * diff_fits[key0][key1].max(), diff_fits[key0][key1])
            crossCorr[key0][key1]   = crossCorr[key0][key1] + imgCenter
            #
            gaussGuess              = models.Gaussian2D(amplitude=diff_fits[key0][key1].max(), 
                                                            x_mean  =crossCorr[key0][key1][0], y_mean   =crossCorr[key0][key1][1], 
                                                            x_stddev=xstd0                   , y_stddev =ystd0                   , 
                                                            theta   =theta0)
            #
            # print '*** REMEMBER THE YOU CHANGED HOW GAUSSFIT WORKS WITHOUT TESTING IT 2-26-16 ***'
            gaussfits[key0][key1]               = {}
            gaussfits[key0][key1]['results']    = fit_lvmq(gaussGuess, xinds[ylower:yupper,xlower:xupper], yinds[ylower:yupper,xlower:xupper], \
                                                    diff_fits[key0][key1][ylower:yupper,xlower:xupper] - skyBG[key0][key1], \
                                                    weights=None, maxiter=1000, acc=1e-07, epsilon=1.4901161193847656e-08, estimate_jacobian=False)
            gaussfits[key0][key1]['param_cov']  = fit_lvmq.fit_info['param_cov']
            print gaussfits[key0][key1]['results'].parameters[:3], gaussfits[key0][key1]['results'].parameters[3:] # 3 = 1/2 * len(gaussfits[key0][key1]['results'].parameters)
--

%cpaste
### Aperture Photometry ###
for key0 in sort(gaussfits.keys()):
    phot_arr_GS[key0]  = {}
    phot_arr_CC[key0]  = {}
    for key1 in sort(gaussfits[key0].keys()):
        phot_arr_GS[key0][key1] = empty(len(radii))
        phot_arr_CC[key0][key1] = empty(len(radii))
        for k, radius in enumerate(radii):
            fluxNowGauss    = aperture_photometry(diff_fits[key0][key1][ylower:yupper,xlower:xupper] - skyBG[key0][key1], 
                                        CircularAperture(gaussfits[key0][key1]['results'].parameters[1:3] - subFrameLower, radius), method='exact')
            #
            fluxNowCCorr    = aperture_photometry(diff_fits[key0][key1][ylower:yupper,xlower:xupper] - skyBG[key0][key1], 
                                        CircularAperture(crossCorr[key0][key1]                 - subFrameLower, radius), method='exact')                
            #
            phot_arr_GS[key0][key1][k]  = fluxNowGauss['aperture_sum'].data[0]
            phot_arr_CC[key0][key1][k]  = fluxNowCCorr['aperture_sum'].data[0]
            #
            outputString    = 'Processing step '+ key0.split('-')[0] + '; ' + key1           + \
                                '; ' + str(k+1) + ' out of ' + str(radii.size)               + \
                                ' with radius ' + str(radius)                                + \
                                '; photsGS = '  + str("{:.3f}".format(phot_arr_GS[key0][key1][k]))            + \
                                '; photsCC = '  + str("{:.3f}".format(phot_arr_CC[key0][key1][k]))            + \
                                '; ycenter = '  + str("{:.3f}".format(gaussfits[key0][key1]['results'].parameters[y+1])) + \
                                '; xcenter = '  + str("{:.3f}".format(gaussfits[key0][key1]['results'].parameters[x+1]))
            #
            # maxOutSt = max([maxOutSt, len(outputString)])
            # print_arbitrary(' '*maxOutSt)
            print_arbitrary(outputString)
--

%cpaste
### Convert  Dict of Dicts to Dict x Arrays ###
skyBGArr        = {}
gaussfitsArr    = {}
crossCorrArr    = {}
phot_arr_GSArr  = {}
phot_arr_CCArr  = {}
times_arr       = {}
for key0 in gaussfits.keys():
    times_arr[key0]         = empty(len(time_fits[key0].keys()))
    skyBGArr[key0]          = empty(len(skyBG[key0].keys()))
    gaussfitsArr[key0]      = empty((len(gaussfits[key0].keys()),6))
    crossCorrArr[key0]      = empty((len(crossCorr[key0].keys()),2))
    phot_arr_GSArr[key0]    = empty((len(phot_arr_GS[key0].keys()), nRads))
    phot_arr_CCArr[key0]    = empty((len(phot_arr_CC[key0].keys()), nRads))
    for k, key1 in enumerate(gaussfits[key0].keys()):
        skyBGArr[key0][k]       = skyBG[key0][key1]
        gaussfitsArr[key0][k]   = gaussfits[key0][key1]['results'].parameters
        crossCorrArr[key0][k]   = crossCorr[key0][key1]
        phot_arr_GSArr[key0][k] = phot_arr_GS[key0][key1]
        phot_arr_CCArr[key0][k] = phot_arr_CC[key0][key1]
        times_arr[key0][k]      = time_fits[key0][key1]

times_arr_481       = times_arr[sort(times_arr.keys())[0]]
times_arr_489       = times_arr[sort(times_arr.keys())[1]]

phot_arr_GSArr_481  = phot_arr_GSArr[sort(phot_arr_GSArr.keys())[0]]
phot_arr_GSArr_489  = phot_arr_GSArr[sort(phot_arr_GSArr.keys())[1]]
phot_arr_CCArr_481  = phot_arr_CCArr[sort(phot_arr_CCArr.keys())[0]]
phot_arr_CCArr_489  = phot_arr_CCArr[sort(phot_arr_CCArr.keys())[1]]
#
gaussfitsArr_481    = gaussfitsArr[sort(gaussfitsArr.keys())[0]]
gaussfitsArr_489    = gaussfitsArr[sort(gaussfitsArr.keys())[1]]
#
crossCorrArr_481    = crossCorrArr[sort(crossCorrArr.keys())[0]]
crossCorrArr_489    = crossCorrArr[sort(crossCorrArr.keys())[1]]
#
skyBGArr_481        = skyBGArr[sort(skyBGArr.keys())[0]]
skyBGArr_489        = skyBGArr[sort(skyBGArr.keys())[1]]
#
for key0 in sort(phot_arr_CCArr.keys())[2:]:
    if '481' in key0:
        times_arr_481       = concatenate((times_arr_481     , times_arr[key0]      ))
        phot_arr_GSArr_481  = concatenate((phot_arr_GSArr_481, phot_arr_GSArr[key0] ))
        phot_arr_CCArr_481  = concatenate((phot_arr_CCArr_481, phot_arr_CCArr[key0] ))
        gaussfitsArr_481    = concatenate((gaussfitsArr_481  , gaussfitsArr[key0]   ))
        crossCorrArr_481    = concatenate((crossCorrArr_481  , crossCorrArr[key0]   ))
        skyBGArr_481        = concatenate((skyBGArr_481      , skyBGArr[key0]       ))
    if '489' in key0:
        times_arr_489       = concatenate((times_arr_489     , times_arr[key0]      ))
        phot_arr_GSArr_489  = concatenate((phot_arr_GSArr_489, phot_arr_GSArr[key0]))
        phot_arr_CCArr_489  = concatenate((phot_arr_CCArr_489, phot_arr_CCArr[key0]))
        gaussfitsArr_489    = concatenate((gaussfitsArr_489  , gaussfitsArr[key0]   ))
        crossCorrArr_489    = concatenate((crossCorrArr_489  , crossCorrArr[key0]   ))
        skyBGArr_489        = concatenate((skyBGArr_489      , skyBGArr[key0]       ))
--

print median(phot_arr_CCArr_481 / median(phot_arr_CCArr_481, axis=0)), std(phot_arr_CCArr_481 / median(phot_arr_CCArr_481, axis=0))*100
print median(phot_arr_GSArr_481 / median(phot_arr_GSArr_481, axis=0)), std(phot_arr_GSArr_481 / median(phot_arr_GSArr_481, axis=0))*100
print median(phot_arr_CCArr_489 / median(phot_arr_CCArr_489, axis=0)), std(phot_arr_CCArr_489 / median(phot_arr_CCArr_489, axis=0))*100
print median(phot_arr_GSArr_489 / median(phot_arr_GSArr_489, axis=0)), std(phot_arr_GSArr_489 / median(phot_arr_GSArr_489, axis=0))*100

phot_arr_CCArr_481_normed       = phot_arr_CCArr_481 / median(phot_arr_CCArr_481, axis=0)
phot_arr_GSArr_481_normed       = phot_arr_GSArr_481 / median(phot_arr_GSArr_481, axis=0)
phot_arr_CCArr_489_normed       = phot_arr_CCArr_489 / median(phot_arr_CCArr_489, axis=0)
phot_arr_GSArr_489_normed       = phot_arr_GSArr_489 / median(phot_arr_GSArr_489, axis=0)

phot_arr_CCArr_481_normed_unc   = 1.0/sqrt(phot_arr_CCArr_481)
phot_arr_GSArr_481_normed_unc   = 1.0/sqrt(phot_arr_GSArr_481)
phot_arr_CCArr_489_normed_unc   = 1.0/sqrt(phot_arr_CCArr_489)
phot_arr_GSArr_489_normed_unc   = 1.0/sqrt(phot_arr_GSArr_489)

plt.subplot(211)
plt.plot((times_arr_481 - times_arr_481.min())*day2sec / 60., phot_arr_GSArr_481_normed, 'o', alpha=0.5)
plt.subplot(212)
plt.plot((times_arr_489 - times_arr_489.min())*day2sec / 60., phot_arr_GSArr_489_normed, 'o', alpha=0.5)
%cpaste
def make_gaussian_data_figure(xdata, ydata, zdata=None, nPts=1e3, width=None, amp=None):
    if xdata.ndim > 1 or ydata.ndim > 1:
        exit('xdata.ndim > 1 or ydata.ndim > 1');
    if xdata.size != ydata.size:
        exit('xdata.size != ydata.size');
    if width is None:
        # print '*** JUST guessing that xwidth = std(xdata)/10 AND ywidth = std(ydata)/10 ***'
        xwidth  = std(xdata) / 10.;
        ywidth  = std(ydata) / 10.;
    else:
        if hasattr(width, '__iter__'):
            if len(width) == 2:
                ywidth, xwidth  = width;
            else:
                exit('`width` must be either a float or a set of 2 float');
        else:
            ywidth = xwidth = width;
    if zdata is None:
        zdata   = ones(ydata.size) / sqrt((2*pi)*(xwidth**2. + ywidth**2.));
    elif zdata.ndim > 1:
        exit('zdata.ndim > 1');
    elif ydata.size != zdata.size:
        exit('xdata.size == ydata.size != zdata.size');
    nPts        = int(nPts);
    zero        = 0.0;
    # print '*** JUST guessing that nPts should be 1e3; maybe compute with nyquist sampling ***'
    gaussianDataPoints  = zeros((nPts, nPts));
    # print '*** JUST guessing that gridFactor  = 1.25; maybe compute with nyquist sampling ***'
    gridFactor  = 1.50;
    yGridWidth  = ydata.max() - ydata.min();
    xGridWidth  = xdata.max() - xdata.min();
    yGridWidth *= gridFactor;
    xGridWidth *= gridFactor;
    yinds, xinds= indices((gaussianDataPoints.shape));
    yinds       = (yinds.astype(float) / yinds.astype(float).max() - 0.5) * yGridWidth + median(ydata);
    xinds       = (xinds.astype(float) / xinds.astype(float).max() - 0.5) * xGridWidth + median(xdata);
    #print '***', (ydata >= yinds.min()).all(), (ydata <= yinds.max()).all(), (xdata >= xinds.min()).all(), (xdata <= xinds.max()).all(), '***'
    for k in range(ydata.size):
        # if not (xinds.min() < xdata[k] and xdata[k] < xinds.max() or yinds.min() < ydata[k] and ydata[k] < yinds.max()):
        #     print k, xinds.min(), xdata[k], xinds.max(), yinds.min(), ydata[k], yinds.max()
        gaussianDataPoints += models.Gaussian2D(amplitude=1.0 , x_mean=xdata[k], y_mean=ydata[k], x_stddev =xwidth   , y_stddev=ywidth  , theta =zero    )(xinds, yinds);
    return gaussianDataPoints

def plot_kde_like_figure():
    GCydata = gaussfitsArr_481[:,2] - median(gaussfitsArr_481[:,2])
    GCxdata = gaussfitsArr_481[:,1] - median(gaussfitsArr_481[:,1])
    Flux    = phot_arr_GSArr_481_normed[:,where(radii == 2.0)[0][0]] # where(radii == 2.0)[0][0] == 19
    zdata   = ones(Flux.size)

    nData   = 100
    nGroups = GCydata.size / nData
    gc_corr_flux_image= []
    for k in range(nGroups):
        gc_corr_flux_image.append(make_gaussian_data_figure(GCydata[k*nData:(k+1)*nData], Flux[k*nData:(k+1)*nData], zdata[k*nData:(k+1)*nData], nPts = 1e3, width=None, amp=None))
        gc_corr_flux_image.append(make_gaussian_data_figure(GCxdata[k*nData:(k+1)*nData], Flux[k*nData:(k+1)*nData], zdata[k*nData:(k+1)*nData], nPts = 1e3, width=None, amp=None))

    for k in range(len(gc_corr_flux_image)):
        figure()
        imshow(gc_corr_flux_image[k])
        title(str(['Y', 'X'][k%2]) + '-Center vs Flux Group ' + str(k))


    # clf()
    imshow(gc_corr_flux_image)

%cpaste
### Calculate and Plot Line Fit 481###
linemodel = models.Linear1D(slope=0, intercept=1)

chisqY_481          = empty((6, len(radii)))
chisqX_481          = empty((6, len(radii)))

slopesX_481         = empty((6, len(radii)))
slopesX_481_unc     = empty((6, len(radii)))
slopesY_481         = empty((6, len(radii)))
slopesY_481_unc     = empty((6, len(radii)))

interceptsX_481     = empty((6, len(radii)))
interceptsX_481_unc = empty((6, len(radii)))
interceptsY_481     = empty((6, len(radii)))
interceptsY_481_unc = empty((6, len(radii)))

med_slopes_481  = zeros((nGroups,2))
min_slopes_481  = zeros((nGroups,2))
max_slopes_481  = zeros((nGroups,2))
linefits_481    = {}
#
x,y = 1,2
for g in range(nGroups):
    linefits_481['group'+str(g)]    = dict(x=[],y=[])
    for r in range(len(radii)):
        xdata   = gaussfitsArr_481[g*nData:(g+1)*nData,x] - median(gaussfitsArr_481[g*nData:(g+1)*nData,x])
        ydata   = gaussfitsArr_481[g*nData:(g+1)*nData,y] - median(gaussfitsArr_481[g*nData:(g+1)*nData,y])
        fdata   = phot_arr_GSArr_481_normed[g*nData:(g+1)*nData,r]
        #
        lineNow     = fit_lvmq(linemodel, xdata, fdata)
        #
        linefits_481['group'+str(g)]['x'].append(lineNow)
        slopesX_481_unc[g][r], interceptsX_481_unc[g][r]  = sqrt(diag(fit_lvmq.fit_info['param_cov']))
        slopesX_481[g][r]    , interceptsX_481[g][r]      = lineNow.parameters
        #
        modelNow            = lineNow.evaluate(xdata, lineNow.slope, lineNow.intercept)
        chisqX_481[g][r]    = (((modelNow - fdata) / fdata.std())**2).sum()
        #
        lineNow     = fit_lvmq(linemodel, ydata, fdata)
        #
        linefits_481['group'+str(g)]['y'].append(lineNow)
        slopesY_481_unc[g][r], interceptsY_481_unc[g][r]  = sqrt(diag(fit_lvmq.fit_info['param_cov']))
        slopesY_481[g][r]    , interceptsY_481[g][r]      = lineNow.parameters
        #
        modelNow            = lineNow.evaluate(ydata, lineNow.slope, lineNow.intercept)
        chisqY_481[g][r]    = (((modelNow - fdata) / fdata.std())**2).sum()
    #
    med_slopes_481[g]   = median(slopesY_481[g]), median(slopesX_481[g])
    min_slopes_481[g]   = npmin(slopesY_481[g]), npmin(slopesX_481[g])
    max_slopes_481[g]   = npmax(slopesY_481[g]), npmax(slopesX_481[g])
--

%cpaste
### Calculate and Plot Line Fit 489###
chisqY_489          = empty((6, len(radii)))
chisqX_489          = empty((6, len(radii)))

slopesX_489         = empty((6, len(radii)))
slopesX_489_unc     = empty((6, len(radii)))
slopesY_489         = empty((6, len(radii)))
slopesY_489_unc     = empty((6, len(radii)))

interceptsX_489     = empty((6, len(radii)))
interceptsX_489_unc = empty((6, len(radii)))
interceptsY_489     = empty((6, len(radii)))
interceptsY_489_unc = empty((6, len(radii)))

med_slopes_489  = zeros((nGroups,2))
min_slopes_489  = zeros((nGroups,2))
max_slopes_489  = zeros((nGroups,2))
linefits_489    = {}
#
x,y = 1,2
for g in range(nGroups):
    linefits_489['group'+str(g)]    = dict(x=[],y=[])
    for r in range(len(radii)):
        xdata   = gaussfitsArr_489[g*nData:(g+1)*nData,x] - median(gaussfitsArr_489[g*nData:(g+1)*nData,x])
        ydata   = gaussfitsArr_489[g*nData:(g+1)*nData,y] - median(gaussfitsArr_489[g*nData:(g+1)*nData,y])
        fdata   = phot_arr_GSArr_489_normed[g*nData:(g+1)*nData,r]
        #
        lineNow = fit_lvmq(linemodel, xdata, fdata)
        linefits_489['group'+str(g)]['x'].append(lineNow)
        slopesX_489_unc[g][r], interceptsX_489_unc[g][r]  = sqrt(diag(fit_lvmq.fit_info['param_cov']))
        slopesX_489[g][r]    , interceptsX_489[g][r]      = lineNow.parameters
        #
        modelNow            = lineNow.evaluate(xdata, lineNow.slope, lineNow.intercept)
        chisqX_489[g][r]    = (((modelNow - fdata) / fdata.std())**2).sum()
        #
        lineNow = fit_lvmq(linemodel, ydata, fdata)
        linefits_489['group'+str(g)]['y'].append(lineNow)
        slopesY_489_unc[g][r], interceptsY_489_unc[g][r]  = sqrt(diag(fit_lvmq.fit_info['param_cov']))
        slopesY_489[g][r]    , interceptsY_489[g][r]      = lineNow.parameters
        #
        modelNow            = lineNow.evaluate(ydata, lineNow.slope, lineNow.intercept)
        chisqY_489[g][r]    = (((modelNow - fdata) / fdata.std())**2).sum()
    #
    med_slopes_489[g]   = median(slopesY_489[g][radii>2.0]), median(slopesX_489[g])
    min_slopes_489[g]   = npmin(slopesY_489[g]), npmin(slopesX_489[g])
    max_slopes_489[g]   = npmax(slopesY_489[g]), npmax(slopesX_489[g])
--

pvaluesX_481    = empty((nGroups, nRads, 2))
pvaluesY_481    = empty((nGroups, nRads, 2))
pvaluesX_489    = empty((nGroups, nRads, 2))
pvaluesY_489    = empty((nGroups, nRads, 2))

tvaluesX_481    = empty((nGroups, nRads))
tvaluesY_481    = empty((nGroups, nRads))
tvaluesX_489    = empty((nGroups, nRads))
tvaluesY_489    = empty((nGroups, nRads))

for g in range(nGroups):
    for r in range(nRads): 
        pvaluesX_481[g,r] = stats.pearsonr(gaussfitsArr_481[g*nData:(g+1)*nData,x], phot_arr_GSArr_481[g*nData:(g+1)*nData,r])
        pvaluesY_481[g,r] = stats.pearsonr(gaussfitsArr_481[g*nData:(g+1)*nData,y], phot_arr_GSArr_481[g*nData:(g+1)*nData,r])
        pvaluesX_489[g,r] = stats.pearsonr(gaussfitsArr_489[g*nData:(g+1)*nData,x], phot_arr_GSArr_489[g*nData:(g+1)*nData,r])
        pvaluesY_489[g,r] = stats.pearsonr(gaussfitsArr_489[g*nData:(g+1)*nData,y], phot_arr_GSArr_489[g*nData:(g+1)*nData,r])
        #
        tvaluesX_481[g,r] = stats.t.cdf(pvaluesX_481[g,r,0] * sqrt((nData-2) / (1-pvaluesX_481[g,r,0]**2.)), nData)
        tvaluesY_481[g,r] = stats.t.cdf(pvaluesY_481[g,r,0] * sqrt((nData-2) / (1-pvaluesY_481[g,r,0]**2.)), nData)
        tvaluesX_489[g,r] = stats.t.cdf(pvaluesX_489[g,r,0] * sqrt((nData-2) / (1-pvaluesX_489[g,r,0]**2.)), nData)
        tvaluesY_489[g,r] = stats.t.cdf(pvaluesY_489[g,r,0] * sqrt((nData-2) / (1-pvaluesY_489[g,r,0]**2.)), nData)

bbox_props  = dict(boxstyle='larrow,pad=0.0', facecolor='white', edgecolor='black', linewidth=1)
for g in range(nGroups):
    plt.plot(radii[24:], abs(tvaluesY_489[g][24:]), 'o-')

plt.axhline(0.01, linestyle='--', c='k')
plt.axhline(0.05, linestyle='--', c='k')
plt.axhline(0.1, linestyle='--', c='k')

plt.annotate(' 1%', (9.5, 0.01), xytext=(10.0, 0.01), bbox=bbox_props, fontsize=4)
plt.annotate(' 5%', (9.5, 0.05), xytext=(10.0, 0.05), bbox=bbox_props, fontsize=4)
plt.annotate('10%', (9.5, 0.10), xytext=(10.0, 0.10), bbox=bbox_props, fontsize=4)
fig.canvas.draw()

plt.title('Pearson Correlation Coefficient between Flux and Y-Center Mod B', fontsize=13)
plt.xlabel('Aperture Radius')
plt.ylabel('Pearson T-Statistic')

rstart  = 24
plt.figure(1)
plt.imshow(abs(pvaluesX_481[:,rstart:,0])*100, extent=[radii[rstart],radii.max(),1,6], vmax=5)
plt.colorbar()
plt.figure(2)
plt.imshow(abs(pvaluesY_481[:,rstart:,0])*100, extent=[radii[rstart],radii.max(),1,6], vmax=5)
plt.colorbar()
plt.figure(3)
plt.imshow(abs(pvaluesX_489[:,rstart:,0])*100, extent=[radii[rstart],radii.max(),1,6], vmax=5)
plt.colorbar()
plt.figure(4)
plt.imshow(abs(pvaluesY_489[:,rstart:,0])*100, extent=[radii[rstart],radii.max(),1,6], vmax=5)
plt.colorbar()

--
%cpaste
def plot_rad_vs_slope_individual_sets(slopes, slopes_unc, mod_XY_string = '', xlim = None,
        nbins = 5, loc = 9, alpha = 0.15, saveFig=False, printRng = False):
    fig = gcf()
    fig.clf()
    ax1 = fig.add_axes([0.125, 0.15, 0.85, 0.75])#fig.add_subplot(111)
    lines   = ax1.plot(radii, slopes.T, lw=1)
    for g in range(len(lines)):
        ax1.fill_between(radii, slopes[g] - slopesY_481_unc[g], slopes[g] + slopes_unc[g], 
                            color=lines[g].get_color(), alpha=alpha)

    if printRng:
        print 'maxISVP_Y_481 = {:1.3f}%'.format(ax1rng*100)
        print 'maxISVP_X_481 = {:1.3f}%'.format(ax2rng*100)
        print 'maxISVP_Y_489 = {:1.3f}%'.format(ax3rng*100)
        print 'maxISVP_X_489 = {:1.3f}%'.format(ax4rng*100)

    if xlim == None:
        ax1.set_xlim(2, radii.max())
    else:
        ax1.set_xlim(xlim)
    
    ax1min  = npmin((slopes - slopes_unc)[:,(radii>xlim[0])*(radii<xlim[1])])
    ax1max  = npmax((slopes + slopes_unc)[:,(radii>xlim[0])*(radii<xlim[1])])
    ax1rng  = npmax([abs(ax1min), abs(ax1max)])
    
    ax1.set_ylim(-ax1rng*1.05, ax1rng*1.05)
    ax1.axhline(xlim[0], color='grey', linestyle='--')
    ax1.axhline(xlim[1], color='grey', linestyle='--')
    for k in arange(xlim[0],xlim[1]):
        ax1.axvline(k, color='grey', lw=3, alpha=alpha)
    
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    
    ax1.plot([],[], lw=3, color=rcParams['axes.color_cycle'][0], label='NRCN821CLRSUB1')
    ax1.plot([],[], lw=3, color=rcParams['axes.color_cycle'][1], label='NRCN821CLRSUB2')
    ax1.plot([],[], lw=3, color=rcParams['axes.color_cycle'][2], label='NRCN821CLRSUB3')
    ax1.plot([],[], lw=3, color=rcParams['axes.color_cycle'][3], label='NRCN821CLRSUB4')
    ax1.plot([],[], lw=3, color=rcParams['axes.color_cycle'][4], label='NRCN821CLRSUB5')
    ax1.plot([],[], lw=3, color=rcParams['axes.color_cycle'][0], label='NRCN821CLRSUB6')
    
    ax1.legend(loc=loc, fontsize=5)
    ax1.set_ylabel('Slopes', fontsize=10)

    # fig.suptitle('Slope of Gaussian Centers vs Flux Correlation over Aperture Radius', fontsize=15)
    ax1.set_xlabel('Aperture Radius', fontsize=10)

    plt.subplots_adjust(left=0.10, bottom=0.12)
    fig.canvas.draw()

    if saveFig:
        fig.savefig('Radii vs Slope Individual Sets - '+ mod_XY_string +'.png')

def plot_rad_vs_slope_individual_sets_multiplot(saveFig=False, printRng = False):
    loc     = 9
    alpha   = 0.15
    fig = gcf()
    fig.clf()
    ax1 = fig.add_subplot(411)
    lines   = ax1.plot(radii, slopesY_481.T, lw=1)
    for g in range(len(lines)):
        ax1.fill_between(radii, slopesY_481[g] - slopesY_481_unc[g], slopesY_481[g] + slopesY_481_unc[g], 
                            color=lines[g].get_color(), alpha=alpha)

    ax2 = fig.add_subplot(412)
    lines   = ax2.plot(radii, slopesX_481.T, lw=1)
    for g in range(len(lines)):
        ax2.fill_between(radii, slopesX_481[g] - slopesX_481_unc[g], slopesX_481[g] + slopesX_481_unc[g], 
                            color=lines[g].get_color(), alpha=alpha)
    
    ax3 = fig.add_subplot(413)
    lines   = ax3.plot(radii, slopesY_489.T, lw=1)
    for g in range(len(lines)):
        ax3.fill_between(radii, slopesY_489[g] - slopesY_489_unc[g], slopesY_489[g] + slopesY_489_unc[g], 
                            color=lines[g].get_color(), alpha=alpha)

    ax4 = fig.add_subplot(414)
    lines   = ax4.plot(radii, slopesX_489.T, lw=1)
    for g in range(len(lines)):
        ax4.fill_between(radii, slopesX_489[g] - slopesX_489_unc[g], slopesX_489[g] + slopesX_489_unc[g], 
                            color=lines[g].get_color(), alpha=alpha)

    ax1.axhline(0.0, color='grey', linestyle='--')
    ax2.axhline(0.0, color='grey', linestyle='--')
    ax3.axhline(0.0, color='grey', linestyle='--')
    ax4.axhline(0.0, color='grey', linestyle='--')

    for k in arange(2,10,2):
        ax1.axvline(k, color='grey', lw=3, alpha=alpha)
        ax2.axvline(k, color='grey', lw=3, alpha=alpha)
        ax3.axvline(k, color='grey', lw=3, alpha=alpha)
        ax4.axvline(k, color='grey', lw=3, alpha=alpha)

    ax1min  = npmin((slopesY_481 - slopesY_481_unc)[:,radii>3])
    ax1max  = npmax((slopesY_481 + slopesY_481_unc)[:,radii>3])
    ax1rng  = npmax([abs(ax1min), abs(ax1max)])

    ax2min  = npmin((slopesX_481 - slopesX_481_unc)[:,radii>3])
    ax2max  = npmax((slopesX_481 + slopesX_481_unc)[:,radii>3])
    ax2rng  = npmax([abs(ax2min), abs(ax2max)])

    ax3min  = npmin((slopesY_489 - slopesY_489_unc)[:,radii>3])
    ax3max  = npmax((slopesY_489 + slopesY_489_unc)[:,radii>3])
    ax3rng  = npmax([abs(ax3min), abs(ax3max)])

    ax4min  = npmin((slopesX_489 - slopesX_489_unc)[:,radii>3])
    ax4max  = npmax((slopesX_489 + slopesX_489_unc)[:,radii>3])
    ax4rng  = npmax([abs(ax4min), abs(ax4max)])

    ax1.set_ylim(-ax1rng, ax1rng)
    ax2.set_ylim(-ax2rng, ax2rng)
    ax3.set_ylim(-ax3rng, ax3rng)
    ax4.set_ylim(-ax4rng, ax4rng)

    if printRng:
        print 'maxISVP_Y_481 = {:1.3f}%'.format(ax1rng*100)
        print 'maxISVP_X_481 = {:1.3f}%'.format(ax2rng*100)
        print 'maxISVP_Y_489 = {:1.3f}%'.format(ax3rng*100)
        print 'maxISVP_X_489 = {:1.3f}%'.format(ax4rng*100)

    # ax2.set_ylim(npmin((slopesX_481 - slopesX_481_unc)[:,radii>3]), npmax((slopesX_481 + slopesX_481_unc)[:,radii>3]))
    # ax3.set_ylim(npmin((slopesY_489 - slopesY_489_unc)[:,radii>3]), npmax((slopesY_489 + slopesY_489_unc)[:,radii>3]))
    # ax4.set_ylim(npmin((slopesX_489 - slopesX_489_unc)[:,radii>3]), npmax((slopesX_489 + slopesX_489_unc)[:,radii>3]))
    
    ax1.set_xlim(2, radii.max())
    ax2.set_xlim(2, radii.max())
    ax3.set_xlim(2, radii.max())
    ax4.set_xlim(2, radii.max())

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax4.yaxis.set_major_locator(MaxNLocator(nbins=3))
    
    ax1.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(axis='x', labelbottom=False)
    ax3.tick_params(axis='x', labelbottom=False)
    
    ax1.plot([],[], lw=3, color=rcParams['axes.facecolor'], label='Mod A: Y vs Flux')
    ax2.plot([],[], lw=3, color=rcParams['axes.facecolor'], label='Mod A: X vs Flux')
    ax3.plot([],[], lw=3, color=rcParams['axes.facecolor'], label='Mod B: Y vs Flux')
    ax4.plot([],[], lw=3, color=rcParams['axes.facecolor'], label='Mod B: X vs Flux')
    
    ax1.legend(loc=loc, fontsize=5)
    ax2.legend(loc=loc, fontsize=5)
    ax3.legend(loc=loc, fontsize=5)
    ax4.legend(loc=loc, fontsize=5)
    
    ax1.set_ylabel('Slopes', fontsize=10)
    ax2.set_ylabel('Slopes', fontsize=10)
    ax3.set_ylabel('Slopes', fontsize=10)
    ax4.set_ylabel('Slopes', fontsize=10)

    fig.suptitle('Slope of Correlation b/n Gaussian Centers and Flux vs Aperture Radius', fontsize=15)
    ax4.set_xlabel('Aperture Radius', fontsize=10)

    plt.subplots_adjust(left=0.10, bottom=0.12)
    fig.canvas.draw()

    if saveFig:
        fig.savefig('Radii vs Slope Individual Sets - Mod A and B.png')

def get_date(year2digit=True):
    dtime   = datetime.fromtimestamp(time())
    
    month   = str(dtime.month)
    if len(month) == 1:
        month = '0' + month

    day   = str(dtime.day)
    if len(day) == 1:
        day = '0' + day

    year    = str(dtime.year)

    if year2digit:
        return month + '.' + day + '.' + year[2:]
    else:
        return month + '.' + day + '.' + year

def plot_phots_over_time(SaveFig=False):
    fig = gcf()
    fig.clf()
    ax  = fig.add_subplot(111)

    # ax.plot(phot_arr_CCArr_481 / median(phot_arr_CCArr_481, axis=0)+0.075, color='lightblue' , alpha=0.014, linewidth=1)
    # ax.plot(phot_arr_GSArr_481 / median(phot_arr_GSArr_481, axis=0)+0.025, color='lightgreen', alpha=0.014, linewidth=1)
    # ax.plot(phot_arr_CCArr_489 / median(phot_arr_CCArr_489, axis=0)-0.025, color='pink'      , alpha=0.014, linewidth=1)
    # ax.plot(phot_arr_GSArr_489 / median(phot_arr_GSArr_489, axis=0)-0.075, color='violet'    , alpha=0.014, linewidth=1)
    ax.plot(phot_arr_CCArr_481_normed+0.075, color='lightblue' , alpha=0.014, linewidth=1)
    ax.plot(phot_arr_GSArr_481_normed+0.025, color='lightgreen', alpha=0.014, linewidth=1)
    ax.plot(phot_arr_CCArr_489_normed-0.025, color='pink'      , alpha=0.014, linewidth=1)
    ax.plot(phot_arr_GSArr_489_normed-0.075, color='violet'    , alpha=0.014, linewidth=1)

    ax.plot([],[], lw=5, color='lightblue'   , label='A1; Cross Correlation')
    ax.plot([],[], lw=5, color='lightgreen'  , label='A1; Gaussian Fit')
    ax.plot([],[], lw=5, color='pink'        , label='B4; Cross Correlation')
    ax.plot([],[], lw=5, color='violet'      , label='B4; Gaussian Fit')
    ax.legend(loc=0)

    ax.set_title('Short Wavelength CV3 Test Photometry In-Focus')
    ax.set_xlabel('Frame Number (~ `time`)')
    ax.set_ylabel('Normalized Flux + offset')
    ax.set_ylim(0.75, 1.11)
    ax.set_xlim(0,phot_arr_CCArr_481.shape[0])

    fig.canvas.draw()
    if SaveFig:
        dateNow = get_date()
        fig.savefig('ShortWave_CV3_Phots_InFocus.' + dateNow + '.png')

def plot_phots_all_over_time0(phots, nPts = 100, loc=0, xfactor = 1.65, SaveFig=False):
    fig = gcf()
    fig.clf()
    nGroups = phots.shape[0] / nPts
    ax  = fig.add_axes([0.125, 0.15, 0.85, 0.75])
    color_cycle = rcParams['axes.color_cycle']
    nColors     = len(color_cycle)
    for k in range(phots.shape[0] / nPts):
        ax.plot(phots[k*nPts:(k+1)*nPts]+k*0.03, alpha=0.014, linewidth=1, color=color_cycle[k%nColors])

    for k in arange(phots.shape[0] / nPts)[::-1]:
        ax.plot([],[], lw=5, color=color_cycle[k%nColors], label='NRCN821CLRSUB{0}'.format(k+1))
    
    ax.legend(loc=loc)
    # plt.rc('legend', fontsize='small')

    ax.axvline(nPts, color='grey')
    ax.set_title('Short Waves In-Focus CV3 Test Phots')
    ax.set_xlabel('Frame Number (~ `time`)')
    ax.set_ylabel('Normalized Flux + offset')
    ax.set_ylim(0.95, 1.18)
    ax.set_xlim(0,nPts*xfactor)

    fig.canvas.draw()
    if SaveFig:
        dateNow = get_date()
        fig.savefig('ShortWave_CV3_Phots_InFocus_Individual.' + dateNow + '.png')

def plot_phots_all_over_time(phots, nPts = 100, loc=0, xfactor = 1.65, SaveFig=False):
    fig = gcf()
    fig.clf()
    nGroups = phots.shape[0] / nPts
    ax  = fig.add_axes([0.125, 0.15, 0.85, 0.75])
    color_cycle = rcParams['axes.color_cycle']
    nColors     = len(color_cycle)
    for k in range(phots.shape[0] / nPts):
        ax.plot(phots[k*nPts:(k+1)*nPts]+k*0.03, alpha=0.014, linewidth=1, color=color_cycle[k%nColors])

    for k in arange(phots.shape[0] / nPts)[::-1]:
        ax.plot([],[], lw=5, color=color_cycle[k%nColors], label='NRCN821CLRSUB{0}'.format(k+1))
    
    ax.legend(loc=loc)
    # plt.rc('legend', fontsize='small')

    ax.axvline(nPts, color='grey')
    ax.set_title('Short Waves In-Focus CV3 Test Phots')
    ax.set_xlabel('Frame Number (~ `time`)')
    ax.set_ylabel('Normalized Flux + offset')
    ax.set_ylim(0.95, 1.18)
    ax.set_xlim(0,nPts*xfactor)

    fig.canvas.draw()
    if SaveFig:
        dateNow = get_date()
        fig.savefig('ShortWave_CV3_Phots_InFocus_Individual.' + dateNow + '.png')

def plot_phots_vs_time(phots, times, nPts = 100, kGap = 0.05, loc=0, SaveFig=False):
    color_cycle = rcParams['axes.color_cycle']
    nColors     = len(color_cycle)
    fig = gcf()
    fig.clf()
    ax  = fig.add_subplot(111)
    for k in range(phots.shape[0] / nPts):
        ax.plot((times[k*nPts:(k+1)*nPts] - times[k*nPts:(k+1)*nPts].min())*day2sec / 60., phots[k*nPts:(k+1)*nPts]+k*kGap, alpha=0.014, linewidth=1, color=color_cycle[k%nColors])
    fig.canvas.draw()

def plot_all_over_time_ModA(SaveFig=False):
    fig = gcf()
    fig.clf()

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412, sharex=ax1)
    ax3 = fig.add_subplot(413, sharex=ax1)
    ax4 = fig.add_subplot(414, sharex=ax1)
    
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    ax1.plot(phot_arr_GSArr_481_normed  , color='lightblue' , alpha=0.014, linewidth=1)

    ax2.plot(gaussfitsArr_481[:,1] - median(gaussfitsArr_481[:,1])      , color='lightgreen', alpha=0.500, linewidth=1)
    ax2.plot(gaussfitsArr_481[:,2] - median(gaussfitsArr_481[:,2])     , color='magenta'  , alpha=0.500, linewidth=1)
    
    ax3.plot(gaussfitsArr_481[:,3]      , color='lightgreen', alpha=0.500, linewidth=1)
    ax3.plot(gaussfitsArr_481[:,4]      , color='magenta'   , alpha=0.500, linewidth=1)
    
    # ax4.plot(crossCorrArr_481[:,1]      , color='pink'      , alpha=0.500, linewidth=1)
    # ax4.plot(crossCorrArr_481[:,0]      , color='pink'      , alpha=0.500, linewidth=1)
    
    ax4.plot(skyBGArr_481               , color='violet'    , alpha=0.500, linewidth=1)

    ax1.plot([],[], lw=1, color='lightblue' , label='A1; Photometry (GS)')

    ax2.plot([],[], lw=1, color='lightgreen', label='A1; Y Centers (GS)')
    ax2.plot([],[], lw=1, color='magenta'   , label='A1; X Centers (GS)')
    
    ax3.plot([],[], lw=1, color='lightgreen', label='A1; Y Widths (GS)')
    ax3.plot([],[], lw=1, color='magenta'   , label='A1; X Widths (GS)')
    
    # ax4.plot([],[], lw=1, color='orange'    , label='A1; Y Centers (CC)')
    # ax4.plot([],[], lw=1, color='cyan'      , label='A1; X Centers (CC)')
    
    ax4.plot([],[], lw=1, color='violet'    , label='A1; Sky Background')

    for k in range(100,700,100):
        ax1.axvline(k, color='grey', alpha=0.25)
        ax2.axvline(k, color='grey', alpha=0.25)
        ax3.axvline(k, color='grey', alpha=0.25)
        ax4.axvline(k, color='grey', alpha=0.25)

    leg1= ax1.legend(loc=0)
    leg2= ax2.legend(loc=0)
    leg3= ax3.legend(loc=0)
    leg4= ax4.legend(loc=0)

    ax1.set_ylim(0.97, 1.035)
    ax1.set_xlim(0,phot_arr_CCArr_481.shape[0])
    
    ax1.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(axis='x', labelbottom=False)
    ax3.tick_params(axis='x', labelbottom=False)

    ax4.set_xlabel('Frame Number (~ `time`)')
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax4.yaxis.set_major_locator(MaxNLocator(nbins=3))

    fig.suptitle('Module A Phots, GS Centers, GS Widths, Sky Background')
    plt.subplots_adjust(left=0.11, bottom=0.15, right=0.90, top=0.90, wspace=0.0, hspace=0.5)

    fig.canvas.draw()
    if SaveFig:
        dateNow = get_date()
        fig.savefig('ShortWave_CV3_Phots_GSCenters_CCCenters_SKyBG_InFocus_ModA.' + dateNow + '.png')

def plot_all_over_time_ModB(SaveFig=False):
    fig = gcf()
    fig.clf()

    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412, sharex=ax1)
    ax3 = fig.add_subplot(413, sharex=ax1)
    ax4 = fig.add_subplot(414, sharex=ax1)
    
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    
    ax1.plot(phot_arr_GSArr_489_normed  , color='lightblue' , alpha=0.014, linewidth=1)

    ax2.plot(gaussfitsArr_489[:,1] - median(gaussfitsArr_489[:,1])      , color='lightgreen', alpha=0.500, linewidth=1)
    ax2.plot(gaussfitsArr_489[:,2] - median(gaussfitsArr_489[:,2])     , color='magenta'  , alpha=0.500, linewidth=1)
    
    ax3.plot(gaussfitsArr_489[:,3]      , color='lightgreen', alpha=0.500, linewidth=1)
    ax3.plot(gaussfitsArr_489[:,4]      , color='magenta'   , alpha=0.500, linewidth=1)
    
    # ax4.plot(crossCorrArr_489[:,1]      , color='pink'      , alpha=0.500, linewidth=1)
    # ax4.plot(crossCorrArr_489[:,0]      , color='pink'      , alpha=0.500, linewidth=1)
    
    ax4.plot(skyBGArr_489               , color='violet'    , alpha=0.500, linewidth=1)

    ax1.plot([],[], lw=1, color='lightblue' , label='A1; Photometry (GS)')

    ax2.plot([],[], lw=1, color='lightgreen', label='A1; Y Centers (GS)')
    ax2.plot([],[], lw=1, color='magenta'   , label='A1; X Centers (GS)')
    
    ax3.plot([],[], lw=1, color='lightgreen', label='A1; Y Widths (GS)')
    ax3.plot([],[], lw=1, color='magenta'   , label='A1; X Widths (GS)')
    
    # ax4.plot([],[], lw=1, color='orange'    , label='A1; Y Centers (CC)')
    # ax4.plot([],[], lw=1, color='cyan'      , label='A1; X Centers (CC)')
    
    ax4.plot([],[], lw=1, color='violet'    , label='A1; Sky Background')

    for k in range(100,700,100):
        ax1.axvline(k, color='grey', alpha=0.25)
        ax2.axvline(k, color='grey', alpha=0.25)
        ax3.axvline(k, color='grey', alpha=0.25)
        ax4.axvline(k, color='grey', alpha=0.25)

    leg1= ax1.legend(loc=0)
    leg2= ax2.legend(loc=0)
    leg3= ax3.legend(loc=0)
    leg4= ax4.legend(loc=0)

    ax1.set_ylim(0.97, 1.035)
    ax1.set_xlim(0,phot_arr_CCArr_489.shape[0])
    
    ax1.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(axis='x', labelbottom=False)
    ax3.tick_params(axis='x', labelbottom=False)

    ax4.set_xlabel('Frame Number (~ `time`)')
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax4.yaxis.set_major_locator(MaxNLocator(nbins=3))

    fig.suptitle('Module B Phots, GS Centers, GS Widths, Sky Background')
    plt.subplots_adjust(left=0.11, bottom=0.15, right=0.90, top=0.90, wspace=0.0, hspace=0.5)

    fig.canvas.draw()
    if SaveFig:
        dateNow = get_date()
        fig.savefig('ShortWave_CV3_Phots_GSCenters_CCCenters_SKyBG_InFocus_ModB.' + dateNow + '.png')


def plot_module_correlation(SaveFig=False):
    fig = gcf()
    fig.clf()
    ax  = fig.add_subplot(111)

    #ax.plot(phot_arr_CCArr_481_normed, phot_arr_CCArr_489_normed, 'o', color='lightblue', alpha=0.25)
    ax.plot(phot_arr_GSArr_481_normed, phot_arr_GSArr_489_normed, '.', color='grey', alpha=0.25)

    #ax.plot([],[], lw=5, color='lightblue'   , label='A1 vs B4; Cross Correlation')
    ax.plot([],[], lw=5, color='grey'  , label='A1 vs B4; Gaussian Fit')

    ax.legend(loc=0)

    ax.set_title('Short Wavelength CV3 Correlation b/n Mod A & B')
    ax.set_xlabel('Module A Normalized Photometry')
    ax.set_ylabel('Module B Normalized Photometry')
    # ax.set_ylim(0.75, 1.11)
    # ax.set_xlim(0,phot_arr_CCArr_481.shape[0])

    fig.canvas.draw()
    if SaveFig:
        dateNow = get_date()
        fig.savefig('ShortWave_CV3_Correlation_ModA_ModB_Phots_InFocus.' + dateNow + '.png')

def plot_module_IPSV(SaveFig=False):
    fig = gcf()
    fig.clf()
    ax1  = fig.add_subplot(211)
    ax2  = fig.add_subplot(212)

    ax1.plot(gaussfitsArr_481[:,1] - median(gaussfitsArr_481[:,1]), phot_arr_CCArr_481_normed, '.', color='lightblue', alpha=0.25)
    ax1.plot(gaussfitsArr_481[:,2] - median(gaussfitsArr_481[:,2]), phot_arr_GSArr_481_normed, '.', color='pink'     , alpha=0.25)

    ax1.plot([],[], lw=5, color='lightblue'  , label='Y vs Flux')
    ax1.plot([],[], lw=5, color='pink'       , label='X vs Flux')

    ax2.plot(gaussfitsArr_489[:,1] - median(gaussfitsArr_489[:,1]), phot_arr_CCArr_489_normed, '.', color='lightblue', alpha=0.25)
    ax2.plot(gaussfitsArr_489[:,2] - median(gaussfitsArr_489[:,2]), phot_arr_GSArr_489_normed, '.', color='pink'     , alpha=0.25)

    ax2.plot([],[], lw=5, color='lightblue'  , label='Y vs Flux')
    ax2.plot([],[], lw=5, color='pink'       , label='X vs Flux')

    ax1.legend(loc=0)
    ax2.legend(loc=0)
    
    ax1.set_title('Short Wavelength CV3 (IPSV) Flux vs Center Positions')
    ax1.set_ylabel('Module A')
    # ax1.set_xlabel('Module A Normalized Flux')

    # ax2.set_title('Short Wavelength CV3 IPSV Correlation Module B')
    ax2.set_xlabel('Centers Position')
    ax2.set_ylabel('Module B')
    
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # ax2.spines['top'].set_color('grey')
    # ax2.spines['top'].set_linewidth(1)
    # ax2.spines['left'].set_visible(True)
    # ax.set_ylim(0.75, 1.11)
    # ax.set_xlim(0,phot_arr_CCArr_481.shape[0])
    plt.subplots_adjust(left=0.11, bottom=0.15, right=0.90, top=0.90, wspace=0.0, hspace=0.0)

    fig.canvas.draw()
    if SaveFig:
        dateNow = get_date()
        fig.savefig('ShortWave_CV3_IPSV_ModA_and_ModB_InFocus.' + dateNow + '.png')

def plot_module_IPSV_wRad(Radius = 0, SaveFig=False):

    if Radius == 0:
        plot_module_IPSV(SaveFig = SaveFig)
        return

    fig = gcf()
    fig.clf()
    ax1  = fig.add_subplot(211)
    ax2  = fig.add_subplot(212)

    radGTRad    = radii > Radius
    ax1.plot(gaussfitsArr_481[radGTRad,1] - median(gaussfitsArr_481[radGTRad,1]), phot_arr_CCArr_481_normed[radGTRad], '.', color='lightblue', alpha=0.25)
    ax1.plot(gaussfitsArr_481[radGTRad,2] - median(gaussfitsArr_481[radGTRad,2]), phot_arr_GSArr_481_normed[radGTRad], '.', color='pink'     , alpha=0.25)

    ax1.plot([],[], lw=5, color='lightblue'  , label='Y vs Flux')
    ax1.plot([],[], lw=5, color='pink'       , label='X vs Flux')

    ax2.plot(gaussfitsArr_489[radGTRad,1] - median(gaussfitsArr_489[radGTRad,1]), phot_arr_CCArr_489_normed[radGTRad], '.', color='lightblue', alpha=0.25)
    ax2.plot(gaussfitsArr_489[radGTRad,2] - median(gaussfitsArr_489[radGTRad,2]), phot_arr_GSArr_489_normed[radGTRad], '.', color='pink'     , alpha=0.25)

    ax2.plot([],[], lw=5, color='lightblue'  , label='Y vs Flux')
    ax2.plot([],[], lw=5, color='pink'       , label='X vs Flux')

    ax1.legend(loc=0)
    ax2.legend(loc=0)
    
    ax1.set_title('Short Wavelength CV3 (IPSV) Flux vs Center Positions')
    ax1.set_ylabel('Module A')
    # ax1.set_xlabel('Module A Normalized Flux')

    # ax2.set_title('Short Wavelength CV3 IPSV Correlation Module B')
    ax2.set_xlabel('Centers Position')
    ax2.set_ylabel('Module B')
    
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.spines['top'].set_color('grey')
    ax2.spines['top'].set_linewidth(1)
    ax2.spines['left'].set_visible(True)
    # ax.set_ylim(0.75, 1.11)
    # ax.set_xlim(0,phot_arr_CCArr_481.shape[0])
    plt.subplots_adjust(left=0.11, bottom=0.15, right=0.90, top=0.90, wspace=0.0, hspace=0.0)

    fig.canvas.draw()
    if SaveFig:
        dateNow = get_date()
        fig.savefig('ShortWave_CV3_IPSV_ModA_and_ModB_InFocus_AboveRad' + str(Radius) + '.' + dateNow + '.png')

def plot_IPSV_slopes_and_intercepts_modA(SaveFig=False):
    fig = gcf()
    fig.clf()
    ax1  = fig.add_subplot(111)
    ax2  = twinx(ax1)

    ax1.grid(False)
    ax2.grid(False)

    ax1.plot(radii, slopesY, color=rcParams['axes.color_cycle'][0], linewidth=1, linestyle='-' )
    ax1.plot(radii, slopesX, color=rcParams['axes.color_cycle'][0], linewidth=1, linestyle='--')
    ax1.axhline(0.0, color='black', linestyle='-', alpha=0.5, lw=1)

    for tick in ax1.yaxis.get_ticklabels():
        tick.set_color(rcParams['axes.color_cycle'][0])

    for tick in ax2.yaxis.get_ticklabels():
        tick.set_color(rcParams['axes.color_cycle'][1])
    
    ax2.plot(radii, interceptsX, color=rcParams['axes.color_cycle'][1], linestyle='--', lw=1)
    ax2.plot(radii, interceptsY, color=rcParams['axes.color_cycle'][1], lw=1)
    ax2.axhline(1.0, color='black', linestyle='-', alpha=0.5, lw=1)
    
    ax1.set_ylim(-0.01, 0.01)
    ax2.set_ylim(.98,1.02)
    
    ax1.legend(loc=4, labels=('Slopes X', 'Slopes Y'))
    ax2.legend(loc=8, labels=('Intercepts X', 'Intercepts Y'))
    
    fig.suptitle('Slope and Intercepts for IPSV A1')
    
    ax1.set_xlabel('Aperture Radii', fontsize=10)
    ax1.set_ylabel('Slopes'     , fontsize=10, color=rcParams['axes.color_cycle'][0])
    ax2.set_ylabel('Intercepts' , fontsize=10, color=rcParams['axes.color_cycle'][1])

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.88, top=0.90, wspace=0.0, hspace=0.5)

    fig.canvas.draw()
    if SaveFig:
        dateNow = get_date()
        fig.savefig('IPSV Mod A Slopes and Intercepts.' + dateNow + '.png')

def plot_IPSV_slopes_and_intercepts(module='A', ylim1 = None, ylim2 = None, SaveFig=False):

    if not module.lower() in ['a', 'b']:
        exit("`module` must be either 'A' or 'B'") 

    if module.lower() == 'a':
        GCNow   = gaussfitsArr_481
        PhotsNow= phot_arr_GSArr_481_normed

    if module.lower() == 'b':
        GCNow   = gaussfitsArr_489
        PhotsNow= phot_arr_GSArr_489_normed
    
    linemodel = models.Linear1D(slope=0, intercept=1)
    
    slopesX         = empty((len(radii)))
    slopesY         = empty((len(radii)))
    
    interceptsX     = empty((len(radii)))
    interceptsY     = empty((len(radii)))
    for k in range(len(radii)):
        lineNow = fit_lvmq(linemodel,GCNow[:,1] - median(GCNow[:,1]), PhotsNow[:,k])
        slopesY[k], interceptsY[k] = lineNow.parameters
        lineNow = fit_lvmq(linemodel, GCNow[:,2] - median(GCNow[:,1]), PhotsNow[:,k])
        slopesX[k], interceptsX[k] = lineNow.parameters

    fig = gcf()
    fig.clf()
    ax1  = fig.add_subplot(111)
    ax2  = twinx(ax1)

    ax1.grid(False)
    ax2.grid(False)

    ax1.plot(radii, slopesY, color=rcParams['axes.color_cycle'][0], linewidth=1, linestyle='-' , label='Slopes Y')
    ax1.plot(radii, slopesX, color=rcParams['axes.color_cycle'][0], linewidth=1, linestyle='--', label='Slopes X')
    ax1.axhline(0.0, color='black', linestyle='-', alpha=0.5, lw=1)

    for tick in ax1.yaxis.get_ticklabels():
        tick.set_color(rcParams['axes.color_cycle'][0])

    for tick in ax2.yaxis.get_ticklabels():
        tick.set_color(rcParams['axes.color_cycle'][1])
    
    ax2.plot(radii, interceptsY, color=rcParams['axes.color_cycle'][1], linestyle='-' , lw=1, label='Intercepts Y')
    ax2.plot(radii, interceptsX, color=rcParams['axes.color_cycle'][1], linestyle='--', lw=1, label='Intercepts X')
    ax2.axhline(1.0, color='black', linestyle='-', alpha=0.5, lw=1)
    
    
    # ax1.set_ylim(-0.01, 0.01)
    # ax2.set_ylim(.98,1.02)
    
    ax1.legend(loc=4, fontsize=10)
    ax2.legend(loc=8, fontsize=10)
    plt.rc('legend', fontsize='small')

    fig.suptitle('Slope and Intercepts for IPSV Mod ' + module.upper())
    
    ax1.set_xlabel('Aperture Radii', fontsize=10)
    ax1.set_ylabel('Slopes'     , fontsize=10, color=rcParams['axes.color_cycle'][0])
    ax2.set_ylabel('Intercepts' , fontsize=10, color=rcParams['axes.color_cycle'][1])

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax1.set_ylim(ylim1)
    ax2.set_ylim(ylim2)

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.88, top=0.90, wspace=0.0, hspace=0.5)

    fig.canvas.draw()
    if SaveFig:
        dateNow = get_date()
        fig.savefig('IPSV Mod '+ module.upper() +' Slopes and Intercepts.' + dateNow + '.png')

def plot_module_correlation_wRadius(SaveFig=False):
    fig = gcf()
    fig.clf()
    ax  = fig.add_subplot(111)

    radgt1  = radii > 1.0
    radgt2  = radii > 2.0
    radgt3  = radii > 3.0
    #ax.plot(phot_arr_CCArr_481_normed, phot_arr_CCArr_489_normed, 'o', color='lightblue', alpha=0.25)
    ax.plot(phot_arr_GSArr_481_normed        , phot_arr_GSArr_489_normed        , '.', ms=8, color='lightblue', alpha=0.5)
    ax.plot(phot_arr_GSArr_481_normed[radgt1], phot_arr_GSArr_489_normed[radgt1], '.', ms=5, color='lightgreen', alpha=0.5)
    ax.plot(phot_arr_GSArr_481_normed[radgt2], phot_arr_GSArr_489_normed[radgt2], '.', ms=3, color='pink', alpha=0.5)
    ax.plot(phot_arr_GSArr_481_normed[radgt3], phot_arr_GSArr_489_normed[radgt3], '.', ms=2, color='violet', alpha=0.5)

    #ax.plot([],[], lw=5, color='lightblue'   , label='A1 vs B4; Cross Correlation')
    ax.plot([],[], lw=5, color='lightblue'  , label='A1 vs B4; Gaussian Fit; rad > 0')
    ax.plot([],[], lw=5, color='lightgreen' , label='A1 vs B4; Gaussian Fit; rad > 1')
    ax.plot([],[], lw=5, color='pink'       , label='A1 vs B4; Gaussian Fit; rad > 2')
    ax.plot([],[], lw=5, color='violet'     , label='A1 vs B4; Gaussian Fit; rad > 3')

    ax.legend(loc=0)

    ax.set_title('Short Wavelength CV3 Correlation b/n Mod A & B vs Radius')
    ax.set_xlabel('Module A Normalized Photometry')
    ax.set_ylabel('Module B Normalized Photometry')
    # ax.set_ylim(0.75, 1.11)
    # ax.set_xlim(0,phot_arr_CCArr_481.shape[0])

    fig.canvas.draw()
    if SaveFig:
        dateNow = get_date()
        fig.savefig('ShortWave_CV3_Correlation_ModA_ModB_Phots_InFocus_vRadius.' + dateNow + '.png')

def plot_lomb_scargle_gatspy(phots, phots_unc, k_rad, dt = 1.0, frange = [0,100], 
                            color = None, nyquist_factor=100, xlim=None, ylim=None, alpha = 1.0, 
                            xlabel='period (frames)', ylabel='Lomb-Scargle Power'):
    
    from gatspy.periodic import LombScargleFast
    
    nRads   = 100
    minRad  = 0.1
    maxRad  = 10
    radii   = linspace(minRad, maxRad, nRads)

    t       = arange(frange[1] - frange[0])*dt
    mag     = phot_arr_GSArr_489_normed[frange[0]:frange[1], k_rad].copy()
    dmag    = phot_arr_GSArr_489_normed_unc[frange[0]:frange[1], k_rad].copy()

    mag     = mag - median(mag)

    model   = LombScargleFast().fit(t, mag, dmag)
    periods, power = model.periodogram_auto(nyquist_factor=nyquist_factor)

    if xlim == None:
        xlim = [diff(t).min(), (t.max() - t.min()) / 2]

    if ylim == None:
        ylim    = min(power[(periods > xlim[0])*(periods < xlim[1])]), max(power[(periods > xlim[0])*(periods < xlim[1])])
        ylim    = array(ylim)
        ylim[1] = ylim[1]*1.01 # one percent more

    model.optimizer.period_range=(xlim[0], xlim[1])
    bestperiod = model.best_period;

    fig = gcf()
    ax  = fig.get_axes()

    if not len(ax):
        ax  = fig.add_subplot(111)
    else:
        ax  = ax[0]

    if color == None:
        ax.plot(periods, power, alpha = alpha)
    else:
        ax.plot(periods, power, color = color, alpha = alpha)

    ax.axvline(bestperiod, linestyle='--')
    ax.set(xlim = xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, 
            title = 'radius = {0:.1f}'.format(radii[k_rad]) + ' period = {0:.2f}'.format(bestperiod))

    fig.canvas.draw()

def plot_lomb_scargle_gatspy_subplots(phots, phots_unc, k_rad, nPts = 100, color = None,
                            nyquist_factor=100, xlim=None, ylim=None, alpha = 1.0, 
                            xlabel='period (frames)', ylabel='Lomb-Scargle Power'):
    
    from gatspy.periodic import LombScargleFast
    
    fig     = gcf()
    ax1     = fig.add_subplot(611)
    ax2     = fig.add_subplot(612, sharex=ax1)
    ax3     = fig.add_subplot(613, sharex=ax1)
    ax4     = fig.add_subplot(614, sharex=ax1)
    ax5     = fig.add_subplot(615, sharex=ax1)
    ax6     = fig.add_subplot(616, sharex=ax1)
    axs     = [ax1, ax2, ax3, ax4, ax5, ax6]

    nRads   = 100
    minRad  = 0.1
    maxRad  = 10
    radii   = linspace(minRad, maxRad, nRads)

    for k in range(len(axs)):
        ax      = axs[k]
        frange  = [k*nPts, (k+1)*nPts]
        t       = arange(frange[1] - frange[0])
        mag     = phot_arr_GSArr_489_normed[frange[0]:frange[1], k_rad]
        dmag    = phot_arr_GSArr_489_normed_unc[frange[0]:frange[1], k_rad]

        model   = LombScargleFast().fit(t, mag, dmag)
        periods, power = model.periodogram_auto(nyquist_factor=nyquist_factor)

        if xlim == None:
            xlim = [diff(t).min(), (t.max() - t.min()) / 2]

        if ylim == None:
            ylim    = min(power[(periods > xlim[0])*(periods < xlim[1])]), max(power[(periods > xlim[0])*(periods < xlim[1])])
            ylim    = array(ylim)
            ylim[1] = ylim[1]*1.01 # one percent more

        model.optimizer.period_range=(xlim[0], xlim[1])
        bestperiod = model.best_period;

        if color == None:
            ax.plot(periods, power, alpha = alpha)
        else:
            ax.plot(periods, power, color = color, alpha = alpha)

        ax.axvline(bestperiod, linestyle='--')
        ax.set(xlim = xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, 
                title = 'radius = {0:.1f}'.format(radii[k_rad]) + ' period = {0:.2f}'.format(bestperiod))

    fig.canvas.draw()

--

''' UNCOMMENT TO SAVE PICKLEOUT '''
'''
pickleOut                       = {}
pickleOut['time_fits']          = time_fits
pickleOut['diff_fits']          = diff_fits
pickleOut['gaussfits']          = gaussfits
pickleOut['skyBG']              = skyBG
pickleOut['crossCorr']          = crossCorr
pickleOut['phot_arr_GS']        = phot_arr_GS
pickleOut['phot_arr_CC']        = phot_arr_CC
pickleOut['skyBGArr']           = skyBGArr
pickleOut['gaussfitsArr']       = gaussfitsArr
pickleOut['crossCorrArr']       = crossCorrArr
pickleOut['phot_arr_GSArr']     = phot_arr_GSArr
pickleOut['phot_arr_CCArr']     = phot_arr_CCArr
pickleOut['phot_arr_GSArr_481'] = phot_arr_GSArr_481
pickleOut['phot_arr_GSArr_489'] = phot_arr_GSArr_489
pickleOut['phot_arr_CCArr_481'] = phot_arr_CCArr_481
pickleOut['phot_arr_CCArr_489'] = phot_arr_CCArr_489
pickleOut['gaussfitsArr_481']   = gaussfitsArr_481
pickleOut['gaussfitsArr_489']   = gaussfitsArr_489
pickleOut['crossCorrArr_481']   = crossCorrArr_481
pickleOut['crossCorrArr_489']   = crossCorrArr_489
pickleOut['skyBGArr_481']       = skyBGArr_481
pickleOut['skyBGArr_489']       = skyBGArr_489

pickleDir                       = 'pickleSaves/'
pickleFileName                  = 'NIRCam_CV3_DefaultFlags_InitialRunThrough.pkl'

joblib.dump(pickleOut, pickleDir + pickleFileName)
'''
''' UNCOMMENT TO LOAD PICKLEOUT '''
'''
pickleDir                       = 'pickleSaves/'
pickleFileName                  = 'NIRCam_CV3_DefaultFlags_InitialRunThrough.pkl'

pickleOut                       = joblib.load(pickleDir + pickleFileName) 
'''

'''
plot_phots_over_time(SaveFig=False)
plot_all_over_time_ModA(SaveFig=False)
plot_all_over_time_ModB(SaveFig=False)
plot_module_correlation(SaveFig=False)
plot_module_correlation_wRadius(SaveFig=False)
plot_module_IPSV(SaveFig=False)
plot_module_IPSV_wRad(Radius = 0, SaveFig=False)
plot_IPSV_slopes_and_intercepts(module='A', SaveFig=False)
plot_rad_vs_slope_individual_sets(slopes, slopes_unc, mod_XY_string = '',saveFig=False, printRng = False)
plot_rad_vs_slope_individual_sets_multiplot(saveFig=False, printRng = False)
plot_lomb_scargle_gatspy(phots = phot_arr_GSArr_489_normed, phots_unc = phot_arr_GSArr_489_normed_unc, k_rad = 50)
plot_lomb_scargle_gatspy_subplots(phots = phot_arr_GSArr_489_normed, phots_unc = phot_arr_GSArr_489_normed_unc, k_rad = 50)
'''

# print abs(med_slopes_481), abs(min_slopes_481), abs(max_slopes_481)
# print abs(med_slopes_489), abs(min_slopes_489), abs(max_slopes_489)

# ### imshow all focused images with center positions ###
# PlotMovie   = False
# if PlotMovie:
#     for key0 in sort(diff_fits.keys()):
#         if not 'wlp' in key0.lower():
#             for key1 in sort(diff_fits[key0].keys()):
#                 ax.clear()
#                 ax.imshow(log10(diff_fits[key0][key1] - diff_fits[key0][key1].min()))
#                 ax.scatter(*gaussfits[key0][key1]['results'].parameters[1:3] - array([160-20, 160-20]), s=1)
#                 ax.set_xlim(0,40)
#                 ax.set_ylim(0,40)
#                 # ax.set_xlim(160-20, 160+20)
#                 # ax.set_ylim(160-20, 160+20)
#                 fig.canvas.draw()

# sumdiff_fits = []#empty((len(wlp_set),nRamps))
# for w in range(len(wlp_set)):
#     sumdiff_fits.append(empty(WLPfits[w][0].data.shape[0]))
#     for k in range(WLPfits[w][0].data.shape[0]):
#         print_arbitrary(array(fitsfiles)[wlp_set][w] + '; w = ' + str(w+1) + ' / ' + str(len(wlp_set)) + '; k = ' + str(k+1) + ' / ' + str(WLPfits[w][0].data.shape[0]))
#         sumdiff_fits[w][k] = (WLPfits[w][0].data[k] - WLPfits[w][0].data[0]).sum()

###
# for k, key in enumerate(wlp_sumdiff_surtr.keys()):
#     fig = figure();ax = fig.add_subplot(211)
#     ax.plot(wlp_sumdiff_surtr[key])
#     ax = fig.add_subplot(212)
#     ax.plot(diff(wlp_sumdiff_surtr[key]))
#     fig.suptitle(key)
#     fig.canvas.draw()
###

# # print 'Creating Differential WLP Images'
# # wlp_diff    = []
# # for w in range(len(wlp_set)):
# #     wlp_diff.append(empty((WLPfits[w][0].data.shape[0]/nRamps, WLPfits[w][0].data.shape[1], WLPfits[w][0].data.shape[2])))
# #     for k in range(WLPfits[w][0].data.shape[0]/nRamps):
# #         print_arbitrary('Processing step ' + str(w) + '; ' + str(k+1) + ' out of ' + str(WLPfits[w][0].data.shape[0]/nRamps) + ' Integrations')
# #         wlp_diff[w][k] = WLPfits[w][0].data[k*nRamps+19] - WLPfits[w][0].data[k*nRamps]

# print 'Assessing Statistical Limits on Electronic Background'
# #imgNow      = wlp_diff[0][0]
# nSig        = 5
# eps         = 1e-6
# upperSig    = 100*erf(float(nSig) / sqrt(2.0))
# lowerSig    = 100 - upperSig
# medSig      = 0.5*(upperSig + lowerSig)

# llim_img    = []
# med_img     = []
# ulim_img    = []
# for w in range(len(wlp_diff)):
#     llim_img.append(empty(wlp_diff[w].shape[0]))
#     med_img.append(empty(wlp_diff[w].shape[0]))
#     ulim_img.append(empty(wlp_diff[w].shape[0]))

# for w in range(len(wlp_diff)):
#     for f in range(wlp_diff[w].shape[0]):
#         med_imgNow  = median(wlp_diff[w][f].ravel())
#         mad_imgNow  = scale.mad(wlp_diff[w][f].ravel())
#         limits0  = percentile(wlp_diff[w][f][abs(wlp_diff[w][f] - med_imgNow) <  nSig*mad_imgNow].ravel(), [lowerSig, medSig, upperSig])
#         limitsSm = percentile(wlp_diff[w][f][abs(wlp_diff[w][f] - med_imgNow) <       mad_imgNow].ravel(), [lowerSig, medSig, upperSig])
#         limits1  = percentile(wlp_diff[w][f][(wlp_diff[w][f] > limits0[0])*(wlp_diff[w][f] < limits0[-1])].ravel(), [lowerSig, medSig, upperSig])
#         limits2  = percentile(wlp_diff[w][f][(wlp_diff[w][f] > limits1[0])*(wlp_diff[w][f] < limits1[-1])].ravel(), [lowerSig, medSig, upperSig])
#         llim_imgNow, med_imgNow, ulim_imgNow = limits2
#         llim_img[w][f]  = llim_imgNow
#         med_img[w][f]   = med_imgNow
#         ulim_img[w][f]  = ulim_imgNow

# ### Aperture Photometry ###
# print 'Setting Up Aperture Photometry'

# nRads           = 100
# minRad          = 0.01
# maxRad          = 150
# positions       = [(wlp_diff[0][0].shape[0]/2, wlp_diff[0][0].shape[1]/2)]
# radii           = linspace(minRad, maxRad, nRads)

# phot_arr_wSB    = []
# for w in range(len(wlp_diff)):
#     phot_arr_wSB.append(empty((wlp_diff[w].shape[0], radii.size, 3)))

# print 'Computing Aperture Photometry'
# for w in range(len(wlp_diff)):
#     for f in range(wlp_diff[w].shape[0]):
#         xcenter, ycenter    = array(wlp_diff[w][f].shape)/2
#         for k, radius in enumerate(radii):
#             print_arbitrary('Processing step ' + str(w) + '; ' + str(f) + '; ' + str(k+1) + ' out of ' + str(radii.size) + ' with radius ' + str(radius))
#             xlower, xupper          = xcenter + array([-(int(radius) + 1), (int(radius) + 1)])
#             ylower, yupper          = ycenter + array([-(int(radius) + 1), (int(radius) + 1)])
#             fluxNow                 = aperture_photometry(wlp_diff[w][f][ylower:yupper,xlower:xupper] - med_img[w][f], CircularAperture(positions, radius), method='exact')
#             phot_arr_wSB[w][f][k]   = [fluxNow['aperture_sum'], fluxNow['xcenter'], fluxNow['ycenter']]

# ### Fitting Gaussian to Core of WLP8 Image ### 
# ylower, yupper  = 155, 165
# xlower, xupper  = 161, 171
# wlpdiff00_core  = wlp_diff[0][0][ylower:yupper, xlower:xupper].copy()

# def fitGauss_astropy(imgArray, iparams = None, nIters = 1):
#     nPix    = imgArray.shape[0]

#     if iparams is not None:
#         hgt_guess       = iparams[0]
#         xctr_guess      = iparams[1]
#         yctr_guess      = iparams[2]
#         xstd_guess      = iparams[3]
#         ystd_guess      = iparams[4]
#         thta_guess      = iparams[5]
#     else:
#         hgt_guess       = mean([imgArray.sum(axis=0).max()/nPix, imgArray.sum(axis=1).max()/nPix])
#         xctr_guess      = nPix / 2.
#         yctr_guess      = nPix / 2.
#         xstd_guess      = nPix / 5.
#         ystd_guess      = nPix / 5.
#         thta_guess      = 0.0
#     #
#     yinds, xinds= indices((nPix, nPix))
#     fit_LMQ     = fitting.LevMarLSQFitter()
#     gaussmodel = models.Gaussian2D(amplitude   = hgt_guess , 
#                                     x_mean      = xctr_guess, 
#                                     y_mean      = yctr_guess, 
#                                     x_stddev    = xstd_guess, 
#                                     y_stddev    = ystd_guess)
#     #
#     gmodel                      = fit_LMQ(gaussmodel, xinds, yinds, imgArray)
#     h0, cx0, cy0, sx0, sy0, th0 = gmodel.parameters
#     #
#     for k in range(nIters-1):
#         gaussmodel = models.Gaussian2D(amplitude   = h0, 
#                                         x_mean      = cx0, 
#                                         y_mean      = cy0, 
#                                         x_stddev    = sx0, 
#                                         y_stddev    = sy0)

#         gmodel                     = fit_LMQ(gaussmodel, xinds, yinds, imgArray)
#         h0, cx0, cy0, sx0, sy0, th0 = gmodel.parameters

#     return gmodel

# # gmodelAP_core   = fitGauss_astropy(wlpdiff00_core, nIters=10)
# # gmodelAP_full   = fitGauss_astropy(wlp_diff[0][0], nIters=10)
# gmodelAP_core   = []
# gmodelAP_full   = []
# crossCorrCenters= []

# webbpsf_150W_10p14  = fits.open('PSF_NIRCam_F150W_10p14.fits')[0]

# nIters  = 2
# for w in range(len(wlp_diff)):
#     #gmodelAP_core.append([])
#     #gmodelAP_full.append([])
#     crossCorrCenters.append(empty((len(wlp_diff[w]),2)))
#     for f in range(len(wlp_diff[w])):
#         crossCorrCenters[-1]    = cross_correlation_shifts(wlp_diff[w][f], webbpsf_150W_10p14)
#         #wlpcoreNow  = wlp_diff[w][f][ylower:yupper, xlower:xupper].copy()
#         #gmodelAP_core[-1].append(fitGauss_astropy(wlpcoreNow    , nIters=nIters))
#         #gmodelAP_full[-1].append(fitGauss_astropy(wlp_diff[w][f], nIters=nIters))
        



"""
nSig    = 3
for w in range(len(wlp_diff)):
    fig = figure(w+1)
    fig.clf()
    med_phot_arr_wSB    = median(phot_arr_wSB[w][:,:,0]             / median(phot_arr_wSB[w][:,:,0]))
    mad_phot_arr_wSB    = scale.mad(phot_arr_wSB[w][:,:,0].ravel()  / median(phot_arr_wSB[w][:,:,0]))


    phot_arr_wSB_med    = (phot_arr_wSB[w][:,:,0] / median(phot_arr_wSB[w][:,:,0], axis=0))
    med_arr_phot_arr_wSB    = empty(phot_arr_wSB_med.shape[0])

    for f in range(phot_arr_wSB_med.shape[0]):
        phNow   = phot_arr_wSB_med[f]
        med_arr_phot_arr_wSB[f] = median(phNow)#[abs(phNow - med_phot_arr_wSB) < nSig*mad_phot_arr_wSB])

    phot_arr_wSB_med_stdBest = std(phot_arr_wSB_med,axis=0).argmin()

    ax  = fig.add_subplot(111)
    for k in range(1000):
        ax.plot(phot_arr_wSB_med[:,k], alpha=0.01, color='grey', lw=1)

    ax.plot(phot_arr_wSB_med[:,phot_arr_wSB_med_stdBest], 'o-')#, color='lightblue')
    ax.plot(med_arr_phot_arr_wSB, 'o--', color='darkorange')

    ax.set_ylim(med_phot_arr_wSB - mad_phot_arr_wSB, med_phot_arr_wSB + mad_phot_arr_wSB)
    ax.set_xlim(0, phot_arr_wSB[w].shape[0] - 1)

    ax.set_ylim(median(phot_arr_wSB_med.ravel()) - scale.mad(phot_arr_wSB_med.ravel())*nSig,median(phot_arr_wSB_med.ravel()) + scale.mad(phot_arr_wSB_med.ravel())*nSig)
    fig.canvas.draw()

diffRad = 0.5*median(diff(radii))
"""
"""
figure()
subplot(211)
plot(radii, phot_arr_wSB[w].T[0])
subplot(212)
plot(radii[1:] - diffRad, diff(phot_arr_wSB[w].T[0],axis=0))

### Radial Profiles vs Flux and RMS ###
azimuthalAverage = AG_image_tools.azimuthalAverage

dummy           = azimuthalAverage(wlp_diff[0][0] - med_img[0][0])
radii_wSB       = [empty((wlp_diff[0].shape[0], dummy.shape[0])), empty((wlp_diff[1].shape[0], dummy.shape[0]))]
radProfile_wSB  = [empty((wlp_diff[0].shape[0], dummy.shape[0])), empty((wlp_diff[1].shape[0], dummy.shape[0]))]
stdProfile_wSB  = [empty((wlp_diff[0].shape[0], dummy.shape[0])), empty((wlp_diff[1].shape[0], dummy.shape[0]))]

for w in range(len(wlp_diff)):
    for f in range(wlp_diff[w].shape[0]):
        radii_wSB[w][f], radProfile_wSB[w][f]   = azimuthalAverage(wlp_diff[w][f] - med_img[w][f], returnradii=True)
        radii_wSB[w][f], stdProfile_wSB[w][f]   = azimuthalAverage(wlp_diff[w][f] - med_img[w][f], returnradii=True, stddev = True)

figure()
for w in range(len(wlp_diff)):
    for f in range(wlp_diff[w].shape[0]):
        subplot(211)
        plot(radii_wSB[w][f], radProfile_wSB[w][f])
        subplot(212)
        plot(radii_wSB[w][f], stdProfile_wSB[w][f])
        plot(radii_wSB[w][f][1:], diff(radProfile_wSB[w][f]))

# ### 3D Images Over Time ###
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = gcf()
# for w in range(len(wlp_diff)):
#     for k in range(wlp_diff[1].shape[0]):
#         fig.clf()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot_surface(xinds, yinds,  wlp_diff[w][k], vmin = 0, vmax = 2e4, cmap=cm.coolwarm , linewidth=0, antialiased=False)#, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#         ax.set_zlim(0,2e4)
#         fig.canvas.draw()
#         fig.savefig('WLP_pngs/wlp_diff' + str(w) + '_' + str(k) + '.png')

"""

'''


for g in range(nGroups):
    for r in range(len(radii)):
        ''' 481 '''
        xdata   = gaussfitsArr_481[g*nData:(g+1)*nData,x] - median(gaussfitsArr_481[g*nData:(g+1)*nData,x])
        ydata   = gaussfitsArr_481[g*nData:(g+1)*nData,y] - median(gaussfitsArr_481[g*nData:(g+1)*nData,y])
        fdata   = phot_arr_GSArr_481_normed[g*nData:(g+1)*nData,r]
        #
        lineNow             = linefits_481['group'+str(g)]['y'][r]
        modelNow            = lineNow.evaluate(ydata, lineNow.slope, lineNow.intercept)
        chisqY_481[g][r]    = (((modelNow - fdata) / fdata.std())**2).sum()
        #
        lineNow             = linefits_481['group'+str(g)]['x'][r]
        modelNow            = lineNow.evaluate(ydata, lineNow.slope, lineNow.intercept)
        chisqY_481[g][r]    = (((modelNow - fdata) / fdata.std())**2).sum()
        #
        ''' 489 '''
        xdata   = gaussfitsArr_489[g*nData:(g+1)*nData,x] - median(gaussfitsArr_489[g*nData:(g+1)*nData,x])
        ydata   = gaussfitsArr_489[g*nData:(g+1)*nData,y] - median(gaussfitsArr_489[g*nData:(g+1)*nData,y])
        fdata   = phot_arr_GSArr_489_normed[g*nData:(g+1)*nData,r]
        #
        lineNow             = linefits_489['group'+str(g)]['y'][r]
        modelNow            = lineNow.evaluate(ydata, lineNow.slope, lineNow.intercept)
        chisqY_489[g][r]    = (((modelNow - fdata) / fdata.std())**2).sum()
        #
        lineNow             = linefits_489['group'+str(g)]['x'][r]
        modelNow            = lineNow.evaluate(ydata, lineNow.slope, lineNow.intercept)
        chisqY_489[g][r]    = (((modelNow - fdata) / fdata.std())**2).sum()
        #
        print_arbitrary('g: {0:d}; r: {1:d}'.format(g,r))

'''