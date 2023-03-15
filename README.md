# Wanderer for Exoplanet Time Series Observation Photometry

Planetai (πλανῆται) is Greek for "Wanderers". Moreover, Spitzer wobble makes the PSF "wandere" across the detector, which we have to tack precisely here.

```python
from wanderer import wanderer

import numpy as np
```

In our example, Spitzer data is expected to be store in the directory structure:

`$HOME/BASE_RESEARCH_DIRECTORY/PLANETNAME/data/raw/AORDIR/CHANNEL/bcd/`

EXAMPLE:

1. On a Linux machine
2. With user `tempuser`,
3. And all Spitzer data is store in `Research/Planets`
4. The planet named `Happy-5b`
5. Observed during AOR r11235813
6. In CH2 (4.5 microns)

For fun, we will call our planet HAPPY5: HApp Planet Population Yield 5  
The `loadfitsdir` should read as: `/home/tempuser/Research/Planets/HAPPY5/data/raw/r11235813/ch2/bcd/`

```python
from os import environ

planetName      = 'HAPPY5' # name of planet AND the directory where all planet files (fits, py, ipynb, joblib, etc) are stored
planetDirectory = '/Research/Planets/' # location of the `HAPPY5` directory

channel = 'ch2' # channel to use for "this" AOR -- must match subdirectory inside `HAPPY5` data/raw/$AOR/ directory

dataSub = 'bcd/' # filetype -- must match subdirectory inside `HAPPY5` data/raw/$AOR/channel/ directory

dataDir     = environ['HOME'] + planetDirectory + planetName + '/data/raw/' + channel + '/' # trailing forward slash may not be necessary

AORs = []
for dirNow in glob(dataDir + '/*'):
    AORs.append(dirNow.split('/')[-1])

fileExt = '*bcd.fits'
uncsExt = '*bunc.fits'

print(dataDir) # just to check that this is where you think it is
```

```python
iAOR        = 0 # Change this to 0,1,2,3,...,N to "cycle" over the N AORs in the base directory /home/tempuser/Research/Planets/HAPPY5/data/raw/
AORNow      = AORs[iAOR] # This will flag an error if no AORs were found in the directory `dataDir`
loadfitsdir = dataDir + AORNow + '/' + channel + '/' + dataSub
print(loadfitsdir) # just to make sure this [directory] means what you think it means
```

Set to max for multiprocessing: use cpu_count() - 1 if you want to use your computer at the same time.

```python
nCores = cpu_count()
```

Check that the files exist in the directory by printing out the number of each file (unc = uncertainty)

```python
fitsFilenames = glob(loadfitsdir + fileExt);print(len(fitsFilenames))
uncsFilenames = glob(loadfitsdir + uncsExt);print(len(uncsFilenames))
```

Check the first header in the list of files to make sure it's the data you were expecting

```python
header_test = fits.getheader(fitsFilenames[0])
print('AORLABEL:\t{}\nNum Fits Files:\t{}\nNum Unc Files:\t{}'.format\
          (header_test['AORLABEL'], len(fitsFilenames), len(uncsFilenames)))
```

# Load Wanderer Class

## Necessary Constants Spitzer (subarray with 32x32 pixels; center = 15,15)

```python
ppm             = 1e6
y,x             = 0,1

yguess, xguess  = 15., 15.   # Specific to Spitzer circa 2010 and beyond
filetype        = 'bcd.fits' # Specific to Spitzer Basic Calibrated Data
```

## Start a New Instance with Median for the Metric

```python
method = 'median'

print('Initialize an instance of `wanderer` as `example_wanderer_median`\n')
example_wanderer_median = wanderer(fitsFileDir=loadfitsdir, filetype=filetype, telescope='Spitzer',
                                            yguess=yguess, xguess=xguess, method=method, nCores=nCores)

example_wanderer_median.AOR        = AORNow
example_wanderer_median.planetName = planetName
example_wanderer_median.channel    = channel
```

```python
print('Load Data From Fits Files in ' + loadfitsdir + '\n')
example_wanderer_median.spitzer_load_fits_file(outputUnits='electrons')#(outputUnits='muJ_per_Pixel')
```

**Double check for NaNs**

```python
example_wanderer_median.imageCube[np.where(isnan(example_wanderer_median.imageCube))] = \
                                                    np.nanmedian(example_wanderer_median.imageCube)
```

**Identifier Strong Outliers**

```python
print('Find, flag, and NaN the "Bad Pixels" Outliers' + '\n')
example_wanderer_median.find_bad_pixels()
```

Flux Weighted Centroiding -- some people like this one

```python
print('Fit for All Centers: Flux Weighted, Gaussian Fitting, Gaussian Moments, Least Asymmetry' + '\n')
example_wanderer_median.fit_flux_weighted_centering()
```

Gaussian centroid fitting -- this is the most widely used centroiding method

```python
start = time()
example_wanderer_median.mp_lmfit_gaussian_centering(subArraySize=6, recheckMethod=None, median_crop=False)
print('Operation took {} seconds with {} cores'.format(time()-start, example_wanderer_median.nCores))
```

Compute the sigma-clipping outliers for plotting purpose

```python
nSig       = 10.1
medY       = median(example_wanderer_median.centering_GaussianFit.T[y])
medX       = median(example_wanderer_median.centering_GaussianFit.T[x])
stdY       = std(example_wanderer_median.centering_GaussianFit.T[y])
stdX       = std(example_wanderer_median.centering_GaussianFit.T[x])

ySig = 4
xSig = 4
outliers   = (((example_wanderer_median.centering_GaussianFit.T[y] - medY)/(ySig*stdY))**2 + \
              ((example_wanderer_median.centering_GaussianFit.T[x] - medX)/(xSig*stdX))**2) > 1
```

Plot the inliers (blue) vs outliers (not blue)

```python
ax = figure().add_subplot(111)
cx, cy = example_wanderer_median.centering_GaussianFit.T[x],example_wanderer_median.centering_GaussianFit.T[y]
ax.plot(cx,cy,'.',ms=1)
ax.plot(cx[outliers],cy[outliers],'.',ms=1)
# ax.plot(median(cx), median(cy),'ro',ms=1)
ax.set_xlim(medX-nSig*stdX,medX+nSig*stdX)
ax.set_ylim(medY-nSig*stdY,medY+nSig*stdY)
```

Use advanced clustering algorithms (DBSCAN) to determine the inliers vs outliers

```python
dbs     = DBSCAN(n_jobs=-1, eps=0.2, leaf_size=10)
dbsPred = dbs.fit_predict(example_wanderer_median.centering_GaussianFit)
```

Check over the clusteres to see the population of each

```python
dbs_options = [k for k in range(-1,100) if (dbsPred==k).sum()]
```

Plot the full extent of the data to show that DBSCAN was able to identify the inliers correctly

```python
fig = figure(figsize=(6,6))
ax  = fig.add_subplot(111)

medGaussCenters   = median(example_wanderer_median.centering_GaussianFit,axis=0)
sclGaussCenters   = scale.mad(example_wanderer_median.centering_GaussianFit)
sclGaussCenterAvg = np.sqrt(((sclGaussCenters**2.).sum()))

yctrs = example_wanderer_median.centering_GaussianFit.T[y]
xctrs = example_wanderer_median.centering_GaussianFit.T[x]

nSigmas         = 5
for nSig in linspace(1,10,10):
    CircularAperture(medGaussCenters[::-1],nSig*sclGaussCenterAvg).plot(ax=ax)

for dbsOpt in dbs_options:
    ax.plot(xctrs[dbsPred==dbsOpt], yctrs[dbsPred==dbsOpt],'.',zorder=0, ms=1)
```

Make sure that there are only a handful (<< 1%) of outliers

```python
npix = 3

stillOutliers = np.where(abs(example_wanderer_median.centering_GaussianFit - medGaussCenters) > 4*sclGaussCenterAvg)[0]
print(len(stillOutliers))
```

Select the "class" dbsClean == 0 for the `inliers`

```python
dbsClean  = 0
dbsKeep   = (dbsPred == dbsClean)
```

```python
nCores = example_wanderer_median.nCores
start = time()
example_wanderer_median.mp_measure_background_annular_mask()
print('AnnularBG took {} seconds with {} cores'.format(time() - start, nCores))
```

Plot the background to make sure that the (to be subtracted) flux is stable overtime

```python
fig = figure(figsize=(20,10))
ax  = fig.add_subplot(111)
ax.plot(example_wanderer_median.timeCube, example_wanderer_median.background_CircleMask,'.',alpha=0.2)
ax.plot(example_wanderer_median.timeCube, example_wanderer_median.background_Annulus,'.',alpha=0.2)
ax.plot(example_wanderer_median.timeCube, example_wanderer_median.background_MedianMask,'.',alpha=0.2)
ax.plot(example_wanderer_median.timeCube, example_wanderer_median.background_KDEUniv,'.',alpha=0.2)
ax.axvline(example_wanderer_median.timeCube.min()-.01+0.02)
ax.set_ylim(-25,100)
```

Compute the `effective widths` of each image to use later as the "beta pixels" and "optimal apertures"

```python
example_wanderer_median.measure_effective_width()
print(example_wanderer_median.effective_widths.mean(), sqrt(example_wanderer_median.effective_widths).mean())
```

```python
print('Pipeline took {} seconds thus far'.format(time() - startFull))
```

Compute the time series with static aperture radii only

```python
print('Iterating over Background Techniques, Centering Techniques, Aperture Radii' + '\n')
centering_choices  = ['Gaussian_Fit']#, 'Gaussian_Mom', 'FluxWeighted']#, 'LeastAsymmetry']
background_choices = ['AnnularMask']#example_wanderer_median.background_df.columns
staticRads         = np.arange(1, 6,0.5)#[1.0 ]# aperRads = np.arange(1, 6,0.5)

for staticRad in tqdm_notebook(staticRads, total=len(staticRads), desc='Static'):
    example_wanderer_median.mp_compute_flux_over_time_varRad(staticRad, varRad=None, centering_choices[0], background_choices[0], useTheForce=True)
```

Create Beta Variable Radius

```python
example_wanderer_median.mp_compute_flux_over_time_betaRad()
```

```python
print('Entire Pipeline took {} seconds'.format(time() - startFull))
```

Use Advanced clustering algorithms `DBSCAN` to compute the outliers of the flux distribution. This is sensitive the structure in the data (i.e. transit vs outlier), which is not always true with sigma-clipping.

```python
example_wanderer_median.mp_DBScan_Flux_All()
```

Check that the majority of data is an `inlier`

```python
inlier_master = array(list(example_wanderer_median.inliers_Phots.values())).mean(axis=0) == 1.0
```

```python
((~inlier_master).sum() / inlier_master.size)*100
```

Compute the PLD components -- normalized and store the PLD vectors

```python
example_wanderer_median.extract_PLD_components()
```

Use Advanced clustering algorithms `DBSCAN` to compute the outliers of the PLD distributions.

```python
example_wanderer_median.mp_DBScan_PLD_All()
```

# Save all of your progress per AOR

```python
print('Saving `example_wanderer_median` to a set of pickles for various Image Cubes and the Storage Dictionary')

savefiledir         = environ['HOME']+'/Research/Planets/'+planetName+'/ExtractedData/' + channel
saveFileNameHeader  = planetName+'_'+ AORNow +'_Median'
saveFileType        = '.joblib.save'

if not path.exists(environ['HOME']+'/Research/Planets/'+planetName+'/ExtractedData/'):
    mkdir(environ['HOME']+'/Research/Planets/'+planetName+'/ExtractedData/')

if not path.exists(savefiledir):
    print('Creating ' + savefiledir)
    mkdir(savefiledir)

print()
print('Saving to ' + savefiledir + saveFileNameHeader + saveFileType)
print()

example_wanderer_median.save_data_to_save_files(savefiledir=savefiledir, \
                                                saveFileNameHeader=saveFileNameHeader, \
                                                saveFileType=saveFileType)
```

Compute the RMS in the raw data as a function of the apeture radius

```python
color_cycle = rcParams['axes.prop_cycle'].by_key()['color']

ax = figure().add_subplot(111)
for key in example_wanderer_median.flux_TSO_df.keys():
    aperRad = float(key.split('_')[-2])
    ax.scatter(aperRad, scale.mad(np.diff(example_wanderer_median.flux_TSO_df[key])), color=color_cycle[0])

ax.set_xlabel('Aperture Radius')
ax.set_ylabel('MAD( Diff ( Flux ) )')
```

```python
print('Entire Pipeline took {} seconds'.format(time() - startFull))
```

# Convert Wanderer output to Skywalker input

# I made up this loop and did not test it

Keys for `skywalker` input

```python
flux_key = 'phots'
time_key = 'times'
flux_err_key = 'noise'
eff_width_key = 'npix'
pld_coeff_key = 'pld'
ycenter_key = 'ycenters'
xcenter_key = 'xcenters'
ywidth_key = 'ywidths'
xwidth_key = 'xwidths'
```

Things that **DON'T** change with respect to aperture radii

```python
timeCube = example_wanderer_median.timeCube
phots_array = example_wanderer_median.flux_TSO_df.values
PLDFeatures = example_wanderer_median.PLD_components.T

try:
    inliers_Phots = example_wanderer_median.inliers_Phots.values()
except:
    inliers_Phots = np.ones(photsLocal.shape)

try:
    inliers_PLD = example_wanderer_median.inliers_PLD.values()
except:
    inliers_PLD = np.ones(PLDFeatureLocal.shape)

inliersMaster = array(list(inliers_Phots)).all(axis=0) # Need to Switch `axis=0` for Qatar-2
inliersMaster = inliersMaster * inliers_PLD.all(axis=1)

nSig = 6 # vary this as desired for 3D sigma clipping double check

ypos, xpos = clipOutlier2D(transpose([example_wanderer_median.centering_GaussianFit.T[y][inliersMaster], \
                                           example_wanderer_median.centering_GaussianFit.T[x][inliersMaster]])).T

npix = sqrt(example_wanderer_median.effective_widths[inliersMaster])
time_c = timeCube[inliersMaster]
ywidths_c, xwidths_c = example_wanderer_median.widths_GaussianFit[inliersMaster].T
pld_comp_c = example_wanderer_median.PLD_components.T # this is new to Carlos's notebook instance
pld_output_c = np.array(list([time_c]) + list(pld_comp_c))
```

Things that **DO** change with respect to aperture radii

```python
for phot_select, key_flux_now in tqdm(enumerate(example_wanderer_median.flux_TSO_df.keys())):
    if key_flux_now[-3:] == 0.0: # only do static radii
        flux_c = example_wanderer_median.flux_TSO_df[key_flux_now].values[inliersMaster]
        noise_c = example_wanderer_median.noise_TSO_df[key_flux_now].values[inliersMaster]

        output_dict = {time_key: time_c,
                       flux_key: flux_c,
                       flux_err_key: noise_c,
                       eff_width_key: npix_c,
                       xcenter_key: xpos_c,
                       ycenter_key: ypos_c,
                       xwidth_key: xwidth_c,
                       ywidth_key: ywidth_c,
                       pld_coeff_key: pld_comp_c}

        # This creates 1 joblib output file for one static aperture radius -- need to be cycled from above: change `staticRad = '2.5'` to new radius
        joblib.dump(output_dict, '{}_full_output_for_skywalker_pipeline_{}_{}_{}.joblib.save'.format(planet_dir_name, channel, staticRad, varRad))
```

### The following code is a copy/paste from a different notebook of mine.

This is the code I used to make the for loop above  
If the for loop does not work, try / check this

```python
timeCube = example_wanderer_median.timeCube
phots_array = example_wanderer_median.flux_TSO_df.values
PLDFeatures = example_wanderer_median.PLD_components.T

try:
    inliers_Phots = example_wanderer_median.inliers_Phots.values()
except:
    inliers_Phots = np.ones(photsLocal.shape)

try:
    inliers_PLD = example_wanderer_median.inliers_PLD.values()
except:
    inliers_PLD = np.ones(PLDFeatureLocal.shape)
```

```python
# Gaussian_Fit_AnnularMask_rad_2.5_0.0

staticRad = '2.5' # Need to cycle over all possible values here: [1.0, 1.5, 2.0, ..., 5.5]
varRad = '0.0'
key_flux_now = 'Gaussian_Fit_AnnularMask_rad_'+staticRad+'_'+varRad
phot_select = np.where(example_wanderer_median.flux_TSO_df.keys() == key_flux_now)[0][0]
```

```python
inliersMaster = array(list(inliers_Phots)).all(axis=0) # Need to Switch `axis=0` for Qatar-2
inliersMaster = inliersMaster * inliers_PLD.all(axis=1)
```

```python
nSig = 6 # vary this as desired

if inliersMaster.all():
    # If inliersMaster keeps ALL values, then double check with 3D inlier flagging
    print('Working on AOR {}'.format(AORNow))
    cy_now, cx_now        = example_wanderer_median.centering_GaussianFit.T
    phots_now             = phots_array[:,phot_select]

    phots_clipped         = clipOutlier2D(phots_now, nSig=nSig)
    cy_clipped, cx_clipped= clipOutlier2D(transpose([cy_now, cx_now]),nSig=nSig).T
    arr2D_clipped         = transpose([phots_clipped, cy_clipped, cx_clipped])

    # 3D inlier selection
    inliersMaster = (phots_clipped == phots_now)*(cy_clipped==cy_now)*(cx_clipped==cx_now)
else:
    print("this box is just to double check -- keeping all inlier flags from above"
```

```python
ypos, xpos = clipOutlier2D(transpose([example_wanderer_median.centering_GaussianFit.T[y][inliersMaster], \
                                           example_wanderer_median.centering_GaussianFit.T[x][inliersMaster]])).T

npix = sqrt(example_wanderer_median.effective_widths[inliersMaster])
```

```python
flux_c = phots_array[:, phot_select][inliersMaster]

# noise_c = np.sqrt(flux_c) # Photon limit
noise_c = example_wanderer_median.noise_TSO_df[key_flux_now].values[inliersMaster]

time_c = timeCube[inliersMaster]

ywidths_c, xwidths_c = example_wanderer_median.widths_GaussianFit[inliersMaster].T
```

```python
# I am guessing that this will work.
# I'm keeping the commented line because that's what I used before
# pld_comp_c = wanderer.extract_PLD_components(example_wanderer_median.imageCube, order=1)

pld_comp_c = example_wanderer_median.PLD_components.T # this is new to Carlos's notebook instance
pld_output_c = np.array(list([time_c]) + list(pld_comp_c))
```

```python
flux_key = 'phots'
time_key = 'times'
flux_err_key = 'noise'
eff_width_key = 'npix'
pld_coeff_key = 'pld'
ycenter_key = 'ycenters'
xcenter_key = 'xcenters'
ywidth_key = 'ywidths'
xwidth_key = 'xwidths'

output_dict = {time_key: time_c,
               flux_key: flux_c,
               flux_err_key: noise_c,
               eff_width_key: npix_c,
               xcenter_key: xpos_c,
               ycenter_key: ypos_c,
               xwidth_key: xwidth_c,
               ywidth_key: ywidth_c,
               pld_coeff_key: pld_comp_c}

# This creates 1 joblib output file for one static aperture radius -- need to be cycled from above: change `staticRad = '2.5'` to new radius
joblib.dump(output_dict, '{}_full_output_for_skywalker_pipeline_{}_{}_{}.joblib.save'.format(planet_dir_name, channel, staticRad, varRad))
```

    ['qatar2_full_output_for_pipeline_ch2_2.5_0.0.joblib.save']
