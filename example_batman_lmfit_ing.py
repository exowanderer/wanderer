import batman
import emcee
import exoparams

from astropy.io         import fits
from batman             import TransitParams, TransitModel
from glob               import glob
from functools          import partial
from lmfit              import Minimizer, Parameters, report_errors
from matplotlib.colors  import LogNorm
from pandas             import DataFrame, rolling_median
from photutils          import RectangularAperture, RectangularAnnulus, aperture_photometry
from pylab              import *;ion()
from statsmodels.robust import scale
from time               import time
from tqdm               import tqdm

rcParams['figure.dpi']  = 300.
rcParams['savefig.dpi'] = 300.

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
        
        if transittype=="secondary" and edepth != 0.0:
            bm_params.delta_phase = deltaphase_eclipse(bm_params.ecc, bm_params.w) if bm_params.ecc is not 0.0 else 0.5
            bm_params.t_secondary = bm_params.t0 + bm_params.per*bm_params.delta_phase
        # print(times)
        m_eclipse = TransitModel(bm_params, times, transittype=transittype).light_curve(bm_params)
    else:
        return ones(times.size)
    
    return m_eclipse

def residuals_func(model_params, times, data, data_err, ldtype='uniform'):
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
    period  = model_params['period']
    tCenter = model_params['tCenter']
    inc     = model_params['inc']
    aprs    = model_params['aprs']
    edepth  = model_params['edepth']
    tdepth  = model_params['tdepth']
    ecc     = model_params['ecc']
    omega   = model_params['omega']
    
    u1      = model_params['u1']
    u2      = model_params['u2']
    intcpt  = model_params['intcpt']
    slope   = model_params['slope']
    crvtur  = model_params['crvtur']
    
    model_now = batman_lmfit_model(period, tCenter, inc, aprs, edepth, tdepth, ecc, omega, times, u1=u1, u2=u2, ldtype=ldtype)
    
    line_model  = intcpt
    if slope != 0:
        line_model += slope*(times - times.mean())
    if crvtur != 0:
        line_model += crvtur*(times - times.mean())**2
    
    model_total = model_now*line_model
    
    return ((model_total - data) / data_err)**2

def exoparams_to_lmfit_params(planet_name):
    ep_params   = exoparams.PlanetParams(planet_name)
    iApRs       = ep_params.ar.value
    iEcc        = ep_params.ecc.value
    iInc        = ep_params.i.value
    iPeriod     = ep_params.per.value
    iTCenter    = ep_params.tt.value
    iTdepth     = ep_params.depth.value
    iOmega      = ep_params.om.value
    
    return iApRs, iEcc, iInc, iPeriod, iTCenter, iTdepth, iOmega

def exoparams_to_batman_model(planet_name, times=None, u1=None, u2=None):
    ep_params           = exoparams.PlanetParams(planet_name)
    bm_params           = TransitParams()

    bm_params.a         = ep_params.ar.value
    bm_params.ecc       = ep_params.ecc.value
    bm_params.inc       = ep_params.i.value
    bm_params.limb_dark = 'uniform'
    bm_params.per       = ep_params.per.value
    bm_params.t0        = ep_params.tt.value
    bm_params.rp        = sqrt(ep_params.depth.value)
    bm_params.w         = ep_params.om.value
    
    ldcs  = []
    if u1 is not None:
        bm_params.limb_dark = 'linear'
        ldcs.append(u1)
        if u2 is not None:
            bm_params.limb_dark = 'quadratic'
            ldcs.append(u2)    
    
    bm_params.u         = ldcs
    if times is not None:
        times = times.copy()
        if len(str(times[0]).split('.')[0]) == 5:
            times += 2400000
        if len(str(times[0]).split('.')[0]) == 4:
            times += 2450000
        
        lc_model  = TransitModel(bm_params, times).light_curve(bm_params)
        
        return lc_model, bm_params, ep_params
    else:
        return bm_params, ep_params


## FITTING STEPS ###
iApRs, iEcc, iInc, iPeriod, iTCenter, iTdepth, iOmega = exoparams_to_lmfit_params('Kepler-62 b')

partial_residuals = partial(residuals_func, data     =      use_lc['Flux2'].values  / median(use_lc['Flux2'].values),
                                            data_err = sqrt(use_lc['Flux2'].values) / median(use_lc['Flux2'].values),
                                            times    =      use_lc['MJD'].values)

nEpoch      = np.int(np.round(abs(iTCenter_init - mean_time) / iPeriod))

init_params_lmfit = Parameters()
init_params_lmfit.add_many(
    # Planetary Parameters
    ('period' , iPeriod , False),
    ('tCenter', iTCenter, True , iTCenter-0.05, iTCenter+0.05), # +\- 1.2 hours
    ('inc'    , iInc    , False, 80.0 ,  90.), # Physically down to 0.0, but 80.0 is already an 'extreme' value
    ('aprs'   , iApRs   , False,  0.0 , 100.), # 100.0 is considered "very large" for transiting exoplanets; but could go up to np.inf
    ('edepth' , iEdepth , False,  0.0 , 1.0 ), # 1.0   is considered "very large" for transiting exoplanets -- could go down to 0.1
    ('tdepth' , iTdepth , False,  0.0 , 1.0 ), # 1.0   is considered "very large" for transiting exoplanets -- could go down to 0.25 
    ('ecc'    , iEcc    , False,  0.0 , 1.0 ), # 1.0   is considered "very large" for transiting exoplanets -- could go down to 0.5
    ('omega'  , iOmega  , False,  0.0 , 360.), # Could be any number around a circle; but prescribing to "near RV results" is probably best
    ('u1'     , u1      , False,  0.0 , 1.0 ), # could go up to 1; but physically, u1 + u2 < 1.0; so u1 and u2 cannot both be 1.0
    ('u2'     , u2      , False,  0.0 , 1.0 ), # could go up to 1; but physically, u1 + u2 < 1.0; so u1 and u2 cannot both be 1.0
    
    # Out of transit linear baselines
    ('intcpt' , 1.0 , True ),  # If the 'linear model' is centered around the median(time), then 1.0 should be a useful initial condition
    ('slope'  , 0.0 , True),   # If the 'linear model' is centered around the median(time), then 0.0 should be a useful initial condition
    ('crvtur' , 0.0 , False))  # If the 'linear model' is centered around the median(time), then 0.0 should be a useful initial condition

mini = Minimizer(partial_residuals, init_params_lmfit)

fitResults_lmfit = mini.leastsq()


'''--------------------------------------------------'''
''' The Emcee Model Selection and Setup is UNTESTED! '''
'''--------------------------------------------------'''

'''
    EMCEE inputs from LMFIT output
    
        Create lnfunc and starting distribution.
        Modeled after: https://github.com/lmfit/lmfit-py/blob/master/examples/lmfit_and_emcee.py
'''
def create_prior(params):
    """
    emccee uses a uniform prior for every variable.
    Here we create a functions which checks the bounds
    and returns np.inf if a value is outside of its
    allowed range. WARNING: A uniform prior may not be
    what you want!
    """
    none_to_inf = lambda x, sign=1: sign*np.inf if x is None else x
    lower_bounds = np.array([none_to_inf(i.min, -1) for i in params.values() if i.vary])
    upper_bounds = np.array([none_to_inf(i.max, 1) for i in params.values() if i.vary])

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
        residuals = mini.residual
        if not sigma:
            # sigma is either the error estimate or it will
            # be part of the sampling.
            sigma = vals[-1]
        val = -0.5*np.sum(np.log(2*np.pi*sigma**2) + (residuals/sigma)**2)
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
    lnprob = create_lnliklihood(mini, sigma=sigma)
    guess = starting_guess(mini, not sigma_given)
    if sigma_given:
        func = lambda x: lnprior(x[:]) + lnprob(x)
    else:
        func = lambda x: lnprior(x[:-1]) + lnprob(x)
    return func, guess

lnfunc, guess = create_all(fitResults_lmfit)

nwalkers, ndim = 100, len(guess)
p0 = emcee.utils.sample_ball(guess, 0.1*np.array(guess), nwalkers)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnfunc)
n_steps = 10000

start = time()
sampler.run_mcmc(p0, n_steps, progress=True)
print("Full phase curve fitting operation took {} seconds".format(time()-start))

params      = fitResults_lmfit.params
lmfit_vals  = [i.value for i in params.values() if i.vary] + [0.0]
lmfit_names = [i.name  for i in params.values() if i.vary] + ['sigma']

fig, axes = plt.subplots(len(guess), 1, sharex=True, figsize=(8, 2*len(guess)))
for (i, name, rv) in zip(range(len(guess)), lmfit_names, lmfit_vals):
    # axes[i].hist(sampler.chain[:, :, i].T.ravel(),  alpha=0.05, bins=sampler.chain[:, :, i].T.size//100);
    axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.05);
    axes[i].yaxis.set_major_locator(plt.MaxNLocator(5));
    axes[i].axhline(rv, color="#888888", lw=2);
    axes[i].set_ylabel("$%s$" % name);

axes[-1].set_xlabel("Steps");

i_use, corner_vals, corner_names = np.transpose([(k, val, name) for k, (val, name) in enumerate(zip(lmfit_vals, lmfit_names))])
i_use = int32(i_use)

corner_names=[r'tCenter', r'inc', r'aprs', r'edepth', r'tdepth', r'u1', r'u2', r'intcpt', r'slope', r'sigma']

# samples.shape, dims, len(i_use), len(corner_names)

burnin  = int(n_steps * 0.2)
samples = sampler.chain[:, burnin:, i_use].reshape((-1, len(i_use)));
# corner.corner(samples, labels=corner_names, truths=corner_vals);

#Stairstep Plot
# labels = corner_names

plt.rc('font',size=8)
dims = len(corner_names)
# fig,axL = plt.subplots(nrows=dims,ncols=dims,figsize=(15,15))

corner_kw = dict(
    truths=corner_vals,
    labels=corner_names,
    levels=[0.68,0.95],
    plot_datapoints=False,
    smooth=True,
    bins=30,
    )

corner.corner(samples, **corner_kw)
