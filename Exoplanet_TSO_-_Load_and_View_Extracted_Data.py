import numpy as np
import os

from astropy.io import fits
from glob import glob
from plotly import graph_objs as go
from time import time

# from statsmodels.robust import scale

# TODO: make this more direct
from wanderer.wanderer import Wanderer
from wanderer.utils import command_line_inputs


def plotly_scattergl_flux_over_time(wanderer, normalise=True):
    times = wanderer.timeCube
    fluxs = wanderer.flux_TSO_df

    plots = [
        go.Scattergl(
            x=times,
            y=fluxs[colname] / np.median(fluxs[colname])
            if normalise else fluxs[colname],
            mode='markers',
            name=colname.replace('Gaussian_Fit_AnnularMask_rad_', '')
        )
        for colname in wanderer.flux_TSO_df.columns
    ]

    fig = go.Figure(data=plots)
    fig.show()


def compute_x_range_dict(ycenters, xcenters, n_sig):
    med_ycenter = ycenters.median()
    std_ycenter = ycenters.std()

    med_xcenter = xcenters.median()
    std_xcenter = xcenters.std()

    y_x_range = (
        (med_ycenter - n_sig * std_ycenter),
        (med_ycenter + n_sig * std_ycenter)
    )
    x_x_range = (
        (med_xcenter - n_sig * std_xcenter),
        (med_xcenter + n_sig * std_xcenter)
    )

    return {'y': y_x_range, 'x': x_x_range}


def plotly_scattergl_flux_vs_centers1D(
        wanderer, normalise=True, n_sig=3, fmt='-', y_range=None, x_range=None,
        colorscale='plasma', width=1600, height=800, margins=None):

    if margins is None:
        margins = {'l': 20, 'r': 20, 't': 50, 'b': 20}

    if (not isinstance(x_range, dict)
            or 'y' not in x_range
            or 'x' not in x_range):
        x_range = None

    ycenters = wanderer.centering_df['FluxWeighted_Y_Centers'].copy()
    xcenters = wanderer.centering_df['FluxWeighted_X_Centers'].copy()
    fluxs = wanderer.flux_TSO_df.copy()

    med_flux = fluxs.median()
    if normalise:
        fluxs = fluxs / med_flux

    if y_range is None and normalise:
        med_flux = fluxs.median()
        std_flux = fluxs.std()
        y_range = (
            (med_flux - n_sig * std_flux).median(),
            (med_flux + n_sig * std_flux).median()
        )

    if x_range is None and normalise:
        x_range = compute_x_range_dict(ycenters, xcenters, n_sig)

    if normalise:
        fluxs = fluxs / med_flux
        fluxs_inliers = fluxs[np.abs(fluxs - med_flux) < n_sig * std_flux]
    else:
        fluxs_inliers = fluxs.copy()

    # plasma_generator = gen_base(plasma)
    plots = []
    plots.extend([
        go.Scattergl(
            x=ycenters,
            y=fluxs_inliers[colname],
            mode='markers',
            name=colname.replace('Gaussian_Fit_AnnularMask_rad_', ''),
            # marker=dict(color=next(plasma_generator)),
            marker={'color': fluxs_inliers[colname]-1, 'colorscale':'plasma'},
            # xaxis="x",
            # yaxis="y",
        )
        for colname in wanderer.flux_TSO_df.columns
    ])
    """
    data = [
        go.Scattergl(
            y=fluxs_inliers[colname]-1,
            name=colname.replace('Gaussian_Fit_AnnularMask_rad_',''),
            mode='markers',
            marker={'color':fluxs_inliers[colname]-1, 'colorscale':'plasma'}
        )
        for colname in example_wanderer_median.flux_TSO_df.columns
    ]
    """
    # plasma_generator = gen_base(plasma)
    plots.extend([
        go.Scattergl(
            x=xcenters,
            y=fluxs_inliers[colname],
            mode='markers',
            name=colname.replace('Gaussian_Fit_AnnularMask_rad_', ''),
            # marker=dict(color=next(plasma_generator)),
            marker={'color': fluxs_inliers[colname] - \
                    1, 'colorscale':colorscale},
            xaxis="x2",
            yaxis="y2",
        )
        for colname in wanderer.flux_TSO_df.columns
    ])

    layout = go.Layout(
        title='Flux vs Centerings',
        xaxis={'range': x_range['y'], 'showgrid': False, 'title': 'y'},
        xaxis2={'range': x_range['x'], 'showgrid': False, 'title': 'x'},
        yaxis={
            'domain': [0, 0.48],
            'range': y_range,
            'showgrid': False
        },
        yaxis2={
            'domain': [0.51, 0.98],
            'anchor': 'x2',
            'range': y_range,
            'showgrid': False
        },
        width=width,
        height=height,
        margin=margins,
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)'
    )

    fig = go.Figure(data=plots, layout=layout)
    fig.show()


def plotly_scattergl_flux_vs_centers2D(
        wanderer, normalise=True, n_sig=3, fmt='-', y_range=None, x_range=None,
        columns=None, colorscale='plasma', width=1600, height=800,
        margins=None):

    if margins is None:
        margins = {'l': 20, 'r': 20, 't': 50, 'b': 20}

    if columns is None:
        columns = wanderer.flux_TSO_df.columns

    if (not isinstance(x_range, dict)
            or 'y' not in x_range
            or 'x' not in x_range):
        x_range = None

    ycenters = wanderer.centering_df['FluxWeighted_Y_Centers'].copy()
    xcenters = wanderer.centering_df['FluxWeighted_X_Centers'].copy()
    fluxs = wanderer.flux_TSO_df.copy()

    med_flux = fluxs.median()
    if normalise:
        fluxs = fluxs / med_flux

    if y_range is None and normalise:
        med_flux = fluxs.median()
        std_flux = fluxs.std()
        y_range = (
            (med_flux - n_sig * std_flux).median(),
            (med_flux + n_sig * std_flux).median()
        )

    if x_range is None and normalise:
        x_range = compute_x_range_dict(ycenters, xcenters, n_sig)

    if normalise:
        fluxs = fluxs / med_flux
        fluxs_inliers = fluxs[np.abs(fluxs - med_flux) < n_sig * std_flux]
    else:
        fluxs_inliers = fluxs.copy()

    # plasma_generator = gen_base(plasma)
    plots = [
        # Plot the centring positions with flux coloring
        go.Scattergl(
            x=xcenters,
            y=ycenters,
            mode='markers',
            name=colname.replace('Gaussian_Fit_AnnularMask_rad_', ''),
            # marker=dict(color=next(plasma_generator)),
            marker={
                'color': fluxs_inliers[colname] - 1,
                'colorscale':colorscale
            },
            xaxis="x",
            yaxis="y",
        )
        for colname in columns
    ]
    """
    data = [
        go.Scattergl(
            y=fluxs_inliers[colname]-1,
            name=colname.replace('Gaussian_Fit_AnnularMask_rad_',''),
            mode='markers',
            marker={'color':fluxs_inliers[colname]-1, 'colorscale':'plasma'}
        )
        for colname in example_wanderer_median.flux_TSO_df.columns
    ]
    """
    plots.append(
        # KDE Plot for Y-centers on Right Subplot
        go.Violin(
            y=ycenters,
            line={
                'color': 'rgb(70, 3, 159, 1.0)',
                'width': 2
            },
            name='Y-Centers',
            xaxis="x2",
            yaxis="y",
            side='positive'
        )
    )
    plots.append(
        # KDE Plot for X-centers on Upper Subplot
        go.Violin(
            x=xcenters,
            line={
                'color': 'rgb(70, 3, 159, 1.0)',
                'width': 2
            },
            name='X-Centers',
            xaxis="x",
            yaxis="y3",
            side='positive'
        )
    )

    layout = go.Layout(
        title='Flux vs Centerings',
        xaxis={'domain': [0, 0.88], 'showgrid': False, 'range': x_range['x']},
        xaxis2={'domain':  [0.9, 1], 'showgrid': False},
        # , 'range': x_range['x']},
        xaxis3={'domain': [0, 0.88], 'showgrid': False},
        # , 'range': x_range['x']},
        yaxis={'domain': [0, 0.88], 'showgrid': False, 'range': x_range['y']},
        yaxis2={'domain': [0, 0.88], 'showgrid': False},
        # , 'range': x_range['y']},
        yaxis3={'domain': [0.9, 1], 'showgrid': False},
        # , 'range': x_range['y']},

        width=width,
        height=height,
        margin=margins,

        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)'
    )

    fig = go.Figure(data=plots, layout=layout)
    fig.show()


def plotly_surface3D_plot_centers_vs_flux(
        wanderer, normalise=True, n_sig=3, fmt='-', y_range=None, x_range=None,
        columns=None, centering='FluxWeighted', colorscale='plasma',
        width=1600, height=800, margins=None):

    if margins is None:
        margins = {'l': 20, 'r': 20, 't': 50, 'b': 20}

    if columns is None:
        column = wanderer.flux_TSO_df.columns[0]

    if (not isinstance(x_range, dict)
            or 'y' not in x_range
            or 'x' not in x_range):
        x_range = None

    ycenters = wanderer.centering_df[f'{centering}_Y_Centers'].copy()
    xcenters = wanderer.centering_df[f'{centering}_X_Centers'].copy()
    fluxs = wanderer.flux_TSO_df.copy()

    med_flux = fluxs.median()
    if normalise:
        fluxs = fluxs / med_flux

    if y_range is None and normalise:
        med_flux = fluxs.median()
        std_flux = fluxs.std()
        y_range = (
            (med_flux - n_sig * std_flux).median(),
            (med_flux + n_sig * std_flux).median()
        )

    if x_range is None and normalise:
        x_range = compute_x_range_dict(ycenters, xcenters, n_sig)

    if normalise:
        fluxs = fluxs / med_flux
        fluxs_inliers = fluxs[np.abs(fluxs - med_flux) < n_sig * std_flux]
    else:
        fluxs_inliers = fluxs.copy()

    fig = go.Figure(
        data=[
            go.Surface(
                x=xcenters,
                y=ycenters,
                z=fluxs_inliers[column]
            )
        ]
    )

    fig.update_traces(
        contours_z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=True
        )
    )

    fig.update_layout(
        title=column,
        autosize=False,
        scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
        width=height,
        height=height,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    fig.show()


if __name__ == '__main__':

    clargs = command_line_inputs(check_defaults=False)

    planet_name = clargs.planet_name  # can be amended here for the planet name
    channel = clargs.channel  # or 'ch2'  # can be amened for the channel
    aor_dir = clargs.aor_dir  # or 'r11235813'  # can be amened for the aor
    planets_dir = clargs.planets_dir
    load_sub_dir = clargs.save_sub_dir
    data_sub_dir = clargs.data_sub_dir
    data_tail_dir = clargs.data_tail_dir
    fits_format = clargs.fits_format
    unc_format = clargs.unc_format
    load_file_type = clargs.save_file_type
    method = clargs.method
    telescope = clargs.telescope
    output_units = clargs.output_units
    data_dir = clargs.data_dir or 'aordirs'
    num_cores = clargs.num_cores
    verbose = clargs.verbose

    startFull = time()
    loadfiledir_parts = [
        planets_dir,
        # planet_name,
        load_sub_dir,
        channel,
        aor_dir
    ]

    print('Accessing stored file directory')
    loadfiledir = ''
    for sfpart in loadfiledir_parts:
        loadfiledir = os.path.join(loadfiledir, sfpart)
        if not os.path.exists(loadfiledir):
            os.mkdir(loadfiledir)

    print(
        '\n\n**Initializing Master Class for '
        'Exoplanet Time Series Observation Photometry**\n\n'
    )

    # As an example, Spitzer data is expected to be store in the directory structure:
    #
    # `PLANET_DIRECTORY/data/raw/AORDIR/CHANNEL/bcd/`
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
    # The `loadfitsdir` should read as:
    #   `./Research/Planets/HAPPY5/data/raw/r11235813/ch2/bcd/`

    # dataSub = f'{fits_format}/'

    if data_dir is None:
        data_dir = os.path.join(
            planets_dir,
            # planet_name,
            data_sub_dir,
            channel,
            data_tail_dir
        )

    print(f'Current Data Dir: {data_dir}')

    fileExt = f'*{fits_format}.fits'
    uncsExt = f'*{unc_format}.fits'

    loadfitsdir = os.path.join(data_dir, aor_dir, channel, fits_format, '')

    print(f'Directory to load fits files from: {loadfitsdir}')

    print(f'Found {num_cores} cores to process')

    fitsFilenames = glob(loadfitsdir + fileExt)
    uncsFilenames = glob(loadfitsdir + uncsExt)

    n_fitsfiles = len(fitsFilenames)
    n_uncfiles = len(uncsFilenames)
    print(f'Found {n_fitsfiles} {fits_format}.fits files')
    print(f'Found {n_uncfiles} unc.fits files')

    if len(fitsFilenames) == 0:
        raise ValueError(
            f'There are NO `{fits_format}.fits` files '
            f'in the directory {loadfitsdir}'
        )
    if len(uncsFilenames) == 0:
        raise ValueError(
            f'There are NO `{unc_format}.fits` files '
            f'in the directory {loadfitsdir}'
        )

    do_db_scan = False  # len(fitsFilenames*64) < 6e4
    if not do_db_scan:
        print('There are too many images for a DB-Scan; i.e. >1e5 images')

    header_test = fits.getheader(fitsFilenames[0])
    print(
        f'\n\nAORLABEL:\t{header_test["AORLABEL"]}'+'\n'
        f'Num Fits Files:\t{len(fitsFilenames)}'+'\n'
        f'Num Unc Files:\t{len(uncsFilenames)}\n\n'
    )

    if verbose:
        print(fitsFilenames)
    if verbose:
        print(uncsFilenames)

    # Necessary Constants Spitzer
    ppm = 1e6
    y, x = 0, 1

    yguess, xguess = 15., 15.   # Specific to Spitzer circa 2010 and beyond
    # Specific to Spitzer Basic Calibrated Data
    filetype = f'{fits_format}.fits'

    print('Initialize an instance of `Wanderer` as `example_wanderer_median`\n')
    example_wanderer_median = Wanderer(
        fitsFileDir=loadfitsdir,
        filetype=filetype,
        telescope=telescope,
        yguess=yguess,
        xguess=xguess,
        method=method,
        num_cores=num_cores
    )

    example_wanderer_median.AOR = aor_dir
    example_wanderer_median.planet_name = planet_name
    example_wanderer_median.channel = channel

    print(
        'Loading `example_wanderer_median` to a set of pickles for various '
        'Image Cubes and the Storage Dictionary'
    )

    load_name_header = f'{planet_name}_{aor_dir}_Median'

    path_to_files = os.path.join(
        planets_dir,
        # planet_name,
        load_sub_dir
    )
    if not os.path.exists(path_to_files):
        raise ValueError()

    if not os.path.exists(loadfiledir):
        print(f'Creating {loadfiledir}')
        os.mkdir(loadfiledir)

    load_path = os.path.join(
        loadfiledir,
        f'{load_name_header}_STRUCTURE{load_file_type}'
    )

    print()
    print(f'Loading to {load_path}')
    print()

    example_wanderer_median.load_data_from_save_files(
        savefiledir=loadfiledir,
        save_name_header=load_name_header,
        save_file_type=load_file_type
    )

    print('Entire Pipeline took {time() - startFull} seconds')

    """
    plasma = [
        'rgb(13, 8, 135, 1.0)',
        'rgb(70, 3, 159, 1.0)',
        'rgb(114, 1, 168, 1.0)',
        'rgb(156, 23, 158, 1.0)',
        'rgb(189, 55, 134, 1.0)',
        'rgb(216, 87, 107, 1.0)',
        'rgb(237, 121, 83, 1.0)',
        'rgb(251, 159, 58, 1.0)',
        'rgb(253, 202, 38, 1.0)',
        'rgb(240, 249, 33, 1.0)'
    ]

    def gen_base(iterable):
        while True:
            yield from iterable
    """

    fmt = '-'

    # example_exoplanet_tso_load_and_view_extracted_data
    plotly_scattergl_flux_over_time(
        wanderer=example_wanderer_median,
        normalise=True
    )

    plotly_scattergl_flux_vs_centers1D(
        wanderer=example_wanderer_median,
        normalise=True,
        n_sig=3,
        fmt=fmt,
        y_range=None,
        x_range=None,
        colorscale='plasma',
        width=1600,
        height=800,
        margins=None
    )

    plotly_scattergl_flux_vs_centers2D(
        wanderer=example_wanderer_median,
        normalise=True,
        n_sig=3,
        fmt=fmt,
        y_range=None,
        x_range=None,
        colorscale='plasma',
        width=1600,
        height=800,
        margins=None
    )

    plotly_surface3D_plot_centers_vs_flux(
        wanderer=example_wanderer_median,
        normalise=True,
        n_sig=3,
        fmt=fmt,
        y_range=None,
        x_range=None,
        columns=None,
        centering='FluxWeighted',
        colorscale='plasma',
        width=1600,
        height=800,
        margins=None
    )
