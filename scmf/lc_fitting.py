"""
lc_fitting.py
-------------
FPCA-based light curve fitting for DESIRT data.

Functions
---------
split_by_filter       : Split photometry arrays into per-filter dicts
compute_priors        : Compute redshift-dependent FPCA score priors
fit_filter_attempts   : Run mpfit multiple times and return best result
fit_mpfit       : Fit all available filters for one supernova
plot_lightcurve       : Plot model + data for one supernova
fit_lc         : Main loop over all light curve CSVs in lc_dir
"""

import os
import math
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import interp1d
from .mpfit import mpfit
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u



warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants  (override via config dict passed to run_pipeline if needed)
# ---------------------------------------------------------------------------

CHI_INITIALIZATION = 10000.0

CHI_FAIL_DOF = 100.0
PHASE_MIN             = -10.0
PHASE_MAX             =  40.0
PC_SCORE_LIMIT        =  10.0
MAGERR_CUT            =   0.5

# Filters whose b1 prior has a quadratic redshift term (vs linear for others)
QUADRATIC_B1_FILTERS = {'g', 'r'}

# Default plot offsets / colours / symbols per filter
DEFAULT_FILTER_CFG = {
    'g': dict(offset= 3.0, color='green', marker='s', ms=4, label='g+3.0'),
    'r': dict(offset= 1.5, color='red',   marker='*', ms=7, label='r+1.5'),
    'i': dict(offset= 0.0, color='blue',marker='o', ms=4, label='i+0.0'),
    'z': dict(offset=-1.5, color='magenta',  marker='^', ms=6, label='z-1.5'),
    'y': dict(offset=-3.0, color='black',  marker='D', ms=3, label='z-3.0'),
    

}


# ---------------------------------------------------------------------------
# Eigenvector interpolators  (call load_eigenvectors once at startup)
# ---------------------------------------------------------------------------

def load_eigenvectors(eigen_path):
    """
    Load FPCA eigenvectors from a text file and return interpolators.

    Parameters
    ----------
    eigen_path :
        Path to eigen_val.txt  (rows: phase, vec0, vec1, vec2, ...)

    Returns
    -------
    eigen_val : np.ndarray   full array (needed for plotting)
    imod0, imod1, imod2 : scipy interp1d objects
    """
    eigen_val = np.genfromtxt(eigen_path)
    ph   = eigen_val[1:, 0]
    imod0 = interp1d(ph, eigen_val[1:, 1], fill_value='extrapolate')
    imod1 = interp1d(ph, eigen_val[1:, 2], fill_value='extrapolate')
    imod2 = interp1d(ph, eigen_val[1:, 3], fill_value='extrapolate')
    return eigen_val, imod0, imod1, imod2


# ---------------------------------------------------------------------------
# Model + residual functions
# ---------------------------------------------------------------------------

def model_calc(t, theta, redshift, imod0, imod1, imod2, sigma_b1, sigma_b2):
    """
    Evaluate the FPCA model at a single time point.

    Returns (y_calc, err_pc) or (nan, nan) if outside phase window.
    """
    t_min, mag_o, a1, a2 = theta
    phase = (t - t_min) / (1.0 + redshift)

    if PHASE_MIN <= phase <= PHASE_MAX:
        v0 = float(imod0(phase))
        v1 = float(imod1(phase))
        v2 = float(imod2(phase))
        y_calc  = mag_o + v0 + a1 * v1 + a2 * v2
        err_cal = np.square(sigma_b1 * v1) + np.square(sigma_b2 * v2)   # PC score variance
        return y_calc, err_cal

    return np.nan, np.nan


def residual_func(p, fjac=None, x=None, y=None, err=None,
                   sigma_b1=None, sigma_b2=None, ini_params=None, priors=None,
                   redshift=None, imod0=None, imod1=None, imod2=None):
    """
    Residual function passed to mpfit.
    Includes Gaussian priors on a1 and a2.
    """
    dev    = np.zeros(len(x), dtype=np.float64)
    pr_arr = np.zeros(2,      dtype=np.float64)

    for k in range(len(x)):
        y_calc, err_cal = model_calc(x[k], p, redshift, imod0, imod1, imod2, sigma_b1, sigma_b2)
        if not np.isnan(y_calc):
            dev[k] = (y[k] - y_calc) / np.sqrt(np.square(err[k]) + err_cal)

    
    pr_arr[0] = (p[2] - priors[0]) / (np.sqrt(2) * sigma_b1)
    pr_arr[1] = (p[3] - priors[1]) / (np.sqrt(2) * sigma_b2)

    return [0, np.concatenate((dev, pr_arr))]


# ---------------------------------------------------------------------------
# 1. split_by_filter
# ---------------------------------------------------------------------------

def split_by_filter(mjd, mag, magerr, obs_bands, available_filters, min_points):
    """
    Split photometry arrays into per-filter dicts, drop filters with too few
    points, and return them sorted by descending number of observations
    (so the richest filter is fitted first).

    Parameters
    ----------
    mjd, mag, magerr : array-like
    obs_bands        : array-like of str  (filter label per observation)
    available_filters: list of unique filters for that particular light curve
    min_points       : int

    Returns
    -------
    ordered_filters : list of str
    data            : dict  {filt: {'time': arr, 'mag': arr, 'err': arr}}
    """
    # Accumulate into lists
    buckets = {f: {'time': [], 'mag': [], 'err': []} for f in available_filters}

    for t, m, e, b in zip(mjd, mag, magerr, obs_bands):
        if b in buckets:
            buckets[b]['time'].append(t)
            buckets[b]['mag'].append(m)
            buckets[b]['err'].append(e)

    # Convert to arrays and drop filters that don't meet min_points
    data = {}
    for f in available_filters:
        arr_t = np.array(buckets[f]['time'])
        arr_m = np.array(buckets[f]['mag'])
        arr_e = np.array(buckets[f]['err'])
        if len(arr_t) >= min_points:
            data[f] = {'time': arr_t, 'mag': arr_m, 'err': arr_e}

    # Sort by descending number of points
    ordered_filters = sorted(data, key=lambda f: -len(data[f]['time']))
    return ordered_filters, data


# ---------------------------------------------------------------------------
# 2. compute_priors
# ---------------------------------------------------------------------------

def compute_priors(filt, redshift, mapping_coeff):
    """
    Compute redshift-dependent priors for the two FPCA scores.

    g and r bands have a quadratic term in b1 (redshift²).
    i and z bands use a linear model.

    Parameters
    ----------
    filt          : filter name
    redshift      : float
    mapping_coeff : pd.DataFrame  indexed by filter name

    Returns
    -------
    b1, b2, sigma_b1, sigma_b2 : float
    """
    mc  = mapping_coeff.loc[filt]
    z   = redshift

    if filt in QUADRATIC_B1_FILTERS:
        b1       = z**2 * mc['m1'] + z * mc['c1'] + mc['c0']
        sigma_b1 = (math.pow(mc['std_m1'], 2) * z**4
                    + math.pow(mc['std_c1'], 2) * z**2
                    + math.pow(mc['std_c0'], 2))
    else:  # linear (i, z, and any future filters)
        b1       = z * mc['m1'] + mc['c1']
        sigma_b1 = np.square(mc['std_m1'] * z) + np.square(mc['std_c1'])

    b2       = z * mc['m2'] + mc['c2']
    sigma_b2 = np.square(mc['std_m2'] * z) + np.square(mc['std_c2'])

    return b1, b2, sigma_b1, sigma_b2


# ---------------------------------------------------------------------------
# 3. fit_filter_attempts
# ---------------------------------------------------------------------------

def fit_filter_attempts(times, mags, errs,
                        t_center, t_window,
                        priors, sigma_b1, sigma_b2, redshift,
                        num_attempts,
                        imod0, imod1, imod2,
                        min_points):
    """
    Run mpfit `num_attempts` times with random initial perturbations and
    return the result with the lowest chi-squared.

    Parameters
    ----------
    times, mags, errs : np.ndarray
    t_center          : float  centre of the time search window
    t_window          : float  half-width of the time constraint (days)
    priors            : [b1, b2]
    imod0/1/2         : scipy interp1d  eigenvector interpolators
    min_points        : minimum phase-window points to attempt fit

    Returns
    -------
    best_params  : np.ndarray shape (4,)  or zeros on failure
    best_errors  : np.ndarray shape (4,)  or zeros on failure
    best_chi     : CHI_INITIALIZATION
    best_leng    : int
    """
    b1, b2 = priors

    # Count points inside phase window
    phase = (times - t_center) / (1.0 + redshift)
    n_in_window = int(np.sum((phase >= PHASE_MIN) & (phase <= PHASE_MAX)))
    if n_in_window < min_points:
        return np.zeros(4), np.zeros(4), CHI_INITIALIZATION, 0

    # Build mpfit parinfo
    def _make_parinfo(p0, t_low, t_high):
        parinfo = [{'value': v, 'fixed': 0,
                    'limited': [0, 0], 'limits': [0., 0.]} for v in p0]
        # Time constraint
        parinfo[0]['limited'] = [1, 1]
        parinfo[0]['limits']  = [t_low, t_high]
        # PC score constraints
        for idx in [2, 3]:
            parinfo[idx]['limited'] = [1, 1]
            parinfo[idx]['limits']  = [-PC_SCORE_LIMIT, PC_SCORE_LIMIT]
        return parinfo

    all_params = np.zeros((num_attempts, 4))
    all_errors = np.zeros((num_attempts, 4))
    all_chi    = np.full(num_attempts, CHI_INITIALIZATION)
    all_leng   = np.zeros(num_attempts, dtype=int)

    t_low  = t_center - t_window * (1.0 + redshift)
    t_high = t_center + t_window * (1.0 + redshift)

    for attempt in range(num_attempts):
        ini_params = np.array([
            t_center + random.uniform(-3, 3) * (1.0 + redshift),
            np.min(mags),
            b1 + random.uniform(-2, 2),
            b2 + random.uniform(-2, 2),
        ])

        parinfo = _make_parinfo(ini_params, t_low, t_high)
        fa = dict(x=times, y=mags, err=errs,
                  sigma_b1=sigma_b1, sigma_b2=sigma_b2, priors=priors,
                  redshift=redshift,
                  imod0=imod0, imod1=imod1, imod2=imod2)

        m = mpfit(residual_func, parinfo=parinfo, quiet=True, functkw=fa)

        if m.status > 0:
            all_params[attempt] = m.params
            all_errors[attempt] = m.perror if m.perror is not None else np.zeros(4)
            all_chi[attempt]    = m.fnorm
            all_leng[attempt]   = n_in_window-2

    best = int(np.argmin(all_chi))
    return all_params[best], all_errors[best], all_chi[best], all_leng[best]


# ---------------------------------------------------------------------------
# 4. fit_mpfit
# ---------------------------------------------------------------------------

def fit_mpfit(lc_df, mapping_coeff, imod0, imod1, imod2,
                   num_attempts, min_points,
                   table_dir):
    """
    Fit all available filters for one supernova light curve.

    Parameters
    ----------
    lc_df            : pd.DataFrame  columns: mjd, mag, magerr, filter, redshift, name
    mapping_coeff    : pd.DataFrame  indexed by filter name
    imod0/1/2        : eigenvector interpolators
    num_attempts     : number of fits done per filter to determine the best fit
    min_points       : minimum observations per filter required for fitting
    tables_dir       : directory for per-filter CSV tables

    Returns
    -------
    results    : list of dicts  (one per successfully fitted filter)
    tables     : list of astropy Table
    """
    available_filters = list(lc_df['filter'].unique())
    os.makedirs(table_dir, exist_ok=True)

    redshift = float(lc_df['redshift'].iloc[0])
    name     = str(lc_df['name'].iloc[0])
    
    # Quality cut
    lc_df = lc_df.loc[lc_df['magerr'] < MAGERR_CUT]

    mjd    = lc_df['mjd'].to_numpy()
    mag    = lc_df['mag'].to_numpy()
    magerr = lc_df['magerr'].to_numpy()
    bands  = lc_df['filter'].to_numpy()

    ordered_filters, fdata = split_by_filter(
        mjd, mag, magerr, bands, available_filters, min_points
    )

    if not ordered_filters:
        return [], []

    results = []
    tables  = []
    t_ref   = None   # peak time from the primary (richest) filter

    for idx, filt in enumerate(ordered_filters):


        times = fdata[filt]['time']
        mags  = fdata[filt]['mag']
        errs  = fdata[filt]['err']

        b1, b2, sigma_b1, sigma_b2 = compute_priors(filt, redshift, mapping_coeff)
        
        priors   = [b1, b2]

        # Primary filter: wider time window, seed t_ref from brightness minimum
        if idx == 0:
            t_center = times[np.argmin(mags)]
            t_window = 10.0
        else:
            t_center = t_ref
            t_window = 7.0

        params, errors, chi, dof = fit_filter_attempts(
            times, mags, errs,
            t_center, t_window,
            priors, sigma_b1, sigma_b2, redshift,
            num_attempts, imod0, imod1, imod2,
            min_points
        )
        
        if (dof<=0 or chi/dof >= CHI_FAIL_DOF):
            if idx == 0:
                break   # primary filter failed → skip this SN entirely
            continue

        if idx == 0:
            t_ref = params[0]   # lock peak time for secondary filters

        row = dict(
            pk_time=params[0], pk_mag=params[1], a1=params[2], a2=params[3],
            std_time=errors[0], std_mag=errors[1], std_a1=errors[2], std_a2=errors[3],
            chi2=chi, filter=filt, name=name, dof=dof, redshift=redshift
        )
        results.append(row)

        tbl = Table(
            [times, mags, errs],
            names=('time', 'mag', 'magerr'),
            meta={'redshift': redshift, 'filter': filt, 'name': name}
        )
        tbl.write(os.path.join(table_dir, f'{name}_{filt}.csv'), overwrite=True)
        tables.append(tbl)

    return results, tables


# ---------------------------------------------------------------------------
# 5. plot_lightcurve
# ---------------------------------------------------------------------------

def plot_lightcurve(results, tables, eigen_val,
                     plot_dir):
    """
    Plot fitted model curves + observed data for one supernova.

    Parameters
    ----------
    results    : list of dicts from fit_mpfit
    tables     : list of astropy Tables from fit_mpfit
    eigen_val  : np.ndarray  full eigenvector array
    filter_cfg :  DEFAULT_FILTER_CFG
    plot_dir  : str
    """
    if not results:
        return

    os.makedirs(plot_dir, exist_ok=True)
    cfg = DEFAULT_FILTER_CFG 

    t0       = results[0]['pk_time']
    redshift = results[0]['redshift']
    name     = results[0]['name']

    for row, tbl in zip(results, tables):
        filt = row['filter']
        fcfg=cfg[filt]
        offset = fcfg['offset']

        plo_x = eigen_val[:, 0] * (1 + redshift) + row['pk_time'] - t0
        plo_y = (row['pk_mag']
                 + eigen_val[:, 1]
                 + row['a1'] * eigen_val[:, 2]
                 + row['a2'] * eigen_val[:, 3]
                 + offset)

        plt.plot(plo_x, plo_y, color=fcfg['color'], linestyle='solid', label='_nolegend_')
        plt.errorbar(
            tbl['time'] - t0,
            tbl['mag'] + offset,
            yerr=tbl['magerr'],
            fmt=fcfg['marker'],
            color=fcfg['color'],
            markersize=fcfg['ms']
        )

        coor = min(250, len(plo_x) - 1)
        plt.annotate(
            fcfg['label'],
            xy=(plo_x[coor], plo_y[coor]),
            xytext=(plo_x[coor], 0.99 * plo_y[coor]),
            color=fcfg['color']
        )

    plt.xlim([PHASE_MIN * (1 + redshift) - 5, PHASE_MAX * (1 + redshift) + 5])
    plt.xlabel('Time (Days)')
    plt.ylabel('Magnitude')
    plt.gca().invert_yaxis()

    total_chi = round(sum(r['chi2'] for r in results), 2)
    total_dof = round(sum(r['dof']  for r in results), 2)
    plt.title(f'ID={name}  z={redshift:.4f}  chisq={total_chi}  dof={total_dof}')

    plt.savefig(os.path.join(plot_dir, f'{name}.png'))
    plt.close()


# ---------------------------------------------------------------------------
# 6. run_pipeline
# ---------------------------------------------------------------------------

            

def fit_lc(lc_dir = '/Volumes/new_drive/Desirt_Work/Desirt_Lightcurves',
           eigen_path = os.path.join(os.path.dirname(__file__), 'data', 'eigen_val.txt'),
mapping_coeff_path = os.path.join(os.path.dirname(__file__), 'data', 'mapping_coeff_original.csv'),
    raw_file   = 'raw.csv',
    wide_file   = 'wide.csv',
    output_file = 'final_fpca_data.csv',
    table_dir='tables',
    plot_dir='plots',
    num_attempts       = 3,
    min_points         = 5,
    Om0=0.3,  H0=70):
    """
    Main pipeline: iterate over all light curve CSVs in lc_dir,
    fit each one, collect results, and write raw.csv.

    Parameters
    ----------
    lc_dir              :   top-level directory containing subdirs of LC CSVs
    eigen_path          :    path to eigen_val.txt
    mapping_coeff_path  :    path to mapping_coeff_original.csv
    num_attempts        :   number of fits done per filter to determine the best fit
    min_points          :    minimum observations per filter required for fitting
    raw_file            :    output raw CSV path
    table_dir           :    directory for per-filter tables
    plot_dir            :   directory for plots
    Om0                 :   matter density in Flat Lambda CDM Model
    H0                  :   Hubble Constant in Flat Lambda CDM Model
    """
    
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(plot_dir,  exist_ok=True)

    eigen_val, imod0, imod1, imod2 = load_eigenvectors(eigen_path)

    mapping_coeff = pd.read_csv(mapping_coeff_path).set_index('Filter')

    all_results = []
    total = sum(
    1 for fname in os.listdir(lc_dir)
    if os.path.join(lc_dir, fname).endswith('.csv'))
    print(total)
    processed=0

  

    for fname in os.listdir(lc_dir):
        fpath = os.path.join(lc_dir, fname)
        if not fpath.endswith('.csv'):
            continue

        lc_df = pd.read_csv(fpath)

        results, tables = fit_mpfit(
            lc_df, mapping_coeff, imod0, imod1, imod2,
            num_attempts,
            min_points,
            table_dir
        )

        if results:
            all_results.extend(results)
            plot_lightcurve(results, tables, eigen_val, 
                            plot_dir)
            
        processed += 1
        pct = int(processed / total * 100)
        prev_pct = int((processed - 1) / total * 100)
        if pct != prev_pct:
            print(f"Processed {pct}% files")

    df_out = pd.DataFrame(all_results, columns=[
        'pk_time', 'pk_mag', 'a1', 'a2',
        'std_time', 'std_mag', 'std_a1', 'std_a2',
        'chi2', 'filter', 'name', 'dof', 'redshift'
    ])
    df_out.to_csv(raw_file, index=False)
    print(f"Done. {len(df_out)} filter fits written to {raw_file}")
   

  
    # postprocess
    df_wide = make_wide(raw_file, wide_file)
    
    # absolute magnitudes
    df_final = compute_absolute_magnitudes(wide_file, output_file, Om0, H0)
    
    return df_final


# ---------------------------------------------------------------------------
# 6. make_wide
# ---------------------------------------------------------------------------
def make_wide(raw_file, wide_file):

    """
    Pivot raw.csv (one row per filter fit) to wide format
    (one row per SN, columns named <quantity>_<filter>).

    Parameters
    ----------
    raw_file  : str   path to raw.csv produced by run_pipeline
    wide_file : str   output path for wide.csv

    Returns
    -------
    df_wide : pd.DataFrame
    """


    df = pd.read_csv(raw_file)
    value_cols = ['pk_time', 'pk_mag', 'a1', 'a2',
                  'std_time', 'std_mag', 'std_a1', 'std_a2', 'chi2', 'dof']
    
    df_wide = df.pivot_table(index=['name', 'redshift'], columns='filter',
                             values=value_cols, aggfunc='first')
    df_wide.columns = [f'{filt}_{col}' for col, filt in df_wide.columns]
    df_wide = df_wide.reset_index()

    filters = sorted(df['filter'].unique())
    ordered_cols = ['name', 'redshift']
    for col in ['pk_time', 'pk_mag', 'a1', 'a2',
                'std_time', 'std_mag', 'std_a1', 'std_a2', 'chi2', 'dof']:
        for filt in filters:
            ordered_cols.append(f'{filt}_{col}')

    df_wide = df_wide[ordered_cols]
    df_wide.to_csv(wide_file, index=False)
    print(f"Wide format written to {wide_file}  ({len(df_wide)} SNe)")
    return df_wide


# ---------------------------------------------------------------------------
# 7. compute_absolute_magnitudes
# ---------------------------------------------------------------------------


def compute_absolute_magnitudes(wide_file, output_file,  Om0, H0):
    
                               
    """
    Compute absolute peak magnitudes for all filters using the distance modulus:

        M = m - 5 * log10(d_L / 10 pc)

    Parameters
    ----------
    wide_file   : str   path to wide.csv
    output_file : str   output path (e.g. data/final_fpca_data.csv)
    H0          : float Hubble constant  [km/s/Mpc]  (default 70)
    Om0         : float matter density parameter     (default 0.3)

    Returns
    -------
    df : pd.DataFrame  with new abs_mag_<filter> columns appended
    """

    cosmo  = FlatLambdaCDM(H0=H0, Om0=Om0)
    df    = pd.read_csv(wide_file)
    filters = [col.replace('_pk_mag', '') for col in df.columns if col.endswith('_pk_mag')]
    d_L_pc = cosmo.luminosity_distance(df['redshift'].to_numpy(float)).to(u.pc).value
    for filt in filters:
        df[f'{filt}_pk_mag'] = df[f'{filt}_pk_mag'] - 5.0 * np.log10(d_L_pc) + 5.0
       
    df.to_csv(output_file, index=False)
    print(f"Absolute magnitudes written to {output_file}")
    return df


##Example Usage

if __name__ == '__main__':
    fit_lc(
        lc_dir  = 'sample_lighturves',
        num_attempts = 3,  Om0=0.3, H0=70, 
        output_file = 'final_fpca_data.csv') 