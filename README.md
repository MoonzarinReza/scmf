# SCMF
**Supernova Classification with Machine Learning and FPCA**

SCMF is a Python package for binary photometric classification of transients to identify Type Ia supernovae (SNe Ia) using Functional Principal Component Analysis (FPCA) for light curve fitting and Gradient Boosting Decision Trees for classification. Given a set of light curves, SCMF outputs a file identifying potential SNe Ia candidates.


## What is FPCA?
Functional Principal Component Analysis (FPCA) is a data-driven decomposition method  that represents light curves as a linear combination of weighted eigenvectors in the phase space:


    g(q) = m + φ_0(q) + a_1*φ_1(q) + a_2*φ_2(q)

where m is a fitted parameter representing the apparent peak magnitude, φ_0, φ_1, φ_2 are the mean template and principal eigenvectors learned from a training set of SNe Ia light curves, and a_1, a_2 are the FPCA scores that characterize each individual light curve (He et al. 2018).

## Fitting light curves
- Multi-band FPCA light curve fitting using Levenberg-Marquardt optimization (mpfit)
- Redshift-dependent  priors on FPCA scores
- Peak time consistency enforced across all filters
- Supports grizy photometric bands
- Designed for LSST/Rubin and DES  data

## Classification
- Features used for classification are the FPCA scores in the available photometric bands and m converted to absolute magnitudes assuming a fixed cosmology. 
- CatBoost is used as the base classifier within a majority-vote ensemble (n_runs, min_votes, and threshold are adjustable).

## Installation
pip install scmf


## Repository
https://github.com/MoonzarinReza/scmf

## Example
A full worked example is available in the [example notebook]
https://github.com/MoonzarinReza/scmf/blob/master/scmf/example/example.ipynb

## Usage

### Step 1: Fit Light Curves
from scmf import fit_lc

fit_lc(lc_dir = '/path/to/lightcurves')


Parameters
----------
lc_dir  : str, required
   Root directory containing the light curve CSV files. Each CSV file corresponds to one transient and must contain the following columns: mjd, mag, magerr, filter, redshift, name.
num_attempts : int, optional
    Number of times the best-fit parameters are determined; the optimum solution is selected based on the minimum chi-square value. Default is 3.


Outputs:
- raw.csv              : per-filter fit results
- wide.csv             : wide format (one row per SN)
- final_fpca_data.csv  : same as wide.csv except that apparent magnitudes are replaced by absolute magnitudes, ready for classification


### Step 2: Classify
from scmf import classify

classify(n_runs = 5, min_votes  = 3, threshold = 0.5)

Parameters
----------
n_runs : int, optional
    Number of times the experiment is repeated. Default is 5.

min_votes : int, optional
    Minimum number of runs that must predict SNe Ia for a transient to be accepted as a candidate. Default is 3.

threshold : float, optional
    Predicted probability above which a transient is determined as SNe Ia in a given run. Must be between 0 and 1. Default is 0.5.


Outputs:
- potential_SNIa_candidates.csv : names and redshifts of SNIa candidates

## Dependencies
- numpy, pandas, scipy, matplotlib
- astropy, catboost, scikit-learn

## Author
Moonzarin Reza, Texas A&M University

## Citation
If you use SCMF in your research, please cite:

Reza, M., Wang, L., & Hu, L. (2025). An FPCA-Enhanced Ensemble Learning Framework for Photometric Identification of Type Ia Supernovae, arXiv:2510.09990.