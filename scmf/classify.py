"""
classify.py
-----------
CatBoost-based identification of Type Ia supernovae using FPCA model parameters

An object is classified as SNIa if its predicted SNIa probability
exceeds `threshold` in at least `min_votes` out of `n_runs` trials.

Functions
---------
classify : Main classification function
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILTERS          = ['g', 'r', 'i', 'z', 'y']
TRAIN_DATA_PATH  = os.path.join(os.path.dirname(__file__), 'data', 'plasticc_train_data.csv')



# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def classify(test_file='final_fpca_data.csv',
             output_file = 'potential_SNIa_candidates.csv',
             train_file  = TRAIN_DATA_PATH,
             n_runs    = 5,
             min_votes  = 3,
             threshold = 0.5): 
    """
    Classify SNe in test_file as SNIa or Non-SNIa using a CatBoost
    majority-vote ensemble.

    Parameters
    ----------
    test_file   : path to data/desirt_final_data.csv (output of postprocess.py)
    output_file : path for output CSV of SNIa candidate names
                       
    train_file  : plasticc_train_data.csv
                        
    n_runs      : number of classifier runs  (default=3)
    min_votes   : minimum votes to classify as SNIa  (default: 3)
    threshold   : float SNIa probability threshold per run  (default: 0.5)

    Returns
    -------
    df_out : pd.DataFrame  with column 'name' listing SNIa candidates
    """

    # --- Load data ----------------------------------------------------------
    df_train = pd.read_csv(train_file)
    df_test  = pd.read_csv(test_file)

    # Group all non-SN Ia as 'Non-SNIa'
    df_train.loc[df_train['type'] != 'SN Ia', 'type'] = 'Non-SNIa'

    # --- Build feature columns dynamically ----------------------------------
    # Use filters present in test data (abs_mag_ + a1_ columns)
    available_filters = [f for f in FILTERS if f'{f}_pk_mag' in df_test.columns]


    if not available_filters:
        raise ValueError(
            "No valid filter columns found in test data. "
           
        )

    feature_cols = (
    [f'{f}_pk_mag' for f in available_filters] +
    [f'{f}_a1'     for f in available_filters]+
    [f'{f}_a2'     for f in available_filters])

    
    # --- Prepare arrays -----------------------------------------------------
    x_train_val = df_train[feature_cols].to_numpy()
    y_train_val = df_train['type'].to_numpy()

    x_test  = df_test[feature_cols].to_numpy()
    x_names = df_test['name'].to_numpy()
    x_redshift = df_test['redshift'].to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.1, random_state=90
    )

    print(f"Training size : {len(x_train)}")
    print(f"Validation size : {len(x_val)}")
    print(f"Test size     : {len(x_test)}")
    print(f"Training with features      : {feature_cols}")

    # --- Majority-vote ensemble ---------------------------------------------
    seeds = [i * 10 for i in range(n_runs)]
    vote_count = {}

    for j, seed in enumerate(seeds):
        print(f"Run {j + 1}/{n_runs}  (seed={seed})")
        clf = CatBoostClassifier(random_seed=seed, verbose=0)
        clf.fit(x_train, y_train, eval_set=(x_val, y_val))
        pr = clf.predict(x_test, prediction_type='Probability')

        for i in range(len(x_test)):
            if pr[i][1] > threshold:
                vote_count[i] = vote_count.get(i, 0) + 1

    # --- Collect results ----------------------------------------------------
    SNIa_names = [x_names[i] for i, count in vote_count.items() if count >= min_votes]
    SNIa_redshifts = [x_redshift[i] for i, count in vote_count.items() if count >= min_votes]
    df_out = pd.DataFrame({'name': SNIa_names, 'redshift': SNIa_redshifts})
    df_out.to_csv(output_file, index=False)

    print(f"\nSNIa candidates : {len(SNIa_names)}")
    print(f"Written to      : {output_file}")

    return df_out


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    classify(
        test_file   = 'final_fpca_data.csv',
        output_file = 'potential_SNIa_candidates.csv',
        n_runs      = 5,
        min_votes   = 3,
        threshold   = 0.5
    )
