import h5py
import numpy as np
import astropy.units as u
import stronglensingncde.preprocessing as prep

from tqdm import tqdm
from pathlib import Path
from astropy.coordinates import SkyCoord

jolteon_path = Path('/home/jacob/PhD/Projects/jolteon_model/Data/JOLTEON_V1')
out_path = Path('/home/jacob/PhD/Projects/jolteon_model/Data/JOLTEON_V1_PREPROCESSED_V4')
out_path.mkdir(parents=True, exist_ok=True)

detection_limit = 5.
delta_trigger_min = -30.
delta_trigger_max = np.inf
asinh_scale = 0.02
test_size = 0.15
random_state = 42
max_images = 4
max_sep = 10*u.arcsec
flux_transform = lambda x: np.asinh(x/2)
flux_err_transform = np.log
redshift_columns = [
    "PHOTOZ", "PHOTOZ_ERR",
    "SPECZ", "SPECZ_ERR"
]

print('Loading dataframes...')
heads, phots, _, phots_list = prep.load_dataframes(jolteon_path)

print('Adding binary labels...')
heads = prep.add_binary_labels(heads)

print('Adding multiclass labels...')
heads = prep.add_multiclass_labels(heads, prep.label_mappings)

print('Adding number of lcs per transient event...')
heads = prep.add_num_transient_lcs(heads)

print('Adding detection flags...')
heads = prep.add_detection_flags(heads, phots_list)

heads = heads[heads['DETECTED']].reset_index(drop=True)

# Remove KNs from the dataset
heads = heads[heads['MULTICLASS_LABEL'] != 'KN'].reset_index(drop=True)

heads = prep.add_match_labels(heads, max_sep=max_sep)

# Transform metadata
for col in redshift_columns:
    heads[f'TRANS_{col}'] = np.asinh(heads[col]/asinh_scale)

train_snids, val_snids = prep.split_snids(
    heads, test_size=test_size,
    random_state=random_state,
    snid_column='MATCH_SNID',
    stratify_on='MATCH_LABELS'
)

snids = heads['MATCH_SNID'].values
idx_train = np.isin(snids, train_snids)
idx_val = np.isin(snids, val_snids)

train_heads = heads[idx_train]
val_heads = heads[idx_val]

print('Creating training set...')

prep.serialize_lightcurves(
    heads=train_heads,
    phots_list=phots_list,
    out_path=out_path,
    dataset_name="train",
    max_images=max_images,
    flux_transform=flux_transform,
    flux_err_transform=flux_err_transform
)

transformed_specz_mean = np.nanmean(train_heads['TRANS_SPECZ'])
transformed_specz_std = np.nanstd(train_heads['TRANS_SPECZ'])
transformed_photoz_mean = np.nanmean(train_heads['TRANS_PHOTOZ'])
transformed_photoz_std = np.nanstd(train_heads['TRANS_PHOTOZ'])
transformed_photoz_err_mean = np.nanmean(train_heads['TRANS_PHOTOZ_ERR'])
transformed_photoz_err_std = np.nanstd(train_heads['TRANS_PHOTOZ_ERR'])

statistics = {
    'TRANS_SPECZ_MEAN': transformed_specz_mean,
    'TRANS_SPECZ_STD': transformed_specz_std,
    'TRANS_PHOTOZ_MEAN': transformed_photoz_mean,
    'TRANS_PHOTOZ_STD': transformed_photoz_std,
    'TRANS_PHOTOZ_ERR_MEAN': transformed_photoz_mean,
    'TRANS_PHOTOZ_ERR_STD': transformed_photoz_std
}

np.savez(
    out_path / f"train_statistics.npz",
    statistics
)

print('Creating validation set...')

prep.serialize_lightcurves(
    heads=val_heads,
    phots_list=phots_list,
    out_path=out_path,
    dataset_name="train",
    max_images=max_images,
    flux_transform=flux_transform,
    flux_err_transform=flux_err_transform
)