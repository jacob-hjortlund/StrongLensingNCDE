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
flux_transform = lambda x: np.asinh(x/12)
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

# Transform metadata

for col in redshift_columns:
    heads[f'TRANS_{col}'] = np.asinh(heads[col]/asinh_scale)

df = heads

ras = df['RA'].values * u.deg
decs = df['DEC'].values * u.deg
max_sep = (10 * u.arcsec).to(u.deg)

coords = SkyCoord(ras, decs)
idxc, idxcat, d2d, d3d = coords.search_around_sky(coords, max_sep)

non_self = idxc != idxcat
idxc_filtered = idxc[non_self]
idxcat_filtered = idxcat[non_self]
d2d_filtered = d2d[non_self]

unique = idxc_filtered < idxcat_filtered

idxc_unique = idxc_filtered[unique]
idxcat_unique = idxcat_filtered[unique]
d2d_unique = d2d_filtered[unique]

print(f"Found {len(idxc_unique)} duplicates in the dataset")

heads['MATCH_SNID'] = heads['SNID'].values
heads.loc[idxc_unique, 'MATCH_SNID'] = heads['SNID'].values[idxcat_unique]

idx_not_lensed = heads['BINARY_LABEL'] == 'unlensed'
unlensed = heads[idx_not_lensed].reset_index(drop=True)
match_snid = unlensed['MATCH_SNID'].value_counts()
match_snids = match_snid[match_snid > 1].index

heads['MATCH_LABELS'] = heads['MULTICLASS_LABEL'].values

for snid in match_snids:

    idx_snid = heads['MATCH_SNID'] == snid
    snid_heads = heads.loc[idx_snid]
    match_label = "_".join(sorted(snid_heads['MULTICLASS_LABEL'].values))
    heads.loc[idx_snid, 'MATCH_LABELS'] = match_label

heads.loc[heads['MATCH_LABELS'].str.contains('_SLSN'), 'MATCH_LABELS'] = 'sibling_SLSN'
heads.loc[heads['MATCH_LABELS'].str.contains('Ia_Iax'), 'MATCH_LABELS'] = 'Ia_Ia'

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

transformed_fluxes_list = []
transformed_flux_errs_list = []

nmax_train = train_heads['NOBS'].max()
with h5py.File(out_path / f"train.h5", "a") as h5file:

    for snid in tqdm(train_snids):
        
        idx_snid = train_heads['MATCH_SNID'] == snid
        snid_heads = train_heads.loc[idx_snid]

        binary_label = snid_heads['BINARY_LABEL'].iloc[0]
        num_imgs = snid_heads['NUM_LCS'].iloc[0]
        num_lcs = len(snid_heads)
        joint_multiclass_labels = prep.join_multiclass_labels(snid_heads, max_images=max_images)

        snid_phots = prep.join_transient_images(snid_heads, phots_list, max_images=max_images)
        (
            transformed_snid_phots, trigger_index,
            transformed_fluxes, transformed_flux_errs
        ) = prep.transform_image_timeseries(snid_phots, flux_transform, flux_err_transform)
        trigger_time = transformed_snid_phots['MJD'].values[trigger_index]

        transformed_fluxes_list.append(transformed_fluxes)
        transformed_flux_errs_list.append(transformed_flux_errs)

        all_t_neg = np.all(transformed_snid_phots['TRANS_MJD'].values < 0)
        if all_t_neg:
            print(f"All negative times on SNID: {snid}")

        columns_to_keep = []
        if len(columns_to_keep) == 0:
            flux_cols = [col for col in transformed_snid_phots.columns if 'FLUX_' in col]
            flux_err_columns = [col for col in transformed_snid_phots.columns if 'FLUXERR_' in col]
            det_columns = [col for col in transformed_snid_phots.columns if 'DET_' in col]
            obs_columns = [col for col in transformed_snid_phots.columns if 'OBS_' in col]
            redshift_columns = [col for col in transformed_snid_phots.columns if 'Z' in col and "TRANS" in col]
            columns_to_keep = ['MJD', 'TRANS_MJD'] + flux_cols + flux_err_columns + det_columns + obs_columns + redshift_columns

        transformed_snid_phots = transformed_snid_phots[columns_to_keep]
        variable_names = ['_'.join(col.split('_')[:-1]) for col in transformed_snid_phots.columns[2:]]
        variable_names = list(dict.fromkeys(variable_names))
        flux_names = variable_names[:6]
        fluxerr_names = variable_names[6:12]
        det_obs_names = variable_names[12:24]
        redshift_names = variable_names[24:]

        flux_arrs = []
        fluxerr_arrs = []
        det_obs_arrs = []
        redshift_arrs = []

        for i in range(1, max_images + 1):
            flux_cols = [f"{var}_{i}" for var in flux_names]
            fluxerr_cols = [f"{var}_{i}" for var in fluxerr_names]
            det_obs_cols = [f"{var}_{i}" for var in det_obs_names]
            redshift_cols = [f"{var}_{i}" for var in redshift_names]

            flux_arrs.append(transformed_snid_phots[flux_cols].to_numpy())
            fluxerr_arrs.append(transformed_snid_phots[fluxerr_cols].to_numpy())
            det_obs_arrs.append(transformed_snid_phots[det_obs_cols].to_numpy())
            redshift_arrs.append(transformed_snid_phots[redshift_cols].to_numpy())

        flux_3d = np.stack(flux_arrs, axis=0)
        fluxerr_3d = np.stack(fluxerr_arrs, axis=0)
        det_obs_3d = np.stack(det_obs_arrs, axis=0)
        redshift_3d = np.stack(redshift_arrs, axis=0)

        grp = h5file.create_group(str(snid))
        grp.attrs['MULTICLASS_LABEL'] = joint_multiclass_labels.astype('S')
        grp.attrs['BINARY_LABEL'] = np.asarray(binary_label, dtype='S')
        grp.attrs['NUM_LCS'] = num_lcs
        grp.attrs['NUM_IMAGES'] = num_imgs
        grp.attrs['TRIGGER_INDEX'] = trigger_index
        grp.attrs['TRIGGER_TIME'] = trigger_time

        t_dset = grp.create_dataset(
            "MJD", data=transformed_snid_phots['MJD'].values
        )
        t_dset.attrs["column_names"] = np.array(['MJD'], dtype='S')

        trans_t_dset = grp.create_dataset(
            "TRANS_MJD", data=transformed_snid_phots['TRANS_MJD'].values
        )
        trans_t_dset.attrs["column_names"] = np.array(['MJD-MJD[0] / 1000'], dtype='S')

        flux_dset = grp.create_dataset(
            "FLUX", data=flux_3d
        )
        flux_dset.attrs["column_names"] = np.array(
            flux_names, dtype='S'
        )
        flux_err_dset = grp.create_dataset(
            "FLUX_ERR", data=fluxerr_3d
        )
        flux_err_dset.attrs["column_names"] = np.array(
            fluxerr_names, dtype='S'
        )

        detobs_dset = grp.create_dataset(
            "DETOBS", data=det_obs_3d
        )
        detobs_dset.attrs["column_names"] = np.array(
            det_obs_names, dtype='S'
        )

        redshift_dset = grp.create_dataset(
            "REDSHIFT", data=redshift_3d
        )
        redshift_dset.attrs["column_names"] = np.array(
            redshift_names, dtype='S'
        )

transformed_fluxes = np.concatenate(transformed_fluxes_list)
transformed_flux_errs = np.concatenate(transformed_flux_errs_list)

transformed_flux_mean = np.nanmean(transformed_fluxes)
transformed_flux_std = np.nanstd(transformed_fluxes)
transformed_flux_err_mean = np.nanmean(transformed_flux_errs)
transformed_flux_err_std = np.nanstd(transformed_flux_errs)
transformed_specz_mean = np.nanmean(train_heads['TRANS_SPECZ'])
transformed_specz_std = np.nanstd(train_heads['TRANS_SPECZ'])
transformed_photoz_mean = np.nanmean(train_heads['TRANS_PHOTOZ'])
transformed_photoz_std = np.nanstd(train_heads['TRANS_PHOTOZ'])
transformed_photoz_err_mean = np.nanmean(train_heads['TRANS_PHOTOZ_ERR'])
transformed_photoz_err_std = np.nanstd(train_heads['TRANS_PHOTOZ_ERR'])

statistics = {
    'TRANS_FLUX_MEAN': transformed_flux_mean,
    'TRANS_FLUX_STD': transformed_flux_std,
    'TRANS_FLUX_ERR_MEAN': transformed_flux_err_mean,
    'TRANS_FLUX_ERR_STD': transformed_flux_err_std,
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

nmax_val = val_heads['NOBS'].max()
with h5py.File(out_path / f"val.h5", "a") as h5file:

    for snid in tqdm(val_snids):
        
        idx_snid = val_heads['MATCH_SNID'] == snid
        snid_heads = val_heads.loc[idx_snid]

        binary_label = snid_heads['BINARY_LABEL'].iloc[0]
        num_imgs = snid_heads['NUM_LCS'].iloc[0]
        num_lcs = len(snid_heads)
        joint_multiclass_labels = prep.join_multiclass_labels(snid_heads, max_images=max_images)

        snid_phots = prep.join_transient_images(snid_heads, phots_list, max_images=max_images)
        transformed_snid_phots, trigger_index, _, _ = prep.transform_image_timeseries(snid_phots, flux_transform, flux_err_transform)
        trigger_time = transformed_snid_phots['MJD'].values[trigger_index]

        all_t_neg = np.all(transformed_snid_phots['TRANS_MJD'].values < 0)
        if all_t_neg:
            print(f"All negative times on SNID: {snid}")

        columns_to_keep = []
        if len(columns_to_keep) == 0:
            flux_cols = [col for col in transformed_snid_phots.columns if 'FLUX_' in col]
            flux_err_columns = [col for col in transformed_snid_phots.columns if 'FLUXERR_' in col]
            det_columns = [col for col in transformed_snid_phots.columns if 'DET_' in col]
            obs_columns = [col for col in transformed_snid_phots.columns if 'OBS_' in col]
            redshift_columns = [col for col in transformed_snid_phots.columns if 'Z' in col and "TRANS" in col]
            columns_to_keep = ['MJD', 'TRANS_MJD'] + flux_cols + flux_err_columns + det_columns + obs_columns + redshift_columns

        transformed_snid_phots = transformed_snid_phots[columns_to_keep]
        variable_names = ['_'.join(col.split('_')[:-1]) for col in transformed_snid_phots.columns[2:]]
        variable_names = list(dict.fromkeys(variable_names))
        flux_names = variable_names[:6]
        fluxerr_names = variable_names[6:12]
        det_obs_names = variable_names[12:24]
        redshift_names = variable_names[24:]

        flux_arrs = []
        fluxerr_arrs = []
        det_obs_arrs = []
        redshift_arrs = []

        for i in range(1, max_images + 1):
            flux_cols = [f"{var}_{i}" for var in flux_names]
            fluxerr_cols = [f"{var}_{i}" for var in fluxerr_names]
            det_obs_cols = [f"{var}_{i}" for var in det_obs_names]
            redshift_cols = [f"{var}_{i}" for var in redshift_names]

            flux_arrs.append(transformed_snid_phots[flux_cols].to_numpy())
            fluxerr_arrs.append(transformed_snid_phots[fluxerr_cols].to_numpy())
            det_obs_arrs.append(transformed_snid_phots[det_obs_cols].to_numpy())
            redshift_arrs.append(transformed_snid_phots[redshift_cols].to_numpy())

        flux_3d = np.stack(flux_arrs, axis=0)
        fluxerr_3d = np.stack(fluxerr_arrs, axis=0)
        det_obs_3d = np.stack(det_obs_arrs, axis=0)
        redshift_3d = np.stack(redshift_arrs, axis=0)

        grp = h5file.create_group(str(snid))
        grp.attrs['MULTICLASS_LABEL'] = joint_multiclass_labels.astype('S')
        grp.attrs['BINARY_LABEL'] = np.asarray(binary_label, dtype='S')
        grp.attrs['NUM_LCS'] = num_lcs
        grp.attrs['NUM_IMAGES'] = num_imgs
        grp.attrs['TRIGGER_INDEX'] = trigger_index
        grp.attrs['TRIGGER_TIME'] = trigger_time

        t_dset = grp.create_dataset(
            "MJD", data=transformed_snid_phots['MJD'].values
        )
        t_dset.attrs["column_names"] = np.array(['MJD'], dtype='S')

        trans_t_dset = grp.create_dataset(
            "TRANS_MJD", data=transformed_snid_phots['TRANS_MJD'].values
        )
        trans_t_dset.attrs["column_names"] = np.array(['MJD-MJD[0] / 1000'], dtype='S')

        flux_dset = grp.create_dataset(
            "FLUX", data=flux_3d
        )
        flux_dset.attrs["column_names"] = np.array(
            flux_names, dtype='S'
        )
        flux_err_dset = grp.create_dataset(
            "FLUX_ERR", data=fluxerr_3d
        )
        flux_err_dset.attrs["column_names"] = np.array(
            fluxerr_names, dtype='S'
        )

        detobs_dset = grp.create_dataset(
            "DETOBS", data=det_obs_3d
        )
        detobs_dset.attrs["column_names"] = np.array(
            det_obs_names, dtype='S'
        )

        redshift_dset = grp.create_dataset(
            "REDSHIFT", data=redshift_3d
        )
        redshift_dset.attrs["column_names"] = np.array(
            redshift_names, dtype='S'
        )