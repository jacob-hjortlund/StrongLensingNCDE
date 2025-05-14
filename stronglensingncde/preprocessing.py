import numpy as np
import pandas as pd
import astropy.table as at

from tqdm import tqdm
from pathlib import Path
from astropy.io import fits
from functools import reduce
from sklearn.model_selection import train_test_split


label_mappings = {
    "Ia": ["SNIa-SALT3"],
    "Iax": ["SNIax"],
    "91bg": ["SNIa-91bg"],
    "Ib/c": [
        'SNIb+HostXT_V19', 'SNIc+HostXT_V19',
        'SNIcBL+HostXT_V19',
    ],
    "II": [
        'SNII+HostXT_V19', 'SNIIb+HostXT_V19',
        'SNIIn+HostXT_V19'
    ],
    "TDE": ['TDE'],
    "KN": ['KN_K17', 'KN_B19'],
    "PISN": ['PISN-STELLA_HYDROGENIC', 'PISN-STELLA_HECORE'],
    "SLSN": ['SLSN-I_no_host', 'SLSN-I+host'],
    "AGN": ["AGN"],
    "glIa": ['glSNIa_ASM'],
    "glCC": ['glSNCC_ASM'],
    "glKN": ['glKN_AG']
}

def load_dataframes(data_dir: str, show_progress_bar=True) -> tuple[pd.DataFrame, pd.DataFrame]:

    data_dir = Path(data_dir)
    heads = sorted(data_dir.glob("*HEAD.FITS"))
    phots = sorted(data_dir.glob("*PHOT.FITS"))
    disable_progress_bar = not show_progress_bar

    head_dfs = []
    phot_dfs = []
    for idx in tqdm(range(len(heads)), disable=disable_progress_bar):

        head = at.Table(fits.open(heads[idx])[1].data).to_pandas()
        head['FILE_INDEX'] = idx
        head_dfs.append(head)

        phot = at.Table(fits.open(phots[idx])[1].data).to_pandas()
        phot['FILE_INDEX'] = idx
        phot_dfs.append(phot)

    heads_df = pd.concat(head_dfs, ignore_index=True)
    phots_df = pd.concat(phot_dfs, ignore_index=True)

    return heads_df, phots_df, head_dfs, phot_dfs

def add_binary_labels(heads: pd.DataFrame) -> pd.DataFrame:

    _heads = heads.copy()
    idx_lensed = _heads['LABEL'].str.contains("gl")

    _heads['BINARY_LABEL'] = 'unlensed'
    _heads.loc[idx_lensed, 'BINARY_LABEL'] = 'lensed'

    return _heads

def add_multiclass_labels(heads: pd.DataFrame, label_mapping: dict) -> pd.DataFrame:

    _heads = heads.copy()
    _heads['MULTICLASS_LABEL'] = None

    transient_model_names = _heads['LABEL']
    for class_label, class_transient_model_names in label_mapping.items():
        idx = transient_model_names.isin(class_transient_model_names)
        _heads.loc[idx, 'MULTICLASS_LABEL'] = class_label

    if _heads['MULTICLASS_LABEL'].isna().any():
        raise ValueError("Some transient models do not have a class label. Please review the label mapping.")
    
    return _heads

def add_num_transient_lcs(heads: pd.DataFrame) -> pd.DataFrame:

    _heads = heads.copy()
    snid_counts = _heads['SNID'].value_counts()
    _heads['NUM_LCS'] = _heads['SNID'].map(snid_counts)

    return _heads

def add_detection_flags(heads: pd.DataFrame, phots: list[pd.DataFrame], snr_limit: float = 5., show_progress_bar=True) -> pd.DataFrame:

    _heads = heads.copy()
    disable_progress_bar = not show_progress_bar

    _heads['DETECTED'] = False

    for i in tqdm(range(len(heads)), disable=disable_progress_bar):

        file_index = heads['FILE_INDEX'].iloc[i]
        ptr_min = heads['PTROBS_MIN'].iloc[i]-1
        ptr_max = heads['PTROBS_MAX'].iloc[i]

        _phot = phots[file_index]
        indeces = _phot.index[ptr_min:ptr_max]

        flux = _phot.loc[indeces, 'FLUXCAL']
        flux_err = _phot.loc[indeces, 'FLUXCALERR']
        snr = flux / flux_err
        idx_detection = snr >= snr_limit
        n_detections = idx_detection.sum()
        
        _heads.loc[i, 'DETECTED'] = n_detections > 0
    
    return _heads

def split_snids(
        heads: pd.DataFrame,
        snid_column: str = 'MATCH_SNID',
        stratify_on: str = 'MULTICLASS_LABELS',
        test_size: float = 0.15,
        random_state: int = 42,
        shuffle: bool = True
) -> tuple[np.ndarray, np.ndarray]:

    filtered_unique_heads = heads.drop_duplicates(subset=snid_column)
    unique_snids = filtered_unique_heads[snid_column].values
    class_labels = filtered_unique_heads[stratify_on].values
    train_snids, val_snids = train_test_split(
        unique_snids, test_size=test_size, stratify=class_labels,
        random_state=random_state, shuffle=shuffle
    )

    return train_snids, val_snids

def get_light_curve(
    head: dict,
    phot_list: list[pd.DataFrame],
    columns_to_add: list[str] = [
        'TRANS_SPECZ', 'TRANS_SPECZ_ERR',
        'TRANS_PHOTOZ', 'TRANS_PHOTOZ_ERR'
    ]
) -> pd.DataFrame:

    file_index = head['FILE_INDEX']
    ptr_min = head['PTROBS_MIN']-1
    ptr_max = head['PTROBS_MAX']

    phot = phot_list[file_index][ptr_min:ptr_max]
    phot = phot.iloc[::-1].reset_index(drop=True)
    for col in columns_to_add:
        phot[col] = head[col]
    
    return phot

def process_light_curve(
    phot: pd.DataFrame,
    detection_limit: float = 5.,
    delta_trigger_max: float = np.inf,
    delta_trigger_min: float = -30.,
    bands: list[str] = ['u','g', 'r', 'i', 'z', 'Y'],
    added_columns: list[str] = [
        'TRANS_SPECZ', 'TRANS_SPECZ_ERR',
        'TRANS_PHOTOZ', 'TRANS_PHOTOZ_ERR'
    ]
) -> pd.DataFrame:

    mjd = phot['MJD']
    snr = phot['FLUXCAL'] / phot['FLUXCALERR']
    idx_detection = snr >= detection_limit
    phot['DETECTION'] = 0.
    phot.loc[idx_detection, 'DETECTION'] = 1.
    
    idx_trigger = np.where(phot['DETECTION'])[0][0]
    t_trigger = mjd[idx_trigger]
    delta_trigger = mjd - t_trigger
    idx_excluded = (delta_trigger < delta_trigger_min) | (delta_trigger > delta_trigger_max)
    phot.loc[idx_excluded, ['FLUXCAL', 'FLUXCALERR']] = np.nan

    for band in bands:

        phot[f'{band}_FLUX'] = np.nan
        phot[f'{band}_FLUXERR'] = np.nan
        phot[f'{band}_DET'] = 0.

        idx_band = phot['BAND'] == band
        idx_band_and_detection = idx_band & idx_detection

        phot.loc[idx_band, f'{band}_FLUX'] = phot.loc[idx_band, 'FLUXCAL']
        phot.loc[idx_band, f'{band}_FLUXERR'] = phot.loc[idx_band, 'FLUXCALERR']
        phot.loc[idx_band_and_detection, f'{band}_DET'] = 1.
        phot[f'{band}_OBS'] = np.cumsum(~np.isnan(phot[f'{band}_FLUX']))
    
    idx_all_nan = np.all(np.isnan(phot[[f'{band}_FLUX' for band in bands]]), axis=1)
    phot.loc[idx_all_nan, added_columns] = np.nan

    return phot

def merge_frames(data_frames: list[pd.DataFrame], on='MJD') -> pd.DataFrame:

    return reduce(lambda left, right: pd.merge(left, right, on=on, how='outer'), data_frames)

def join_transient_images(
    snid_heads,
    phots,
    max_images = 4,
    added_columns: list[str] = [
        'SPECZ', 'SPECZ_ERR',
        'PHOTOZ', 'PHOTOZ_ERR',
        'TRANS_SPECZ', 'TRANS_SPECZ_ERR',
        'TRANS_PHOTOZ', 'TRANS_PHOTOZ_ERR'
    ]
):
    
    snid_phots = []
    for i, (_, img_head) in enumerate(snid_heads.iterrows()):
        
        img_phot = get_light_curve(img_head, phots, columns_to_add=added_columns)
        img_phot = process_light_curve(img_phot, added_columns=added_columns)

        columns = img_phot.columns
        cols_to_rename = columns.delete(0)
        name_mapping = {
            col: f"{col}_{i+1}" for col in cols_to_rename
        }
        img_phot = img_phot.rename(columns=name_mapping)

        snid_phots.append(img_phot)


    merged_snid_phot = merge_frames(snid_phots, on='MJD')
    n_images = len(snid_phots)
    unmodified_columns = cols_to_rename
    detobs_columns = [col for col in unmodified_columns if '_DET' in col or '_OBS' in col]
    indeces = merged_snid_phot.index

    empty_phots = []
    for i in range(n_images, max_images):
        
        empty_phot = pd.DataFrame(np.nan, index=indeces, columns=unmodified_columns)
        empty_phot[detobs_columns] = 0.
        
        name_mapping = {
            col: f"{col}_{i+1}" for col in cols_to_rename
        }
        empty_phot = empty_phot.rename(columns=name_mapping)

        empty_phots.append(empty_phot)
    
    merged_snid_phot = pd.concat([merged_snid_phot] + empty_phots, axis=1)
    regex = "|".join(['MJD', '_FLUX', '_DET', '_OBS'] + added_columns)
    merged_snid_phot = merged_snid_phot.filter(regex=regex)

    flux_cols = merged_snid_phot.filter(like='_FLUX')
    idx_valid_times = ~np.all(np.isnan(flux_cols.values), axis=1)
    merged_snid_phot = merged_snid_phot[idx_valid_times].reset_index(drop=True)

    return merged_snid_phot

def join_multiclass_labels(
    snid_heads: pd.DataFrame,
    max_images: int = 4,
) -> np.ndarray:
    
    n_snid = len(snid_heads)
    n_remaining = max_images - n_snid
    labels = snid_heads['MULTICLASS_LABEL'].values
    labels = np.concatenate([labels, ['None'] * n_remaining])

    return labels


def transform_image_timeseries(
    image_timeseries: pd.DataFrame,
    flux_transform: callable,
    fluxerr_transform: callable,
) -> pd.DataFrame:

    image_timeseries = image_timeseries.copy()
    
    det_columns = [col for col in image_timeseries.columns if '_DET' in col]
    t = image_timeseries['MJD'].values
    trigger_index = np.where(
        np.any(image_timeseries[det_columns], axis=1)
    )[0][0]
    t_trigger = t[trigger_index]
    

    flux_columns = [col for col in image_timeseries.columns if '_FLUX' in col and 'ERR' not in col]
    fluxerr_columns = [col for col in image_timeseries.columns if '_FLUXERR' in col]

    transformed_flux = flux_transform(image_timeseries[flux_columns].values)
    image_timeseries[flux_columns] = transformed_flux
        
    transformed_flux_errs = fluxerr_transform(image_timeseries[fluxerr_columns].values)
    image_timeseries[fluxerr_columns] = transformed_flux_errs

    image_timeseries['TRANS_MJD'] = (image_timeseries['MJD'].values - t_trigger) / 1000.

    new_column_names = [(old_name, 'TRANS_' + old_name) for old_name in flux_columns + fluxerr_columns]
    image_timeseries = image_timeseries.rename(columns=dict(new_column_names))


    return image_timeseries, trigger_index