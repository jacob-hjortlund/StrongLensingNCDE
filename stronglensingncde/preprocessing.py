import h5py
import warnings
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.table as at

from tqdm import tqdm
from pathlib import Path
from astropy.io import fits
from functools import reduce
from collections import defaultdict
from sklearn.model_selection import train_test_split
from astropy.coordinates import SkyCoord, SkyOffsetFrame

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

def add_detection_flags(heads, phots, snr_limit=5.0, show_progress_bar=True):
    # 1 copy of heads and phots
    out_heads = heads.copy()
    out_heads['DETECTED'] = False

    out_phots = []
    for p in phots:
        cp = p.copy()
        cp['DETECTION'] = 0.0
        cp['SNR']       = np.nan
        out_phots.append(cp)

    # group heads by which phot they refer to
    grouped = out_heads.groupby('FILE_INDEX', sort=False)

    for file_idx, head_block in tqdm(grouped, disable=not show_progress_bar):
        phot = out_phots[file_idx]
        # pull out the .values once for speed
        flux_arr   = phot['FLUXCAL'].values
        ferr_arr   = phot['FLUXCALERR'].values
        snr_arr    = phot['SNR'].values
        det_arr    = phot['DETECTION'].values

        # for each head in that phot, update the arrays
        for i, head in head_block.iterrows():
            i0 = head['PTROBS_MIN'] - 1
            i1 = head['PTROBS_MAX']
            rng = np.arange(i0, i1)

            # compute
            slice_snr = np.abs(flux_arr[rng] / ferr_arr[rng])
            snr_arr[rng] = slice_snr

            mask = slice_snr >= snr_limit
            det_arr[rng[mask]] = 1.0

            # mark head as detected if any True in mask
            out_heads.at[i, 'DETECTED'] = bool(mask.any())

        # arrays mutated in-place, so just reassign
        out_phots[file_idx] = phot

    return out_heads, out_phots

def rescale_lensed_coords(heads: pd.DataFrame, scale=1/3600) -> pd.DataFrame:

    _heads = heads.copy()
    mask = _heads['NUM_LCS'] > 1

    for snid, group in _heads.loc[mask].groupby('SNID'):
        idx = group.index
        ras = group['RA'].values * u.deg
        decs = group['DEC'].values * u.deg

        # original coords
        coords = SkyCoord(ras, decs)

        # pick the first image as tangent point
        ref = coords[0]
        frame = SkyOffsetFrame(origin=ref)

        # offsets in ARC seconds
        offsets = coords.transform_to(frame)
        scaled = SkyCoord(lon=offsets.lon * scale,
                        lat=offsets.lat * scale,
                        frame=frame)

        new_coords = scaled.transform_to('icrs')
        _heads.loc[idx, 'RA']  = new_coords.ra.deg
        _heads.loc[idx, 'DEC'] = new_coords.dec.deg

    return _heads

def sample_light_curve_coords(
    head: pd.Series, phots: pd.DataFrame,
    astrometric_error_floor: float = 0.05,
    pixel_to_arcsec: float = 0.2
) -> pd.DataFrame:

    ra0, dec0 = head['RA'], head['DEC']
    i0, i1 = head['PTROBS_MIN']-1, head['PTROBS_MAX']
    block = phots.iloc[i0:i1]
    detected = block['DETECTION'].values == 1.0
    if not detected.any():
        return phots

    # get integer positions within the slice
    slice_pos = np.nonzero(detected)[0]
    abs_pos   = slice_pos + i0

    psf  = block['PSF'].values[detected] * pixel_to_arcsec
    fwhm = 2*np.sqrt(2*np.log(2)) * psf
    snr  = block['SNR'].values[detected]
    err  = np.sqrt((fwhm/snr)**2 + astrometric_error_floor**2)
    err_deg = err / 3600.0

    d_ra, d_dec = np.random.normal(0, err_deg, size=(2,len(err_deg)))
    d_ra /= np.cos(np.deg2rad(dec0))

    # pull out the underlying arrays
    ra_arr, dec_arr, err_arr = (
        phots['RA'].values,
        phots['DEC'].values,
        phots['POS_ERR'].values,
    )

    # vectorized assignment
    err_arr[abs_pos]   = err
    ra_arr[abs_pos]    = ra0  + d_ra
    dec_arr[abs_pos]   = dec0 + d_dec

    # Fill Forced Photometry Before First Detection
    if abs_pos[-1] < i1:
        first_err = err[-1]
        first_ra = ra0 + d_ra[-1]
        first_dec = dec0 + d_dec[-1]
        forced_pos = np.arange(abs_pos[-1]+1, i1)

        err_arr[forced_pos] = first_err
        ra_arr[forced_pos] = first_ra
        dec_arr[forced_pos] = first_dec

    # pandas will see the mutated arrays
    return phots

def add_coords_to_timeseries(heads, phots_list):
    # make one copy of each phot, set up the three new columns
    phots_list = [p.copy() for p in phots_list]
    for p in phots_list:
        p['RA']      = np.nan
        p['DEC']     = np.nan
        p['POS_ERR'] = np.nan

    # group heads by which phot they belong to
    for (file_idx, heads_grp) in tqdm(heads.groupby('FILE_INDEX')):
        phot = phots_list[file_idx]  # one copy per file, not per head
        
        # process all head-segments for this file in one go
        for _, head in heads_grp.iterrows():
            phot = sample_light_curve_coords(head, phot)

        phots_list[file_idx] = phot

    return phots_list

def add_match_labels(heads: pd.DataFrame, max_sep: float = 10) -> pd.DataFrame:
    """
    For each row in `heads`, find any other row within `max_sep` on the sky.
    Record the neighbor’s SNID in MATCH_SNID; then for all “unlensed” rows
    whose MATCH_SNID occurs >1×, combine their MULTICLASS_LABELs into
    MATCH_LABELS (with special recodings at the end).
    
    Parameters
    ----------
    heads : pandas.DataFrame
        Must contain columns ['RA','DEC','SNID','BINARY_LABEL','MULTICLASS_LABEL'].
    max_sep : astropy.units.Quantity (angle), optional
        Maximum sky separation to consider a “match” (default 10″).
    
    Returns
    -------
    out : pandas.DataFrame
        A copy of `heads` with two new columns:
        - MATCH_SNID : SNID of the closest neighbor (≤max_sep), or own SNID.
        - MATCH_LABELS : combined labels among matching groups.
    """
    df = heads.copy()
    
    if isinstance(max_sep, u.Quantity):
        if max_sep.unit != u.arcsec:
            max_sep = max_sep.to(u.arcsec)
    else:
        max_sep = max_sep * u.arcsec

    # 1) set up coords
    ras  = df['RA'].values  * u.deg
    decs = df['DEC'].values * u.deg
    coords = SkyCoord(ras, decs)
    
    # 2) find all pairs within max_sep
    idxc, idxcat, d2d, _ = coords.search_around_sky(coords, max_sep.to(u.deg))
    
    # 3) remove self-matches and duplicate orderings
    mask_nonself = idxc != idxcat
    i1 = idxc[mask_nonself]
    i2 = idxcat[mask_nonself]
    keep = i1 < i2
    i1u = i1[keep]
    i2u = i2[keep]
    
    # 4) build MATCH_SNID: default to own SNID, then overwrite with neighbor’s
    snids = df['SNID'].values.copy()
    snids[i1u] = df['SNID'].values[i2u]
    df['MATCH_SNID'] = snids
    
    # 5) focus on “unlensed”, find which MATCH_SNID occurs >1×
    is_unlensed = df['BINARY_LABEL'] == 'unlensed'
    counts = df.loc[is_unlensed, 'MATCH_SNID'].value_counts()
    multi = counts[counts > 1].index
    
    # 6) build MATCH_LABELS by default = own MULTICLASS_LABEL
    labels = df['MULTICLASS_LABEL'].values.copy()
    
    # 7) for each SNID with multiple hits, combine & sort their multiclass labels
    for sn in multi:
        mask = (df['MATCH_SNID'] == sn)
        combo = "_".join(sorted(df.loc[mask, 'MULTICLASS_LABEL'].values))
        labels[mask] = combo
    
    # 8) apply the two special recodings
    mask_slsn = np.char.find(labels.astype(str), '_SLSN') >= 0
    labels[mask_slsn] = 'sibling_SLSN'
    mask_ia   = np.char.find(labels.astype(str), 'Ia_Iax') >= 0
    labels[mask_ia]   = 'Ia_Ia'
    
    df['MATCH_LABELS'] = labels
    return df

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

def sphere2cart(ra, dec):
    x = np.cos(ra)*np.cos(dec)
    y = np.sin(ra)*np.cos(dec)
    z = np.sin(dec)
    p = np.array([x,y,z])

    return p

def nanweighted_average(x, errs, axis=None):

    w = 1/errs**2

    w_sum = np.nansum(w, axis=axis)
    mean = np.nansum(x * w, axis=axis) / w_sum
    mean_err = np.sqrt(1/w_sum)

    return mean, mean_err

def process_light_curve(
    phot: pd.DataFrame,
    delta_trigger_max: float = np.inf,
    delta_trigger_min: float = -30.,
    bands: list[str] = ['u','g', 'r', 'i', 'z', 'Y'],
    added_columns: list[str] = [
        'TRANS_SPECZ', 'TRANS_SPECZ_ERR',
        'TRANS_PHOTOZ', 'TRANS_PHOTOZ_ERR'
    ]
) -> pd.DataFrame:

    mjd = phot['MJD']
    idx_detection = phot['DETECTION'] == 1.
    
    idx_trigger = np.where(phot['DETECTION'])[0][0]
    t_trigger = mjd[idx_trigger]
    delta_trigger = mjd - t_trigger
    idx_excluded = (delta_trigger < delta_trigger_min) | (delta_trigger > delta_trigger_max)
    phot.loc[idx_excluded, ['FLUXCAL', 'FLUXCALERR']] = np.nan

    for band in bands:

        phot[f'{band}_FLUX'] = np.nan
        phot[f'{band}_FLUXERR'] = np.nan
        phot[f'{band}_DET'] = 0.
        phot[f'{band}_RA'] = np.nan
        phot[f'{band}_DEC'] = np.nan
        phot[f'{band}_POS_ERR'] = np.nan

        idx_band = phot['BAND'] == band
        idx_band_and_detection = idx_band & idx_detection

        phot.loc[idx_band, f'{band}_FLUX'] = phot.loc[idx_band, 'FLUXCAL']
        phot.loc[idx_band, f'{band}_FLUXERR'] = phot.loc[idx_band, 'FLUXCALERR']
        phot.loc[idx_band_and_detection, f'{band}_DET'] = 1.
        phot.loc[idx_band_and_detection, f'{band}_RA'] = phot.loc[idx_band, 'RA']
        phot.loc[idx_band_and_detection, f'{band}_DEC'] = phot.loc[idx_band, 'DEC']
        phot.loc[idx_band_and_detection, f'{band}_POS_ERR'] = phot.loc[idx_band, 'POS_ERR']
        phot[f'{band}_OBS'] = np.cumsum(~np.isnan(phot[f'{band}_FLUX']))
    
    
    idx_all_nan = np.all(np.isnan(phot[[f'{band}_FLUX' for band in bands]]), axis=1)
    phot.loc[idx_all_nan, added_columns] = np.nan

    ra_cols = phot.columns[phot.columns.str.contains('_RA')]
    dec_cols = phot.columns[phot.columns.str.contains('_DEC')]
    err_cols = phot.columns[phot.columns.str.contains('_POS_ERR')]

    ras = phot.loc[:, ra_cols].to_numpy()
    decs = phot.loc[:, dec_cols].to_numpy()
    errs = phot.loc[:, err_cols].to_numpy()
    errs = errs[:, :, None]
    errs = np.tile(errs, (1, 1, 3))

    phot = phot.drop(columns=ra_cols)
    phot = phot.drop(columns=dec_cols)
    phot = phot.drop(columns=err_cols)

    coords = sphere2cart(ras, decs)
    coords = np.swapaxes(coords, 1, 0)
    coords = np.swapaxes(coords, 1, 2)

    avg_coords, avg_err = nanweighted_average(coords, errs, axis=1)
    phot[['X', 'Y', 'Z']] = avg_coords
    phot['POS_ERR'] = avg_err[:, 0]

    return phot

def group_by_night(t):
    """
    Group Modified Julian Date times by astronomical night.
    
    A night is defined as the period from day.5 to (day+1).5, where day is an integer.
    For example, night 6091 spans from MJD 6090.5 to 6091.5.
    
    Parameters:
    -----------
    t : array-like
        Array of Modified Julian Date values
    
    Returns:
    --------
    dict
        Dictionary where keys are night numbers (integers) and values are lists
        of indices from the original array that belong to that night
    """
    t = np.array(t)
    nights = defaultdict(list)
    
    # For each time, determine which night it belongs to
    for i, time in enumerate(t):
        # Night number is floor(time + 0.5)
        # This ensures that times from day.5 to (day+1).5 map to night (day+1)
        night = int(np.floor(time + 0.5))
        nights[night].append(i)
    
    return dict(nights)

def stack_observations(
    img_phot, relative_error_floor=0.01,
    bands = ['u', 'g', 'r', 'i', 'z', 'Y'],
    added_columns: list[str] = [
        'TRANS_SPECZ', 'TRANS_SPECZ_ERR',
        'TRANS_PHOTOZ', 'TRANS_PHOTOZ_ERR'
    ]
):

    warnings.filterwarnings(
        'ignore',
        message='Mean of empty slice',
        category=RuntimeWarning
    )

    times = img_phot['MJD'].values
    nights_dict = group_by_night(times)

    nights = sorted(nights_dict.keys())
    n_nights = len(nights)

    columns = {
        'MJD': np.zeros(n_nights),
    }
    for band in bands:
        columns[f'{band}_FLUX'] = np.full(n_nights, np.nan)
        columns[f'{band}_FLUXERR'] = np.full(n_nights, np.nan)
        columns[f'{band}_DET'] = np.zeros(n_nights)
        columns[f'{band}_OBS'] = np.zeros(n_nights)
    for added_column in added_columns:
        columns[added_column] = np.full(n_nights, np.nan)

    flux_cols = [f'{band}_FLUX' for band in bands]
    flux_err_cols = [f'{band}_FLUXERR' for band in bands]
    det_cols = [f'{band}_DET' for band in bands]
    obs_cols = [f'{band}_OBS' for band in bands]

    flux = img_phot[flux_cols].values
    flux_err = img_phot[flux_err_cols].values
    det = img_phot[det_cols].values
    obs = img_phot[obs_cols].values
    weights = 1. / flux_err**2
    added_cols = img_phot[added_columns].values

    for i, night in enumerate(nights):

        idx = nights_dict[night]
        columns['MJD'][i] = float(night)

        for j, band in enumerate(bands):

            band_weights = weights[:, j]
            band_flux = flux[:, j]
            band_det = det[:, j]
            band_obs = obs[:, j]

            valid_obs = ~np.isnan(band_weights[idx])
            any_valid = np.any(valid_obs)

            avg_flux = np.nan
            avg_flux_err = np.nan
            night_det = 0
            night_obs = 0

            if any_valid:
                normed_weights = band_weights[idx] / np.nansum(band_weights[idx])
                avg_flux = np.nansum(band_flux[idx] * normed_weights)

                if np.sum(valid_obs) > 1:
                    avg_flux_err = np.sqrt(
                        1. / np.nansum(band_weights[idx]) +
                        (relative_error_floor*avg_flux)**2
                    )
                else:
                    avg_flux_err = np.sqrt(1. / np.nansum(band_weights[idx]))

                night_det = np.max(band_det[idx])
                night_obs = np.max(band_obs[idx])

            columns[f'{band}_FLUX'][i] = avg_flux
            columns[f'{band}_FLUXERR'][i] = avg_flux_err
            columns[f'{band}_DET'][i] = night_det
            columns[f'{band}_OBS'][i] = night_obs

        for j, added_column in enumerate(added_columns):
            columns[added_column][i] = np.nanmean(added_cols[idx, j])

    for band in bands:
        columns[f'{band}_OBS'] = np.cumsum(
            columns[f'{band}_OBS'] > 0
        )
    
    stacked_phot = pd.DataFrame(columns)

    warnings.resetwarnings()

    return stacked_phot

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
    ],
    stack_nightly=True,
):
    
    if stack_nightly:
        stack_fn = lambda phot: stack_observations(
            phot, relative_error_floor=0.01,
            added_columns=added_columns,
        )
    else:
        stack_fn = lambda phot: phot

    snid_phots = []
    for i, (_, img_head) in enumerate(snid_heads.iterrows()):
        
        img_phot = get_light_curve(img_head, phots, columns_to_add=added_columns)
        img_phot = process_light_curve(img_phot, added_columns=added_columns)
        img_phot = stack_fn(img_phot)

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


    return image_timeseries, trigger_index, transformed_flux, transformed_flux_errs

def serialize_lightcurves(
    heads,
    phots_list,
    out_path,
    dataset_name,
    max_images,
    flux_transform,
    flux_err_transform,
    stack_nightly = True,
    ):
    """
    Serialize a set of light-curve image time-series into an HDF5 file.

    Parameters
    ----------
    heads : pandas.DataFrame
        Must contain columns:
          ['MATCH_SNID', 'BINARY_LABEL', 'NUM_LCS', 'NUM_IMAGES',
           'PTROBS_MIN', 'PTROBS_MAX', 'MULTICLASS_LABEL']
        and any others used by prep.join_* and prep.transform_*.
        Should already be filtered to either train or val.
    phots_list : list of pandas.DataFrame
        The photometry tables you originally passed around.
    out_path : pathlib.Path or str
        Directory where `{dataset_name}.h5` will be created.
    dataset_name : str
        e.g. "train" or "val"; used to name the file.
    max_images : int
        How many image‐frames per SN to pad/trim to.
    flux_transform, flux_err_transform : callables
        Functions applied in `prep.transform_image_timeseries`.

    Returns
    -------
    None
    """

    out_path = Path(out_path)
    h5file_path = out_path / f"{dataset_name}.h5"
    
    # open (or overwrite) the file
    with h5py.File(h5file_path, "w") as h5file:
        
        # loop over each unique SNID in this subset
        for snid in tqdm(heads["MATCH_SNID"].unique(), desc=f"Writing {dataset_name}"):
            
            # select all rows matching this SNID
            sel = (heads["MATCH_SNID"] == snid)
            sub = heads.loc[sel]
            
            # pull per-SN metadata
            binary_label  = sub["BINARY_LABEL"].iat[0]
            num_lcs       = len(sub)
            num_imgs      = sub["NUM_IMAGES"].iat[0]
            joint_labels  = join_multiclass_labels(
                snid_heads=sub, max_images=max_images
            )
            
            # stitch & transform the photometry
            sn_phots = join_transient_images(
                snid_heads=sub,
                phots=phots_list,
                max_images=max_images,
                stack_nightly=stack_nightly,
            )
            (
                transformed,
                trigger_index,
                flux_arr,
                flux_err_arr
            ) = transform_image_timeseries(
                    image_timeseries=sn_phots,
                    flux_transform=flux_transform,
                    fluxerr_transform=flux_err_transform
                )
            trigger_time = transformed["MJD"].iat[trigger_index]
            
            # pick columns
            cols = ["MJD", "TRANS_MJD"] + [
                c for c in transformed.columns
                if any(pref in c for pref in ("FLUX_", "FLUXERR_", "DET_", "OBS_", "TRANS_Z"))
            ]
            tdf = transformed[cols]
            
            # infer variable groups by stripping trailing "_{i}"
            var_bases = []
            for c in cols[2:]:
                base = "_".join(c.split("_")[:-1])
                if base not in var_bases:
                    var_bases.append(base)
                    
            # split into flux / fluxerr / detobs / redshift
            flds = {
                "flux":      var_bases[:6],
                "flux_err":  var_bases[6:12],
                "detobs":    var_bases[12:24],
                "redshift":  var_bases[24:],
            }
            
            # stack into 3D arrays [image, epoch, filter]
            cubes = {}
            for key, names in flds.items():
                arrs = []
                for i in range(1, max_images+1):
                    cols_i = [f"{nm}_{i}" for nm in names]
                    arrs.append(tdf[cols_i].to_numpy())
                cubes[key] = np.stack(arrs, axis=0)
            
            # create HDF5 group and datasets
            grp = h5file.create_group(str(snid))
            # attributes
            grp.attrs.update({
                "MULTICLASS_LABEL": joint_labels.astype("S"),
                "BINARY_LABEL":     np.asarray(binary_label, dtype="S"),
                "NUM_LCS":          num_lcs,
                "NUM_IMAGES":       num_imgs,
                "TRIGGER_INDEX":    trigger_index,
                "TRIGGER_TIME":     trigger_time,
            })

            # time axes
            grp.create_dataset("MJD",       data=tdf["MJD"].values).attrs["column_names"] = np.array(["MJD"], dtype="S")
            grp.create_dataset("TRANS_MJD", data=tdf["TRANS_MJD"].values).attrs["column_names"] = np.array(["MJD-MJD[0] / 1000"], dtype="S")
            
            # data cubes
            for key in ("flux", "flux_err", "detobs", "redshift"):
                dset = grp.create_dataset(
                    key.upper() if key!="flux_err" else "FLUX_ERR",
                    data=cubes[key]
                )
                dset.attrs["column_names"] = np.array(flds[key], dtype="S")
    print(f"✅ Wrote {h5file_path}")