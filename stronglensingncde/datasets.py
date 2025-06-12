import jax
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from jax.dlpack import from_dlpack
from collections import defaultdict
from torch.utils.data import Sampler
from torch.utils.dlpack import to_dlpack
from torch.nn.utils.rnn import pad_sequence
from flax.jax_utils import prefetch_to_device
from torch.utils.data import Dataset, DataLoader, Subset, get_worker_info


def create_normalization_func(stats, key):

    mean = stats[key + '_MEAN']
    std = stats[key + '_STD']

    norm_func = lambda x: (x - mean) / std

    return norm_func

def create_redshift_transform(specz_norm, specz_err_norm, photoz_norm, photoz_err_norm):

    def redshift_transform(redshift, redshift_type):

        N_img, T, N_z = redshift.shape
        dtype, device = redshift.dtype, redshift.device
        flag_shape = (N_img, T, 1)

        if redshift_type == 'spec':

            redshift[..., 0] = specz_norm(redshift[..., 0])
            redshift[..., 1] = specz_err_norm(redshift[..., 1])
            flag = torch.zeros(flag_shape, dtype=dtype, device=device)
            redshift = torch.concat((redshift, flag), dim=-1)

        elif redshift_type == 'photo':

            redshift[..., 0] = photoz_norm(redshift[..., 0])
            redshift[..., 1] = photoz_err_norm(redshift[..., 1])    
            flag = torch.ones(flag_shape, dtype=dtype, device=device)
            redshift = torch.concat((redshift, flag), dim=-1)

        elif redshift_type == 'none':

            flag = torch.full(flag_shape, 2, dtype=dtype, device=device)
            redshift = torch.concat((redshift, flag), dim=-1)

        elif redshift_type == 'all':

            redshift[..., 0] = specz_norm(redshift[..., 0])
            redshift[..., 1] = specz_err_norm(redshift[..., 1])
            redshift[..., 2] = photoz_norm(redshift[..., 2])
            redshift[..., 3] = photoz_err_norm(redshift[..., 3])

        return redshift
    
    return redshift_transform

class HDF5TimeSeriesDataset(Dataset):
    def __init__(
        self,
        h5_file_path,
        flux_transform: callable = None,
        flux_err_transform: callable = None,
        redshift_transform: callable = None,
        sample_redshift: bool = False,
        sample_redshift_probs: np.ndarray = None,
        seed: int = 42,
        classes: list[str] = None,
        verbose: bool = True,
        min_n_obs: int = 3,
        **kwargs
    ):
        
        """
        Args:
            h5_file_path (str): Path to the HDF5 file.
            timeseries_transform (callable, optional): Optional transform to apply to sample timeseries

        Expected HDF5 structure:
            /sample_0/
                atttr.BINARY_LABEL: NumPy Array of dtype 'S' containing either
                                    the string 'lensed' or 'unlensed'.
                attr.MULTICLASS_LABEL: (N_max_images,) NumPy array containing
                                        class label for each of the N_max_images
                                        time series. Dtype is 'S'
                attr.TRIGGER_INDEX: Int denoting TRANS_MJD of event trigger
                TRANS_MJD: Dataset of shape (N_epochs_0, )
                FLUX: Dataset of shape (N_max_images, N_epochs_0, N_bands)
                FLUX_ERR: Dataset of shape (N_max_images, N_epochs_0, N_bands)
                DETOBS: Dataset of shape (N_max_images, N_epochs_0, 2*N_bands)
                REDSHIFT: Dataset of shape (N_max_images, N_epochs_0, 4)
            /sample_1/
                ...
            /sample_2/
                ...
        """
        
        self.h5_file_path = h5_file_path
        self.sample_keys = []
        self.labels = []
        self.max_length = 0
        self.n_max_images = 0
        self.flux_transform = flux_transform
        self.flux_err_transform = flux_err_transform
        self.redshift_transform = redshift_transform
        self.sample_redshift = sample_redshift
        if sample_redshift_probs:
            sample_redshift_probs = np.asarray(sample_redshift_probs)
        self.sample_redshift_probs = sample_redshift_probs
        self.rng = np.random.default_rng(seed=seed)
        
        label_subset = set(classes) if classes else set()
        class_labels = set()
        class_label_counter = Counter()
        with h5py.File(self.h5_file_path, 'r') as f:

            # 1) collect all keys…
            all_keys = sorted(f.keys())

            n_obs_filtered_count = 0
            class_filtering_count = 0
            self.sample_keys = []
            self.key_to_label = {}

            for sample_key in all_keys:

                n_obs = len(f[sample_key]['TRANS_MJD'])
                if n_obs < min_n_obs:
                    n_obs_filtered_count += 1
                    continue

                multiclass_labels = f[sample_key].attrs['MULTICLASS_LABEL']
                multiclass_labels = np.char.decode(multiclass_labels)
                multiclass_label_set = set(multiclass_labels)
                no_intersection = (len(label_subset & multiclass_label_set) < 1) or (len(multiclass_label_set) != 2)

                if no_intersection and classes:
                    class_filtering_count += 1
                    continue
                else:
                    self.sample_keys.append(sample_key)
                    class_labels.update(set(multiclass_labels))
                    class_label_counter.update(multiclass_labels.tolist())
                    self.key_to_label[sample_key] = multiclass_labels[0]

                    # Determine max length for padding
                    t_mjd = f[sample_key]['TRANS_MJD']
                    if len(t_mjd) > self.max_length:
                        self.max_length = len(t_mjd)
                    


            class_labels.remove('None')
            n_labels = len(class_labels)

            redshift_cols = f[sample_key]['REDSHIFT'].attrs['column_names'][()]
            self.redshift_cols = np.char.decode(redshift_cols)
            self.n_max_images = f[sample_key]['FLUX'][()].shape[0]
            self.label_to_idx = (
                {str(label): idx for idx, label in enumerate(sorted(class_labels))} | {'None' : n_labels}
            )
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

            self.class_counts = dict(class_label_counter)
            total_no_none = sum(
                count for label, count in self.class_counts.items() if label != 'None'
            )
            self.class_counts_dict = {
                label: count
                for label, count in self.class_counts.items()
                if label != 'None'
            }
            self.class_counts_array = np.array(
                [
                    self.class_counts_dict[
                        self.idx_to_label[i]
                    ] for i in range(n_labels)
                ]
            )
            self.class_frequencies_dict = {
                label: count / total_no_none
                for label, count in self.class_counts.items()
                if label != 'None'
            }
            self.class_frequencies_dict['None'] = 1.
            self.class_frequencies_array = np.array(
                [
                    self.class_frequencies_dict[
                        self.idx_to_label[i]
                    ] for i in range(n_labels)
                ]
            )

            self.labels = [
                self.label_to_idx[
                    self.key_to_label[key]
                ] for key in self.sample_keys
            ]

            unfiltered_size = len(all_keys)
            filtered_size = len(self.sample_keys)
            has_been_filtered = filtered_size < unfiltered_size
            if verbose:
                if has_been_filtered:
                    print(f"\nDropped {n_obs_filtered_count} samples with len(TRANS_MJD)<=1")
                    if classes:
                        print(f"Droppped {class_filtering_count} samples due to user-provided class filtering")
                    size_ratio = filtered_size / unfiltered_size
                    print(f"Current dataset length is {filtered_size}, or {size_ratio*100:.2f}% of original dataset.\n")
                
                print("\nClass Counts / Frequencies:")
                for label in self.class_counts_dict.keys():
                    label_count = self.class_counts_dict[label]
                    label_freq = self.class_frequencies_dict[label]
                    print(f"{label}: {label_count} / {label_freq*100:.2f}%")

                print(f"\nMaximum No. of Observations: {self.max_length}")
                print("\n")

    def __len__(self):
        return len(self.sample_keys)
    
    def resample_redshift(self, redshift):

        mode = self.rng.choice(
            ['spec', 'photo','none'],
            p=self.sample_redshift_probs
        )
        if mode == 'spec':
            output_redshift = redshift[..., :2]
        if mode == 'photo':
            output_redshift = redshift[..., 2:]
        if mode == 'none':
            output_redshift = np.full_like(redshift[..., :2], np.nan)
        
        return output_redshift, mode

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_file_path, 'r')

        sample_key = self.sample_keys[idx]
        sample_group = self.h5_file[sample_key]

        flux = torch.tensor(sample_group['FLUX'][()], dtype=torch.float64)
        flux_err = torch.tensor(sample_group['FLUX_ERR'][()], dtype=torch.float64)
        detobs = torch.tensor(sample_group['DETOBS'][()], dtype=torch.float64)
        t_mjd = torch.tensor(sample_group['TRANS_MJD'][()], dtype=torch.float64)
        length = len(t_mjd)
        max_time = t_mjd[-1]
        try:
            peak_time = sample_group.attrs['PEAK_TIME']
        except:
            peak_time = torch.full((flux.shape[0],), max_time)

        redshift = sample_group['REDSHIFT'][()]

        if self.sample_redshift:
            redshift, redshift_type = self.resample_redshift(redshift)
        else:
            redshift_type = 'all'
            
        redshift = torch.tensor(redshift, dtype=torch.float64)

        trigger_idx = sample_group.attrs['TRIGGER_INDEX']
        trigger_idx = torch.tensor(trigger_idx, dtype=torch.long)

        multiclass_labels = sample_group.attrs['MULTICLASS_LABEL'][()]
        multiclass_labels = np.char.decode(multiclass_labels)
        numeric_multiclass_labels = torch.tensor([self.label_to_idx[lbl] for lbl in multiclass_labels], dtype=torch.long)
        valid_lightcurve_mask = numeric_multiclass_labels != self.label_to_idx['None']

        binary_label = np.char.decode(sample_group.attrs['BINARY_LABEL'])
        numeric_binary_label = 1 if binary_label == 'lensed' else 0

        # Apply individual transforms if provided
        if self.flux_transform:
            flux = self.flux_transform(flux)
        if self.flux_err_transform:
            flux_err = self.flux_err_transform(flux_err)
        if self.redshift_transform:
            try:
                redshift = self.redshift_transform(redshift, redshift_type)
            except Exception as e:
                print(f"Exception for {redshift_type}:", e)


        partial_ts = torch.concat((flux_err, detobs, redshift), dim=-1)
        partial_ts = torch.permute(partial_ts, (1, 0, 2))
        flux = torch.permute(flux, (1,0,2))

        output = (
            t_mjd, flux, partial_ts,
            numeric_multiclass_labels,
            numeric_binary_label,
            trigger_idx, length,
            max_time, peak_time,
            valid_lightcurve_mask,
        )

        return output

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

def pad_last(ts_list, lengths, max_length):
    """
    Args:
        ts_list (list or tuple of tensors): each tensor has shape (T_i, *F)
        lengths  (LongTensor): shape (batch,), the original T_i for each element
        max_length (int): the length to pad each to along dim=0

    Returns:
        padded: Tensor of shape (batch, max_length, *F), where each sequence
                has been padded (after its real data) with its final value.
    """

    batch_size = len(ts_list)
    device = ts_list[0].device
    dtype = ts_list[0].dtype

    # make it mutable and grab the “feature” shape
    ts_list = list(ts_list)
    feature_shape = ts_list[0].shape[1:]    # () if 1D

    # append a dummy “all-zeros” tensor so pad_sequence will give us the right length
    dummy = torch.zeros((max_length, *feature_shape),
                        dtype=dtype,
                        device=device)
    ts_list.append(dummy)

    # now pad all sequences to length `max_length` along dim=0
    # result has shape (batch+1, max_length, *F)
    padded = pad_sequence(ts_list, batch_first=True)[:-1]  # discard the dummy

    # build a mask of “where we went past the true length”
    time_idx = torch.arange(max_length, device=device).unsqueeze(0).expand(batch_size, max_length)
    mask = time_idx >= lengths.unsqueeze(1)  # (batch, max_length)

    # gather the last real row for each sample
    idx = lengths - 1   # (batch,)
    last_row = padded[torch.arange(batch_size), idx]    # shape (batch, *F)

    # expand it to overwrite all padded positions
    expand_shape = (batch_size, max_length) + tuple(feature_shape)
    last_exp = last_row.unsqueeze(1).expand(*expand_shape)

    # flatten out batch & time, assign, then reshape back
    padded_flat = padded.reshape(-1, *feature_shape)    # (batch*max_length, *F)
    last_flat = last_exp.reshape(-1, *feature_shape)    # same
    mask_flat = mask.reshape(-1)    # (batch*max_length,)

    padded_flat[mask_flat] = last_flat[mask_flat]
    padded = padded_flat.reshape(batch_size, max_length, *feature_shape)

    return padded

def collate_fn(batch, max_length=None, nmax=None):
    
    (
        t_list, flux_list, partial_ts_list, numeric_multiclass_labels_list,
        numeric_binary_labels_list, trigger_idx_list, lengths_list, max_time,
        peak_times_list, valid_lightcurve_mask_list
    ) = zip(*batch)

    
    numeric_binary_labels = torch.tensor(numeric_binary_labels_list, dtype=torch.int)
    numeric_multiclass_labels = torch.stack(numeric_multiclass_labels_list)
    valid_lightcurve_mask = torch.stack(valid_lightcurve_mask_list)
    trigger_idx = torch.tensor(trigger_idx_list, dtype=torch.long)
    lengths = torch.tensor(lengths_list, dtype=torch.long)
    max_times = torch.tensor(max_time, dtype=torch.float64)
    peak_times = torch.stack(peak_times_list)

    # Get the lengths and determine max length.
    if max_length is None:
        max_length = lengths.max().item()

    padded_t = pad_last(t_list, lengths=lengths, max_length=max_length)

    padded_flux = pad_last(flux_list, lengths=lengths, max_length=max_length)
    padded_flux = torch.permute(padded_flux, (0, 2, 1, 3))
    
    padded_partial_ts = pad_last(partial_ts_list, lengths=lengths, max_length=max_length)
    padded_partial_ts = torch.permute(padded_partial_ts, (0, 2, 1, 3))

    if nmax:
        padded_t = padded_t[:, :nmax]
        padded_flux = padded_flux[:, :, :nmax, :]
        padded_partial_ts = padded_partial_ts[:, :, :nmax, :]
        max_times = torch.take_along_dim(
            padded_t, torch.minimum(lengths, torch.tensor(nmax))[:, None]-1, dim=1
        ).squeeze()
        trigger_idx = torch.where(
            trigger_idx >= nmax,
            torch.tensor(0, dtype=torch.long),
            trigger_idx
        )
        lengths = torch.minimum(lengths, torch.tensor(nmax))

    output = (
        padded_t, padded_flux, padded_partial_ts,
        trigger_idx, lengths, peak_times, max_times,
        numeric_binary_labels, numeric_multiclass_labels,
        valid_lightcurve_mask
    )

    return output

def _worker_init_fn(worker_id):
    """
    Ensure each worker process opens its own HDF5 file handle,
    rather than inheriting one from the parent.
    """
    worker_dataset = get_worker_info().dataset
    if hasattr(worker_dataset, 'h5_file'):
        # delete any inherited file handle so __getitem__ reopens
        del worker_dataset.h5_file

def create_subset(dataset, max_size=None, seed=42):

    labels = dataset.labels
    class_to_idxs = defaultdict(list)

    for idx, lbl in enumerate(labels):
        class_to_idxs[lbl].append(idx)

    min_count = min(len(lst) for lst in class_to_idxs.values())
    if max_size:
        min_count = min((min_count, max_size))

    rng = np.random.default_rng(seed=seed)
    balanced_indices = []
    for lbl, idx_list in class_to_idxs.items():
        chosen = rng.choice(idx_list, size=min_count, replace=False).tolist()
        balanced_indices.extend(chosen)
    
    subset = Subset(dataset, balanced_indices)
    
    return subset

def torch_to_jax(batch):

    jax_batch = []
    for item in batch:
        cuda_item = item.to(non_blocking=True)
        dl = to_dlpack(cuda_item)
        jax_arr = from_dlpack(dl)
        jax_batch.append(jax_arr)

    return jax_batch

def make_jax_prefetched_loader(torch_loader, prefetch_size=2):
    """
    Wraps the PyTorch loader and DLPack conversion into a prefetching JAX iterator.
    """
    def gen():
        for torch_batch in torch_loader:
            yield torch_to_jax(torch_batch)

    return prefetch_to_device(gen(), size=prefetch_size)

def make_dataloader(
    h5_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_length: int = None,
    max_obs: int = None,
    subsample: bool = False,
    subsample_max_size: int = None,
    seed: int = 42,
    prefetch_factor: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """
    Factory to create a DataLoader for your HDF5TimeSeriesDataset.

    Args:
        h5_path:       path to your .h5
        batch_size:    samples per batch
        shuffle:       whether to shuffle each epoch
        num_workers:   subprocesses to use for data loading
        pin_memory:    pin CUDA memory
        max_length:    if None, inferred from dataset.max_length
        max_obs:       maximum number of observations in time series
        **dataset_kwargs: passed on to HDF5TimeSeriesDataset constructor
    """
    ds = HDF5TimeSeriesDataset(h5_path, **dataset_kwargs)

    if max_length is None:
        max_length = ds.max_length

    if subsample:
        ds = create_subset(
            ds, max_size=subsample_max_size, seed=seed
        )

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_fn(batch, max_length=max_length, nmax=max_obs),
        worker_init_fn=_worker_init_fn,
        prefetch_factor=prefetch_factor
    )
    dataloader_len = len(dataloader)

    dataloader = make_jax_prefetched_loader(dataloader, prefetch_size=prefetch_factor)

    return dataloader, ds, dataloader_len