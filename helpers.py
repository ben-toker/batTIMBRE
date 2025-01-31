from scipy import *
from scipy import interpolate
from scipy import stats
from keras import models
from scipy import signal
import numpy as np
from random import sample
from numba import jit
from dataset import *

'''
A set of helper functions for preprocessing the bat data.
@author: Kevin Kaiwen Qi
@author: Ben Toker
'''

def mask_by_cluster(session, masked_BIGRAW, cluster_id, buffer=10_000):
    """
    Masks rows of masked_BIGRAW for a specific cluster, preserving LFP integrity and flight info.

    Args:
        session (FlightRoomSession): The session object with flight data.
        masked_BIGRAW (np.ndarray): Array with timestamps, LFP data, and positions.
        cluster_id (int): The cluster ID to filter by.
        buffer (int): Time buffer (in microseconds) around flight times.

    Returns:
        np.ndarray: Filtered BIGRAW with an additional column for flight numbers.
    """
    # Extract timestamps
    timestamps = masked_BIGRAW[:, 0]  # First column is timestamps
    flights = session.get_flights_by_cluster([cluster_id])  # Retrieve flights for the cluster

    print(f"Retrieved {len(flights)} flights for cluster {cluster_id}.")

    # Initialize mask for selecting rows
    row_mask = np.zeros(len(timestamps), dtype=bool)  # False mask for all rows
    flight_info = np.zeros(len(timestamps), dtype=int)  # Column to store flight numbers

    for flight_idx, flight in enumerate(flights, start=1):
        # Calculate start and end times with buffer
        start_time = flight.start_time * 1e6 - buffer  # Convert to microseconds
        end_time = flight.end_time * 1e6 + buffer  # Convert to microseconds
        
        # Create a mask for this flight
        flight_mask = (timestamps >= start_time) & (timestamps <= end_time)
        
        # Combine with the overall mask
        row_mask |= flight_mask
        flight_info[flight_mask] = flight_idx  # Assign flight number to these rows

    # Apply the mask
    filtered_BIGRAW = masked_BIGRAW[row_mask]
    flight_info_filtered = flight_info[row_mask]

    # Combine filtered data with flight numbers
    result = np.column_stack((filtered_BIGRAW, flight_info_filtered))

    # Debug: Print retained timestamps
    retained_timestamps = result[:, 0]
    print(f"Filtered BIGRAW timestamp range: {retained_timestamps[0]} to {retained_timestamps[-1]}")

    return result


def find_multi_label_bins(spk_timebins, labels, label_timestamps_sec):
    """
    Finds the bins with multiple different labels and returns the unique combinations and counts.

    Parameters:
        spk_timebins (array-like): The time bins.
        labels (array-like): The labels corresponding to label_timestamps_sec.
        label_timestamps_sec (array-like): The timestamps of the labels.

    Returns:
        tuple: A tuple containing:
            - bins_with_multi_unique_labels (array-like): The indices of bins with multiple different labels.
            - unique_combinations (ndarray): The unique combinations of bin-label pairs.
            - counts (ndarray): The counts of each unique combination.
    """
    bin_indices = np.digitize(label_timestamps_sec, spk_timebins) - 1
    valid_mask = (bin_indices >= 0) & (bin_indices < len(spk_timebins) - 1)
    bin_indices = bin_indices[valid_mask]
    valid_labels = labels[valid_mask]
   
    # Find unique bin-label combinations
    unique_combinations, counts = np.unique(np.column_stack((bin_indices, valid_labels)), axis=0, return_counts=True)
   
    # Count the number of unique labels for each bin
    bin_label_counts = np.zeros(len(spk_timebins) - 1, dtype=int)
    np.add.at(bin_label_counts, unique_combinations[:, 0], 1)
   
    # Find bins with multiple different labels
    bins_with_multi_unique_labels = np.where(bin_label_counts > 1)[0]

    return bins_with_multi_unique_labels, unique_combinations, counts

def label_timebins(spk_timebins, labels, label_timestamps_sec, is_discrete):
    """
    Resamples labels to match the time bins defined by spk_timebins.
    
    Parameters:
        spk_timebins (array-like): The time bins.
        labels (array-like): The labels corresponding to label_timestamps_sec.
        label_timestamps_sec (array-like): The timestamps of the labels.
        is_discrete (bool): Indicates whether the labels are discrete or continuous.
        
    Returns:
        array-like: The resampled labels.
    """
    
    # Ensure inputs are numpy arrays
    spk_timebins = np.array(spk_timebins)
    labels = np.array(labels)
    label_timestamps_sec = np.array(label_timestamps_sec)
   
    # Calculate the midpoints of spk_timebins
    spk_midpoints = (spk_timebins[:-1] + spk_timebins[1:]) / 2
   
    if is_discrete:
        # For discrete labels, use nearest neighbor interpolation
        f = interpolate.interp1d(label_timestamps_sec, labels, kind='nearest',
                                 bounds_error=False, fill_value=0)
        resampled_labels = f(spk_midpoints)
       
        # Find bins with multiple different labels
        multi_label_bins, unique_combinations, counts = find_multi_label_bins(spk_timebins, labels, label_timestamps_sec)
       
        # Correct labels for bins with multiple different labels
        if len(multi_label_bins) > 0:
            for bin_index in multi_label_bins:
                bin_label_counts = np.argmax(counts[unique_combinations[:, 0] == bin_index])
                resampled_labels[bin_index] = unique_combinations[unique_combinations[:, 0] == bin_index, 1][bin_label_counts]
       
        # Set labels to 0 for bins without any labels
        valid_bins = np.digitize(label_timestamps_sec, spk_timebins) - 1
        valid_bins = valid_bins[(valid_bins >= 0) & (valid_bins < len(spk_timebins) - 1)]
        invalid_bins = np.setdiff1d(np.arange(len(spk_timebins) - 1), valid_bins)
        resampled_labels[invalid_bins] = 0
       
    else:
        # For continuous labels, use linear interpolation
        f = interpolate.interp1d(label_timestamps_sec, labels, kind='linear',
                                 bounds_error=False, fill_value=np.nan)
        resampled_labels = f(spk_midpoints)
       
        # Set labels to NaN for bins without any labels
        valid_bins = np.digitize(label_timestamps_sec, spk_timebins) - 1
        valid_bins = valid_bins[(valid_bins >= 0) & (valid_bins < len(spk_timebins) - 1)]
        invalid_bins = np.setdiff1d(np.arange(len(spk_timebins) - 1), valid_bins)
        resampled_labels[invalid_bins] = np.nan
   
    return resampled_labels

def label_in_timebin(timebin_edges, label_timestamps):
    """
    Determines which time bins each label falls into.

    Parameters:
    - timebin_edges (array-like): An array of time bin edges.
    - label_timestamps (array-like): An array of label timestamps.

    Returns:
    - result (ndarray): An array of booleans indicating which time bins each label falls into.
    """

    # Convert input arrays to numpy arrays
    timebin_edges = np.array(timebin_edges)
    label_timestamps = np.array(label_timestamps)
   
    # Create an array of booleans, one for each time bin
    result = np.zeros(len(timebin_edges) - 1, dtype=bool)
   
    # Use numpy's digitize to find which bin each label falls into
    bin_indices = np.digitize(label_timestamps, timebin_edges) - 1
   
    # Filter out any indices that are out of bounds
    valid_indices = (bin_indices >= 0) & (bin_indices < len(result))
    bin_indices = bin_indices[valid_indices]
   
    # Set the corresponding bins to True
    result[bin_indices] = True
   
    return result


import numpy as np

def interpolate_nans(array, max_nan_span=10000000):
    """
    Interpolates NaN values in a given array.

    Args:
        array (ndarray): The input array.
        max_nan_span (int, optional): The maximum span of consecutive NaN values to interpolate. Defaults to 10000000.

    Returns:
        ndarray: The array with NaN values interpolated.

    """
    n = len(array)
    is_nan = np.isnan(array)
    indices = np.arange(n)
    interpolated_array = np.copy(array)
    
    start = None
    for i in range(n):
        if is_nan[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                span_length = i - start
                if span_length <= max_nan_span:
                    if start == 0:
                        start_value = array[i]
                    else:
                        start_value = array[start - 1]
                    
                    end_value = array[i]
                    
                    interpolated_array[start:i] = np.interp(
                        np.arange(start, i),
                        [start - 1, i],
                        [start_value, end_value]
                    )
                start = None
    
    # Handle the case where the array ends with NaNs
    if start is not None and (n - start) <= max_nan_span:
        start_value = array[start - 1]
        end_value = array[start - 1]  # The array ends with NaNs
        interpolated_array[start:n] = np.interp(
            np.arange(start, n),
            [start - 1, n],
            [start_value, end_value]
        )
    
    return interpolated_array




def safe_accumarray(subs, vals, size=None, func=np.mean):
    if size is None:
        size = np.max(subs) + 1
    
    if vals.ndim == 1:
        result = np.full(size, np.nan)
    else:
        result = np.full((size, vals.shape[1]), np.nan)
    
    for i, sub in enumerate(subs):
        if 0 <= sub < size:
            if np.isnan(result[sub]).all():
                result[sub] = vals[i]
            else:
                if vals.ndim == 1:
                    result[sub] = func([result[sub], vals[i]])
                else:
                    result[sub] = func([result[sub], vals[i]], axis=0)
    return result

def balanced_indices(labels, indices, random_state=None):
    """
    Returns a balanced subset of indices with equal number of samples per class.

    Parameters:
    - labels: Array of labels corresponding to the indices.
    - indices: Array of indices to select from.
    - random_state: Seed for random number generator.

    Returns:
    - balanced_indices: Array of indices with balanced samples across classes.
    """
    from collections import defaultdict
    import numpy as np

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)

    # Group indices by label
    label_to_indices = defaultdict(list)
    for idx in indices:
        label = labels[np.where(indices == idx)[0][0]]
        label_to_indices[label].append(idx)

    # Find the minimum number of samples across all classes
    min_samples = min(len(idxs) for idxs in label_to_indices.values())

    # Collect balanced indices
    balanced_indices = []
    for idxs in label_to_indices.values():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        balanced_indices.extend(idxs[:min_samples])

    return np.array(balanced_indices)

from sklearn.model_selection import StratifiedKFold

def test_train_bat(flightID, n_folds=5, which_fold=0, num_samples_at_end=5):
    """
    Returns test and train samples for bat flight data using stratified k-fold cross-validation.

    Parameters:
    - flightID: numpy array with columns:
        Column 0: Flight Number
        Column 1: Flight Type (0 to 6)
        Column 2-4: Positional data (X, Y, Z)
    - n_folds: Number of folds for cross-validation
    - which_fold: Index of the fold to use as the test set
    - num_samples_at_end: Number of samples at the end of each flight to use for classification

    Returns:
    - test_inds: Indices of samples to use for testing
    - train_inds: Indices of samples to use for training
    """
    # Map flight numbers to flight types
    flight_numbers = np.unique(flightID[:, 0]).astype(int)
    flight_types = []
    for flight_num in flight_numbers:
        inds = flightID[:, 0] == flight_num
        flight_type = int(flightID[inds][0, 1])
        flight_types.append(flight_type)

    flight_types = np.array(flight_types)

    # Use Stratified K-Fold to split flights into folds, ensuring balance across flight types
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(flight_numbers, flight_types))

    # Get train and test flight numbers for the specified fold
    train_flight_nums = flight_numbers[splits[which_fold][0]]
    test_flight_nums = flight_numbers[splits[which_fold][1]]

    # Collect sample indices for train and test sets
    train_inds = []
    for flight_num in train_flight_nums:
        inds = np.where(flightID[:, 0] == flight_num)[0]
        last_indices = inds[-num_samples_at_end:]
        train_inds.extend(last_indices)

    test_inds = []
    for flight_num in test_flight_nums:
        inds = np.where(flightID[:, 0] == flight_num)[0]
        last_indices = inds[-num_samples_at_end:]
        test_inds.extend(last_indices)

    train_inds = np.array(train_inds)
    test_inds = np.array(test_inds)

    # Balance the training indices across flight types
    train_labels = flightID[train_inds, 1]
    train_inds_balanced = balanced_indices(train_labels, train_inds)

    # Debug statements to verify the process
    print(f"Total samples: {flightID.shape[0]}")
    print(f"Test samples: {len(test_inds)}")
    print(f"Train samples before balancing: {len(train_inds)}")
    print(f"Train samples after balancing: {len(train_inds_balanced)}")

    # Ensure indices are within bounds
    if np.any(train_inds_balanced >= flightID.shape[0]) or np.any(test_inds >= flightID.shape[0]):
        print("Error: Indices out of bounds")
    else:
        print("All indices are within bounds")

    return test_inds, train_inds_balanced

#original TIMBRE helpers

def balanced_indices(vector, bool_indices):
    """
    Returns indices that balance the number of samples for each label in vector

    Parameters:
    vector: The input vector from which to select indices.
    bool_indices: A boolean array indicating which indices in the vector to consider.

    Returns:
    list: A list of indices representing a balanced selection of the unique values in the subset of the vector.
    """
    # Convert boolean indices to actual indices
    actual_indices = np.where(bool_indices)[0]
    print(f"actual_indices length: {len(actual_indices)}")
    
    # Extract the elements and their corresponding indices
    selected_elements = [(vector[i], i) for i in actual_indices]

    # Find unique elements
    unique_elements = np.unique(vector[bool_indices])
    print(f"unique_elements: {unique_elements}")

    # Group elements by value and collect their indices
    elements_indices = {element: [] for element in unique_elements}
    for value, idx in selected_elements:
        if value in elements_indices:
            elements_indices[value].append(idx)

    # Debug: Print groupings
    print(f"elements_indices: {elements_indices}")

    # Find the minimum count among the unique elements
    min_count = min(len(elements_indices[element]) for element in unique_elements)
    print(f"min_count: {min_count}")

    # Create a balanced set of indices
    balanced_indices_set = []
    for element in unique_elements:
        if len(elements_indices[element]) >= min_count:
            balanced_indices_set.extend(sample(elements_indices[element], min_count))

    balanced_indices_array = np.array(balanced_indices_set)

    # Debug: Print the final balanced indices
    print(f"balanced_indices_array length: {len(balanced_indices_array)}")

    return balanced_indices_array

def layer_output(X, m, layer_num):
    """
    Returns response of one of TIMBRE's layers

    Parameters:
    - X: Input data
    - m: Trained model
    - layer_num: Which layer's output to return

    Returns:
    - Layer's response to input
    """
    # stack the real and imaginary components of the data
    X = np.concatenate((np.real(X), np.imag(X)), axis=1)
    m1 = models.Model(inputs=m.inputs, outputs=m.layers[layer_num].output)
    return m1.predict(X)  # return output of layer layer_num

def accumarray(subs, vals, size=None, fill_value=0):
    """
    Averages all values that are associated with the same index. Does this separately for each column of vals.
    Useful for visualizing dependency of layer outputs on behavioral features. 

    Parameters:
    - subs: An MxN array of subscripts, where M is the number of entries in vals and N is the number of dimensions of the output.
    - vals: An MxK matrix of values.
    - size: Tuple specifying the size of the output array (default is based on the maximum index in each column of subs)
    - fill_value: The value to fill in cells of the output that have no entries (default is 0).

    Returns:
    - result: An array of accumulated values.
    
    Generated using ChatGPT
    """
    subs = subs.astype(int)
    if subs.ndim == 1:
        subs = subs[:, np.newaxis]
    if size is None:
        size = tuple(np.max(subs, axis=0) + 1)
    else:
        assert len(size) == subs.shape[1], "Size mismatch between size and subs."

    # Handle single column vals
    if len(vals.shape) == 1:
        vals = vals[:, np.newaxis]

    # Convert subscripts to linear indices.
    indices = np.ravel_multi_index(tuple(subs.T), size)

    K = vals.shape[1]
    result = np.full((*size, K), fill_value, dtype=float)

    for k in range(K):
        total = np.bincount(indices, weights=vals[:, k], minlength=np.prod(size))
        count = np.bincount(indices, minlength=np.prod(size))
        with np.errstate(divide='ignore', invalid='ignore'):  # Ignore divide by zero and invalid operations
            averaged = np.where(count != 0, total / count, fill_value)
        result[..., k] = averaged.reshape(size)

    return result if K > 1 else result.squeeze(-1)


def filter_data(data, cutoff, fs, filt_type='high', order=5, use_hilbert=False):
    """
    Applies a column-wise zero-phase filter to data
    
    Parameters:
    data : a T x N array with filtered data
    cutoff : cutoff frequency (should be 2 numbers for 'band')
    fs : sampling rate
    filt_type : specify as 'high', 'low', or 'band'.
    order : filter order. The default is 5.
    use_hilbert: whether to apply a Hilbert transform (default = False)

    Returns
    -------
    data : a T x N array with filtered data
    """
    nyq = 0.5 * fs
    if filt_type == 'band':
        if len(cutoff) != 2:
            raise ValueError("Cutoff should contain exactly two numbers for 'band' filter type.")
        normal_cutoff = [c / nyq for c in cutoff]
    else:
        normal_cutoff = cutoff / nyq
    
    b, a = signal.butter(order, normal_cutoff, btype=filt_type, analog=False)
    data = signal.filtfilt(b, a, data, axis=0)
    if use_hilbert:
        data = signal.hilbert(data, axis=0)
    return data

def whiten(X, inds_train, fudge_factor=10 ** -5):
    """
    Decorrelates the input data

    Parameters:
    - X: A TxN array of data, can be complex-valued
    - inds_train: which samples to use to estimate correlations
    - fudge_factor: adds a small constant to lessen the influence of small, noisy directions in the data

    Returns:
    - X: decorrelated data
    - u: directions of highest variance in original data
    - Xv: scaling factor used to normalize decorrelated data
    """
    _, _, u = np.linalg.svd(X[inds_train, :], full_matrices=False, compute_uv=True)
    X = X @ np.conj(u.T)
    Xv = np.var(X[inds_train, :], axis=0)
    Xv = np.sqrt(Xv + sum(Xv) * fudge_factor)
    X = X / Xv
    return X, u, Xv

