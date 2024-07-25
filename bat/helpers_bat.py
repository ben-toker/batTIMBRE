from scipy import *
from scipy import interpolate
from scipy import stats
import numpy as np
from random import sample
from bat.dataset import *
from rat.helpers import balanced_indices

'''
A set of helper functions for preprocessing the bat data.
@author: Kevin Kaiwen Qi
@author: Ben Toker
'''

def find_multi_label_bins(spk_timebins, labels, label_timestamps_sec):
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
    #print(unique_combinations, counts)

    return bins_with_multi_unique_labels, unique_combinations, counts

def label_timebins(spk_timebins, labels, label_timestamps_sec, is_discrete):
    """
    result = label_timebins([0,2,4,6,8], np.array([2,2,4,5,6,6,6,6,1,2,2,4,3,3,3]), np.array([0.5,1,1.5,1.6,1.5,1,1,1,3.5,6.5,6.6,6.5,6.4,6.5,6.6]), is_discrete=True)
    print(result)
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
        #print(multi_label_bins)
       
        # Correct labels for bins with multiple different labels
        if len(multi_label_bins) > 0:
            for bin_index in multi_label_bins:
                bin_label_counts = np.argmax(counts[unique_combinations[:, 0] == bin_index])
                #print(unique_combinations[unique_combinations[:, 0] == bin_index,:])
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
    timebin_edges = np.array(timebin_edges)
    label_timestamps = np.array(label_timestamps)
   
    # Create an array of booleans, one for each timebin
    result = np.zeros(len(timebin_edges) - 1, dtype=bool)
   
    # Use numpy's digitize to find which bin each label falls into
    bin_indices = np.digitize(label_timestamps, timebin_edges) - 1
   
    # Filter out any indices that are out of bounds
    valid_indices = (bin_indices >= 0) & (bin_indices < len(result))
    bin_indices = bin_indices[valid_indices]
   
    # Set the corresponding bins to True
    result[bin_indices] = True
   
    return result


def interpolate_nans(array, max_nan_span=10000000):
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

def get_flightID(session, binned_pos, binned_indices):
    """
    Organizes information about bat behavior into a matrix (flightID).

    Input:
        session: session object containing flight data
        binned_pos: binned positional data array (3 x N) for X, Y, Z positions
        binned_indices: Array of indices mapping original to binned timestamps

    Output:
        flightID = Data stored in the format listed below

    flightID format:
        Column 0: Flight Number
        Column 1: Feeder visited (0: perch, 1: feeder 1, 2: feeder 2)
        Column 2: X position in M
        Column 3: Y position in M
        Column 4: Z position in M
    """
    # Initialize variables
    flight_info = []
    flight_number = 1

    # Iterate through each flight
    for flight_number in range(session.num_flights):
        start_idx = session.flight_start_idx[flight_number]
        end_idx = session.flight_end_idx[flight_number]
        print(f"Processing flight {flight_number + 1} from {start_idx} to {end_idx}")

        # Convert original indices to binned indices
        binned_start_idx = binned_indices[start_idx]
        binned_end_idx = binned_indices[end_idx]

        # Ensure indices are within bounds
        if binned_end_idx >= binned_pos.shape[1]:
            print(f"Skipping flight with end_idx {binned_end_idx} as it exceeds bounds.")
            continue

        # Determine which feeder the bat visited based on the x and y coordinates of the last position
        end_x = binned_pos[0, binned_end_idx]
        end_y = binned_pos[1, binned_end_idx]
        
        if end_y > 0 and end_x < 0:
            feeder_visited = 0  # Perch
        elif end_y > 0 and end_x > 0:
            feeder_visited = 1  # Feeder 1
        elif end_y < 0 and end_x > 0:
            feeder_visited = 2  # Feeder 2
        else:
            feeder_visited = -1  # In case it doesn't match any criteria (should not happen)
        
        # Store the flight information for each sample in the binned_pos array
        for idx in range(binned_start_idx, binned_end_idx + 1):
            if idx < binned_pos.shape[1]:
                flight_info.append([flight_number + 1, feeder_visited, binned_pos[0, idx], binned_pos[1, idx], binned_pos[2, idx]])
            else:
                print(f"Index {idx} out of bounds for binned_pos with shape {binned_pos.shape}")
    
    # Convert to numpy array
    flightID = np.array(flight_info)
    
    return flightID

def test_train_bat(flightID, n_folds=5, which_fold=0, num_samples_at_end=5):
    """
    Returns test and train samples for bat flight data.

    Parameters:
    - flightID: contains info about flight number, feeder visited, and positional data
    - n_folds: how many folds to assign
    - which_fold: which fold to return values for
    - num_samples_at_end: number of samples at the end of each flight to use for classification

    Returns:
    - train_inds: which samples to use for training model
    - test_inds: which samples to use for testing model
    """
    # Initialize variables
    ctr = np.zeros(3)  # Assuming three classes: perch, feeder 1, feeder 2
    fold_assign = -np.ones(flightID.shape[0])
    
    for i in range(int(np.max(flightID[:, 0])) + 1):
        inds = flightID[:, 0] == i
        if np.sum(inds):
            # Use the last few samples of the current flight
            flight_indices = np.where(inds)[0]
            last_indices = flight_indices[-num_samples_at_end:]
            feeder = int(flightID[last_indices[0], 1])
            fold_assign[last_indices] = ctr[feeder] % n_folds
            ctr[feeder] += 1
    
    test_inds = fold_assign == which_fold
    train_inds = np.isin(fold_assign, np.arange(n_folds)) & ~test_inds
    print(f"Initial train_inds (before balancing): {np.sum(train_inds)}")
    
    train_inds_balanced = helpers.balanced_indices(flightID[:, 1], train_inds)
    print(f"Balanced train_inds: {len(train_inds_balanced)}")
    
    # Debug: Print the shapes to verify alignment
    print(f"flightID length: {flightID.shape[0]}")
    print(f"test_inds length: {np.sum(test_inds)}")
    print(f"train_inds length: {np.sum(train_inds)}")
    print(f"Balanced train_inds length: {len(train_inds_balanced)}")
    
    # Ensure `train_inds_balanced` are valid indices
    if np.any(train_inds_balanced >= flightID.shape[0]) or np.any(test_inds >= flightID.shape[0]):
        print("Error: Indices out of bounds")
    else:
        print("All indices are within bounds")
    
    return test_inds, train_inds_balanced
