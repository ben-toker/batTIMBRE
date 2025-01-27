import math
import os
import numpy as np
from numba import jit
import pickle
import hashlib
from scipy.interpolate import interp1d
from scipy.signal import decimate
from multiprocessing import Pool
from functools import partial
from helpers import label_timebins
from helpers import interpolate_nans
from scipy.stats import mode
from dataset import FlightRoomSession
import hdf5storage

"""
A set of helper functions for downloading and preprocessing flight path data.

@author: Ben Toker
@author: Kevin Qi 
"""
def load_and_align_lfp_and_pos(data_path, bat_id, date, lfp_file_path, use_cache=True, cache_dir='./lfp_pos_cache'):
    """
    Loads LFP data from two probes, cleans and interpolates positional data, and aligns them into a combined structure.

    Args:
        data_path (str): Path to the directory containing the bat data.
        bat_id (str): ID of the bat.
        date (str): Date of the recording.
        lfp_file_path (str): Path to the LFP data file.
        use_cache (bool): Whether to use cached data.
        cache_dir (str): Directory for caching aligned data.

    Returns:
        np.ndarray: A single array with columns for timestamps, LFP data, and position (x, y, z).
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a unique cache key for this dataset
    cache_key = hashlib.md5(f"{bat_id}_{date}_{lfp_file_path}".encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}_aligned_data.npy")

    # Load from cache if available
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached aligned data from {cache_file}")
        return np.load(cache_file, allow_pickle=True)

    # Load LFP data from .mat file
    lfp_mat = hdf5storage.loadmat(lfp_file_path)
    lfp_data_1 = lfp_mat['lfp'][0, 0]
    lfp_data_2 = lfp_mat['lfp'][0, 1]
    n_channels_1 = lfp_data_1.shape[0]
    n_channels_2 = lfp_data_2.shape[0]
    sampling_rate = 2500  # Initial sampling rate of LFP data

    print(f"LFP data shapes: {lfp_data_1.shape}, {lfp_data_2.shape}, n_channels: {n_channels_1 + n_channels_2}")

    # Combine LFP data from both probes along the channel axis
    lfp_data = np.concatenate((lfp_data_1, lfp_data_2), axis=0)
    lfp_timestamps = np.arange(lfp_data.shape[1]) / sampling_rate * 1e6  # Generate timestamps in microseconds
    print(f"Combined LFP data shape: {lfp_data.shape}, LFP timestamps shape: {lfp_timestamps.shape}")

    # Load positional data
    session = FlightRoomSession(data_path, bat_id, date, use_cache=use_cache)
    pos_data = session.cortex_data.bat_pos
    pos_timestamps = session.cortex_data.cortex_global_sample_timestamps_sec * 1e6
    print(f"Position data shape: {pos_data.shape}, Timestamps shape: {pos_timestamps.shape}")

    # Clean and interpolate positional data
    cleaned_pos = np.copy(pos_data)
    for axis in range(3):  # Interpolate x, y, z
        cleaned_pos[:, axis] = interpolate_nans(pos_data[:, axis])

    # Interpolate positional data to align with LFP timestamps
    interp_func_x = interp1d(pos_timestamps, cleaned_pos[:, 0], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_func_y = interp1d(pos_timestamps, cleaned_pos[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_func_z = interp1d(pos_timestamps, cleaned_pos[:, 2], kind='linear', bounds_error=False, fill_value="extrapolate")

    interpolated_pos = np.vstack((
        interp_func_x(lfp_timestamps),
        interp_func_y(lfp_timestamps),
        interp_func_z(lfp_timestamps)
    )).T
    print(f"Interpolated positional data shape: {interpolated_pos.shape}")

    # Combine LFP timestamps, LFP data (all channels), and interpolated positional data
    aligned_array = np.column_stack((
        lfp_timestamps,
        lfp_data.T,  # Transpose to align channels as columns
        interpolated_pos
    ))
    print(f"Aligned array shape: {aligned_array.shape}")

    # Save to cache
    if use_cache:
        np.save(cache_file, aligned_array)
        print(f"Aligned data cached to {cache_file}")

    return aligned_array


def load_and_clean_bat_data(data_path, bat_id, date, lfp_file_path, use_cache=True):
    """
    Loads the LFP data and positional data for a given bat, cleans the positional data
    by interpolating NaN values, and returns the LFP data and cleaned positional data.
    
    Args:
        data_path (str): Path to the directory containing the bat data.
        bat_id (str): ID of the bat.
        date (str): Date of the recording.
        lfp_file_path (str): Path to the LFP data file.
        use_cache (bool): Whether to use cached data for the flight room session.
        
    Returns:
        lfp_data (numpy.ndarray): The loaded LFP data.
        cleaned_pos (numpy.ndarray): The cleaned positional data with NaN values interpolated.
    """
    
    # Load bat LFP data
    lfp_mat = hdf5storage.loadmat(lfp_file_path)
    lfp_data = lfp_mat['lfp']
    print(f"Structure of lfp_data: {type(lfp_data)}, {lfp_data.shape}")
    
    # Load bat positional data
    session = FlightRoomSession(data_path, bat_id, date, use_cache=use_cache)
    pos = session.cortex_data.bat_pos
    print(f"Positional data shape: {pos.shape}")

    # Clean positional data by interpolating NaN values
    cleaned_pos = np.copy(pos)
    cleaned_pos[:, 0] = interpolate_nans(pos[:, 0])
    cleaned_pos[:, 1] = interpolate_nans(pos[:, 1])
    cleaned_pos[:, 2] = interpolate_nans(pos[:, 2])
    print(f"Cleaned positional data shape: {cleaned_pos.shape}")
    
    return lfp_mat, cleaned_pos, session

def extract_and_downsample_lfp_data(lfp_mat, sampling_rate=2500, dfs=25, use_cache=True, cache_file_path='lfp_bat_combined_cache.npy'):
    """
    Extracts LFP data from the specified MATLAB file, optionally downsamples the data, and combines the channels.

    Args:
        lfp_mat (dict): MATLAB data structure containing LFP data.
        sampling_rate (int): Original sampling rate of the LFP data. Default is 2500 Hz.
        dfs (int): Desired final sampling rate after downsampling. Default is 25 Hz.
        use_cache (bool): Whether to use cached data for LFP extraction and decimation.
        cache_file_path (str): Path to the cache file for saving or loading combined LFP data.

    Returns:
        lfp_bat_combined (numpy.ndarray): The optionally downsampled LFP data combined across channels.
    """

    # Check if cache should be used and if the file exists
    if use_cache and os.path.exists(cache_file_path):
        print("Loading cached LFP data...")
        lfp_bat_combined = np.load(cache_file_path)
        print(f"LFP combined shape: {lfp_bat_combined.shape}")
        return lfp_bat_combined

    # Extract subarrays for LFP data
    lfp_data_1 = lfp_mat['lfp'][0, 0]
    lfp_data_2 = lfp_mat['lfp'][0, 1]
    n_channels = lfp_data_1.shape[0]

    print(f"Type of lfp_data_1: {type(lfp_data_1)}, Shape of lfp_data_1: {lfp_data_1.shape}")
    print(f"Type of lfp_data_2: {type(lfp_data_2)}, Shape of lfp_data_2: {lfp_data_2.shape}")
    print(f"Number of channels per array: {n_channels}")

    # Downsample LFP data using get_LFP_from_mat
    lfp_bat_1 = get_LFP_from_mat(lfp_data_1, n_channels, sampling_rate, dfs, use_cache=use_cache)
    lfp_bat_2 = get_LFP_from_mat(lfp_data_2, n_channels, sampling_rate, dfs, use_cache=use_cache)

    # Combine the LFP data (raw or downsampled)
    lfp_bat_combined = np.concatenate((lfp_bat_1, lfp_bat_2), axis=1)
    print(f"LFP combined shape: {lfp_bat_combined.shape}")

    # Save the combined LFP data to cache
    print("Saving LFP data to cache...")
    np.save(cache_file_path, lfp_bat_combined)

    return lfp_bat_combined


# Decimate function defined at the global level for parallel processing
def decimate_func(channel_data, dec):
    """
    Decimates the channel data using scipy.signal.decimate.
    Args:
        channel_data (numpy.ndarray): The data for a single channel.
        dec (int): The decimation factor.
    Returns:
        numpy.ndarray: The decimated data.
    """
    return decimate(channel_data, dec, ftype='fir', zero_phase=True)

def get_LFP_from_mat(lfp_data, n_channels, init_fs, fs=25, use_cache=False, cache_dir='./lfp_cache'):
    """
    Returns raw or decimated LFP data to the desired sampling rate from a MATLAB file.
    
    Args:
        lfp_data (numpy.ndarray): The LFP data array from the MATLAB file.
        n_channels (int): Number of channels in the data.
        init_fs (int): Initial sampling rate of the LFP data (e.g., 2500 Hz).
        fs (int): Desired final sampling rate (default is 25 Hz).
        use_cache (bool): Whether to use cached data.
        cache_dir (str): Directory to store cache files.
        
    Returns:
        numpy.ndarray: Raw or decimated LFP data.
    """
    # Handle cache setup
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = hashlib.md5(lfp_data.tobytes()).hexdigest()
        cache_filename = os.path.join(cache_dir, f"{cache_key}-{n_channels}-{init_fs}-{fs}.npy")
        
        # Return cached data if it exists
        if os.path.exists(cache_filename):
            print(f"Loading cached data from {cache_filename}")
            return np.load(cache_filename, mmap_mode='r')
    
    # If no decimation is needed, return the raw data
    if fs == init_fs:
        print("No decimation needed. Returning raw data.")
        if use_cache:
            # Save raw data to cache
            np.save(cache_filename, lfp_data)
        return lfp_data
    
    # Compute decimation factor
    dec = int(init_fs / fs)
    n_samples = lfp_data.shape[1]
    n_keep = min(255, n_channels) if n_channels > 192 else 192
    final_length = int(np.ceil(n_samples / dec))
    
    # Use memory mapping for large datasets
    X = np.memmap(os.path.join(cache_dir, 'temp_memmap.dat'), dtype='float32', mode='w+', shape=(final_length, n_keep))
    
    # Parallel processing for decimation
    with Pool() as pool:
        results = pool.starmap(decimate_func, [(lfp_data[channel, :], dec) for channel in range(n_keep)])
    
    # Store the decimated results in memory-mapped array
    for channel, result in enumerate(results):
        X[:, channel] = result[:final_length]
    
    # Save decimated data to cache if required
    if use_cache:
        np.save(cache_filename, X)
    
    return X

def get_cluster_labels(session, cluster):
    flight_behavior = session.cortex_data
    labels = np.full([flight_behavior.num_cortex_timebins], 0)
    
    # Get flights for the specified cluster
    cluster_flights = session.get_flights_by_cluster((cluster,))
    
    # Label each flight in the cluster
    for i_flight, flight in enumerate(cluster_flights, start=1):
        s = flight.timebin_start_idx
        e = flight.timebin_end_idx
        labels[s:e] = i_flight
    
    return labels

def get_flight_boolean_array(session, cluster_id, cluster_flight_number, off_samples=0):
    """
    Generate a boolean array for a specific flight within a given cluster and return its phase labels.

    Input:
    - session: A session object containing flight data and cluster information.
    - cluster_id: An integer representing the cluster ID to retrieve the flight from.
    - cluster_flight_number: An integer representing the flight number within the specified cluster.
    - off_samples: Number of samples to include before and after the flight.

    Output:
    - boolean_array: A numpy array of booleans, where True indicates the timebins
      corresponding to the specified flight.
    - phase_labels: A numpy array with values indicating phase: 0 for pre-flight,
      1 for in-flight, and 2 for post-flight.
    - cluster_id: The cluster ID of the flight (same as the input cluster_id).

    Raises:
    - ValueError: If the cluster or flight number is invalid.
    """
    flight_behavior = session.cortex_data
    boolean_array = np.zeros(flight_behavior.num_cortex_timebins, dtype=bool)
    phase_labels = np.full(flight_behavior.num_cortex_timebins, -1, dtype=int)  # Initialize with -1 for unused samples

    # Retrieve flights for the specified cluster
    cluster_flights = session.get_flights_by_cluster([cluster_id])
    
    # Validate cluster_flight_number
    if cluster_flight_number < 1 or cluster_flight_number > len(cluster_flights):
        raise ValueError(f"Invalid cluster flight number. Must be between 1 and {len(cluster_flights)}.")

    # Get the specified flight
    flight = cluster_flights[cluster_flight_number - 1]  # Convert 1-based to 0-based index
    start_idx, end_idx = flight.timebin_start_idx, flight.timebin_end_idx
    pre_start_idx = max(start_idx - off_samples, 0)
    post_end_idx = min(end_idx + off_samples, flight_behavior.num_cortex_timebins)

    # Set boolean array and phase labels
    boolean_array[pre_start_idx:post_end_idx] = True
    if off_samples > 0:
        phase_labels[pre_start_idx:start_idx] = 0  # pre-flight
        phase_labels[start_idx:end_idx] = 1  # in-flight
        phase_labels[end_idx:post_end_idx] = 2  # post-flight
    else:
        phase_labels[start_idx:end_idx] = 1  # in-flight

    return boolean_array, phase_labels

@jit(nopython=True)
def determine_feeder(pos):
    # pos[0] = X, pos[1] = Y, pos[2] = Z
    x = pos[0]
    y = pos[1]
    z = pos[2]
    
    # Expanded thresholds to cover the wider distribution
    if x < -1.0 and y > 1.0 and z > 1.0:
        return 'perch'  # Perch with consideration for higher Z values
    elif x > 1.0 and -1.0 < y < 2.0:
        return 'feeder1'  # Feeder 1 with a broader Y range
    elif x > 1.0 and y < -1.0:
        return 'feeder2'  # Feeder 2 with a similar X but broader Y range
    else:
        return 'unknown'  # Unknown if outside defined regions


def get_flightID(session, binned_pos, valid_indices, lfp_timestamps_decimated_bins, pos_timestamps, off_samples=0):
    """
    Construct a flightID array for all flights in the session with updated classification.

    Input:
    - session: A session object containing flight data and cluster information.
    - binned_pos: A numpy array of pre-filtered binned position data with shape (n, 3) [x, y, z].
    - valid_indices: Array of valid indices to use for the binned position data.
    - lfp_timestamps_decimated_bins: Timestamps for the LFP bins.
    - pos_timestamps: Timestamps for the position data.

    Output:
        flightID: Data stored in the format listed below.

    flightID format:
        Column 0: Flight Number
        Column 1: Flight Type (1-6, see below)
        Column 2: X position in M
        Column 3: Y position in M
        Column 4: Z position in M
        Column 5: Flight Phase (if included)

    Flight Type Codes:
        1: Perch to Feeder 1
        2: Feeder 1 to Perch
        3: Perch to Feeder 2
        4: Feeder 2 to Perch
        5: Feeder 1 to Feeder 2
        6: Feeder 2 to Feeder 1
        0: Other/Unknown
    """
    import numpy as np

    def determine_flight_type(start_pos, end_pos):
        # Use determine_feeder to get start and end locations
        start_location = determine_feeder(start_pos)
        end_location = determine_feeder(end_pos)
        #print(f"Start Location: {start_location}, End Location: {end_location}")

        # Assign flight type code based on start and end locations
        if start_location == 'perch' and end_location == 'feeder1':
            flight_type = 1  # Perch to Feeder 1
        elif start_location == 'feeder1' and end_location == 'perch':
            flight_type = 2  # Feeder 1 to Perch
        elif start_location == 'perch' and end_location == 'feeder2':
            flight_type = 3  # Perch to Feeder 2
        elif start_location == 'feeder2' and end_location == 'perch':
            flight_type = 4  # Feeder 2 to Perch
        elif start_location == 'feeder1' and end_location == 'feeder2':
            flight_type = 5  # Feeder 1 to Feeder 2
        elif start_location == 'feeder2' and end_location == 'feeder1':
            flight_type = 6  # Feeder 2 to Feeder 1
        else:
            flight_type = 0  # Other/Unknown

        return flight_type

    all_flight_data = []
    flight_count = 0

    all_clusters = sorted([int(cluster_id) for cluster_id in session.flights_by_cluster.keys() if int(cluster_id) != 1])

    for cluster_id in all_clusters:
        cluster_flights = session.get_flights_by_cluster([cluster_id])

        for flight in cluster_flights:
            flight_count += 1
            flight_bool, phase_labels, _ = get_flight_boolean_array(session, flight_count, off_samples)

            # Apply valid_indices to the flight boolean array and phase labels
            labels = flight_bool[valid_indices]
            valid_phase_labels = phase_labels[valid_indices]

            # Label timebins for this flight
            timebin_labels = label_timebins(lfp_timestamps_decimated_bins, labels, pos_timestamps, is_discrete=True)

            # Get binned position data for this flight
            flight_pos = binned_pos[timebin_labels > 0]

            # Adjust valid_phase_labels to match timebin_labels
            adjusted_phase_labels = label_timebins(lfp_timestamps_decimated_bins, valid_phase_labels, pos_timestamps, is_discrete=True)
            flight_phases = adjusted_phase_labels[timebin_labels > 0]

            if len(flight_pos) == 0:
                continue  # Skip this flight if no valid positions are found

            if len(flight_pos) != len(flight_phases):
                continue  # Skip this flight if shapes don't match

            # Determine start and end positions
            start_pos = flight_pos[0]
            end_pos = flight_pos[-1]

            # Determine flight type based on start and end positions
            flight_type = determine_flight_type(start_pos, end_pos)

            # Create flight data for all timebins of this flight
            flight_data = np.column_stack((
                np.full(len(flight_pos), flight_count),
                np.full(len(flight_pos), flight_type),
                flight_pos,
                flight_phases  # Include this if you need flight phases
            ))

            all_flight_data.append(flight_data)

    # Concatenate all flight data
    flightID = np.vstack(all_flight_data) if all_flight_data else np.array([])

    return flightID


def get_flightLFP(session, LFPs, valid_indices, lfp_timestamps_decimated_bins, pos_timestamps, off_samples=0):
    """
    Construct a flightLFP array for all flights in the session.
    Input:
    - session: A session object containing flight data and cluster information.
    - LFPs: A numpy array of LFP data with shape (n_timepoints, n_channels)
    - valid_indices: Array of valid indices to use for the LFP data.
    - lfp_timestamps_decimated_bins: Timestamps for the LFP bins.
    - pos_timestamps: Timestamps for the position data.
    - off_samples: Number of samples to include before and after each flight.
    Output:
        flightLFP = Data stored in the format listed below
    flightLFP format:
        Column 0: Flight Number
        Column 1: Flight phase (0: pre-flight, 1: in-flight, 2: post-flight)
        Columns 2+: LFP data for each channel
    """
    all_flight_data = []
    flight_count = 0
    all_clusters = sorted([int(cluster_id) for cluster_id in session.flights_by_cluster.keys() if int(cluster_id) != 1])
    
    print(f"Shape of LFPs: {LFPs.shape}")
    print(f"Length of valid_indices: {len(valid_indices)}")
    print(f"Length of lfp_timestamps_decimated_bins: {len(lfp_timestamps_decimated_bins)}")
    print(f"Length of pos_timestamps: {len(pos_timestamps)}")

    for cluster_id in all_clusters:
        cluster_flights = session.get_flights_by_cluster([cluster_id])
        print(f"Cluster {cluster_id}: {len(cluster_flights)} flights")
        
        for flight in cluster_flights:
            flight_count += 1
            flight_bool, phase_labels, _ = get_flight_boolean_array(session, flight_count, off_samples)
            
            print(f"\nProcessing flight {flight_count}")
            print(f"Length of flight_bool: {len(flight_bool)}")
            print(f"Sum of flight_bool: {np.sum(flight_bool)}")
            
            # Apply valid_indices to the flight boolean array and phase labels
            labels = flight_bool[valid_indices]
            valid_phase_labels = phase_labels[valid_indices]
            
            print(f"Length of labels after valid_indices: {len(labels)}")
            print(f"Sum of labels: {np.sum(labels)}")
            
            # Label timebins for this flight
            timebin_labels = label_timebins(lfp_timestamps_decimated_bins, labels, pos_timestamps, is_discrete=True)
            print(f"Length of timebin_labels: {len(timebin_labels)}")
            print(f"Sum of timebin_labels: {np.sum(timebin_labels)}")
            
            # Get LFP data for this flight
            flight_lfp = LFPs[timebin_labels > 0]
            
            # Adjust valid_phase_labels to match timebin_labels
            adjusted_phase_labels = label_timebins(lfp_timestamps_decimated_bins, valid_phase_labels, pos_timestamps, is_discrete=True)
            flight_phases = adjusted_phase_labels[timebin_labels > 0]

            print(f"Shape of flight_lfp: {flight_lfp.shape}")
            print(f"Shape of flight_phases: {flight_phases.shape}")

            if len(flight_lfp) == 0:
                print(f"Skipping flight {flight_count} due to no valid LFP data")
                continue  # Skip this flight if no valid LFP data are found

            if len(flight_lfp) != len(flight_phases):
                print(f"Warning: Mismatch in shapes for flight {flight_count}")
                print(f"flight_lfp length: {len(flight_lfp)}, flight_phases length: {len(flight_phases)}")
                continue  # Skip this flight if shapes don't match

            # Create flight data for all timebins of this flight
            flight_data = np.column_stack((
                np.full(len(flight_lfp), flight_count),
                flight_phases,
                flight_lfp
            ))

            all_flight_data.append(flight_data)
            print(f"Added flight data with shape: {flight_data.shape}")
            print(f"Unique phase labels: {np.unique(flight_phases)}")

    # Concatenate all flight data
    flightLFP = np.vstack(all_flight_data) if all_flight_data else np.array([])

    print(f"\nFinal flightLFP shape: {flightLFP.shape}")
    return flightLFP

def get_combined_LFP(lfp_mat, init_fs, fs=250, use_cache=False, cache_dir='./lfp_cache'):
    """
    Decimates and combines LFPs from two datasets
    """
    lfp_data_1 = lfp_mat['lfp'][0, 0]
    lfp_data_2 = lfp_mat['lfp'][0, 1]
    n_channels = lfp_data_1.shape[0]
    
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = hashlib.md5(lfp_data_1.tobytes() + lfp_data_2.tobytes()).hexdigest()
        cache_filename = os.path.join(cache_dir, f"{cache_key}-{n_channels}-{init_fs}-{fs}.npy")
        
        if os.path.exists(cache_filename):
            return np.load(cache_filename, mmap_mode='r')
    
    dec = int(init_fs / fs)
    n_samples_1 = lfp_data_1.shape[1]
    n_samples_2 = lfp_data_2.shape[1]
    n_keep = min(255, n_channels) if n_channels > 192 else 192
    final_length_1 = math.ceil(n_samples_1 / dec)
    final_length_2 = math.ceil(n_samples_2 / dec)
    
    # Combine data before decimation
    combined_data = np.concatenate((lfp_data_1, lfp_data_2), axis=1)
    
    # Use memory mapping for large datasets
    X = np.memmap(os.path.join(cache_dir, 'temp_memmap.dat'), dtype='float32', mode='w+', 
                  shape=(n_keep, final_length_1 + final_length_2))
    
    # Parallel processing
    with Pool() as pool:
        decimate_func = partial(decimate, dec=dec)
        results = pool.map(decimate_func, [combined_data[channel, :] for channel in range(n_keep)])
    
    for channel, result in enumerate(results):
        X[channel, :] = result[:(final_length_1 + final_length_2)]
    
    # Transpose to match the original output shape
    X = X.T
    
    if use_cache:
        np.save(cache_filename, X)
    
    return X

def longest_stretch(bool_array):
    """
    Finds longest contiguous stretch of True values

    Input:
    bool_array = boolean vector

    Output:
    bool_most_common = boolean vector, True only for longest stretch of 'True' in bool_array
    """
    bool_array_diff = np.append(bool_array[0], bool_array)
    bool_array_diff = np.cumsum(np.abs(np.diff(bool_array_diff)))
    bool_most_common = bool_array_diff == mode(bool_array_diff[bool_array])[0]
    return bool_most_common
