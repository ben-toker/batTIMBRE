import numpy as np
from numba import jit
import pickle
import hashlib
from multiprocessing import Pool
from functools import partial
from bat.helpers_bat import label_timebins
"""
A set of helper functions for downloading and preprocessing flight path data.

@author: Ben Toker
@author: Kevin Qi 
"""
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

def get_flight_boolean_array(session, global_flight_number, off_samples=0):
    """
    Generate a boolean array for a specific flight across all clusters and return its cluster ID
    and phase labels (pre, in, post flight).

    Input:
    - session: A session object containing flight data and cluster information.
    - global_flight_number: An integer representing the overall flight number to retrieve,
      counting sequentially across all clusters (excluding cluster 1).
    - off_samples: Number of samples to include before and after the flight.

    Output:
    - boolean_array: A numpy array of booleans, where True indicates the timebins
      corresponding to the specified flight.
    - phase_labels: A numpy array with values indicating phase: 0 for pre-flight,
      1 for in-flight, and 2 for post-flight.
    - cluster_id: An integer representing the cluster ID of the specified flight.

    Raises:
    - ValueError: If the global_flight_number is invalid (i.e., higher than the total
      number of flights across all clusters).

    Note:
    - Cluster 1 is excluded from the count.
    - Flights are counted sequentially across clusters in ascending order of cluster IDs.
    """
    flight_behavior = session.cortex_data
    boolean_array = np.zeros(flight_behavior.num_cortex_timebins, dtype=bool)
    phase_labels = np.full(flight_behavior.num_cortex_timebins, -1, dtype=int)  # Initialize with -1 for unused samples
    
    all_clusters = sorted([int(cluster_id) for cluster_id in session.flights_by_cluster.keys() if int(cluster_id) != 1])
    
    flight_count = 0
    for cluster_id in all_clusters:
        cluster_flights = session.get_flights_by_cluster([cluster_id])
        for flight in cluster_flights:
            flight_count += 1
            if flight_count == global_flight_number:
                start_idx, end_idx = flight.timebin_start_idx, flight.timebin_end_idx
                pre_start_idx = max(start_idx - off_samples, 0)
                post_end_idx = min(end_idx + off_samples, flight_behavior.num_cortex_timebins)
                boolean_array[pre_start_idx:post_end_idx] = True

                # Set phase labels
                if off_samples > 0:
                    phase_labels[pre_start_idx:start_idx] = 0  # pre-flight
                    phase_labels[start_idx:end_idx] = 1  # in-flight
                    phase_labels[end_idx:post_end_idx] = 2  # post-flight
                else:
                    phase_labels[start_idx:end_idx] = 1  # in-flight

                return boolean_array, phase_labels, cluster_id
    
    raise ValueError(f"Invalid flight number. Must be between 1 and {flight_count}")

def get_flightID_modified(session, binned_pos, valid_indices, lfp_timestamps_decimated_bins, pos_timestamps):
    flightID = np.zeros((len(binned_pos), 5), dtype=float)  # Initialize with non-flight state
    flightID[:, 1] = -1  # Set feeder visited to -1 for non-flight periods
    flightID[:, 2:5] = binned_pos  # Set position data for all time points
    
    flight_count = 0
    for cluster_id in session.flights_by_cluster.keys():
        if cluster_id == 1:  # Skip unstructured flights if desired
            continue
        for flight in session.get_flights_by_cluster([cluster_id]):
            flight_count += 1
            flight_bool, _ = get_flight_boolean_array(session, flight_count)
            labels = flight_bool[valid_indices]
            timebin_labels = label_timebins(lfp_timestamps_decimated_bins, labels, pos_timestamps, is_discrete=True)
            
            flight_indices = np.where(timebin_labels > 0)[0]
            if len(flight_indices) > 0:
                flightID[flight_indices, 0] = flight_count
                flightID[flight_indices, 1] = determine_feeder(binned_pos[flight_indices[-1], :])
    
    return flightID



@jit(nopython=True)
def determine_feeder(end_pos):
    if end_pos[1] > 0 and end_pos[0] < 0:
        return 0  # perch
    elif end_pos[1] > 0 and end_pos[0] > 0:
        return 1  # feeder 1
    elif end_pos[1] < 0 and end_pos[0] > 0:
        return 2  # feeder 2
    else:
        return -1  # unknown

import numpy as np

def get_flightID(session, binned_pos, valid_indices, lfp_timestamps_decimated_bins, pos_timestamps, off_samples=0):
    """
    Construct a flightID array for all flights in the session.
    Input:
    - session: A session object containing flight data and cluster information.
    - binned_pos: A numpy array of pre-filtered binned position data with shape (n, 3) [x, y, z].
    - valid_indices: Array of valid indices to use for the binned position data.
    - lfp_timestamps_decimated_bins: Timestamps for the LFP bins.
    - pos_timestamps: Timestamps for the position data.
    Output:
        flightID = Data stored in the format listed below
    flightID format:
        Column 0: Flight Number
        Column 1: Feeder visited (0: perch, 1: feeder 1, 2: feeder 2)
        Column 2: X position in M
        Column 3: Y position in M
        Column 4: Z position in M
    """
    all_flight_data = []
    flight_count = 0

    all_clusters = sorted([int(cluster_id) for cluster_id in session.flights_by_cluster.keys() if int(cluster_id) != 1])

    print(f"Total clusters: {len(all_clusters)}")
    print(f"Shape of binned_pos: {binned_pos.shape}")
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
            
            # Get binned position data for this flight
            flight_pos = binned_pos[timebin_labels > 0]
            
            # Adjust valid_phase_labels to match timebin_labels
            adjusted_phase_labels = label_timebins(lfp_timestamps_decimated_bins, valid_phase_labels, pos_timestamps, is_discrete=True)
            flight_phases = adjusted_phase_labels[timebin_labels > 0]

            print(f"Shape of flight_pos: {flight_pos.shape}")
            print(f"Shape of flight_phases: {flight_phases.shape}")

            if len(flight_pos) == 0:
                print(f"Skipping flight {flight_count} due to no valid positions")
                continue  # Skip this flight if no valid positions are found

            if len(flight_pos) != len(flight_phases):
                print(f"Warning: Mismatch in shapes for flight {flight_count}")
                print(f"flight_pos length: {len(flight_pos)}, flight_phases length: {len(flight_phases)}")
                continue  # Skip this flight if shapes don't match

            # Determine feeder visited based on end position
            end_pos = flight_pos[-1]
            feeder = determine_feeder(end_pos)

            # Create flight data for all timebins of this flight
            flight_data = np.column_stack((
                np.full(len(flight_pos), flight_count),
                np.full(len(flight_pos), feeder),
                flight_pos,
                flight_phases
            ))

            all_flight_data.append(flight_data)
            print(f"Added flight data with shape: {flight_data.shape}")
            print(f"Unique phase labels: {np.unique(flight_phases)}")

    # Concatenate all flight data
    flightID = np.vstack(all_flight_data) if all_flight_data else np.array([])

    print(f"\nFinal flightID shape: {flightID.shape}")
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

def decimate_channel(channel_data, dec):
    return channel_data[::dec]

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
        decimate_func = partial(decimate_channel, dec=dec)
        results = pool.map(decimate_func, [combined_data[channel, :] for channel in range(n_keep)])
    
    for channel, result in enumerate(results):
        X[channel, :] = result[:(final_length_1 + final_length_2)]
    
    # Transpose to match the original output shape
    X = X.T
    
    if use_cache:
        np.save(cache_filename, X)
    
    return X
