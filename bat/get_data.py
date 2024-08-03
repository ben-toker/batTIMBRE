import numpy as np
from numba import jit
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

def get_flight_boolean_array(session, global_flight_number):
    """
    Generate a boolean array for a specific flight across all clusters and return its cluster ID.

    Input:
    - session: A session object containing flight data and cluster information.
    - global_flight_number: An integer representing the overall flight number to retrieve,
      counting sequentially across all clusters (excluding cluster 1).

    Output:
    - boolean_array: A numpy array of booleans, where True indicates the timebins
      corresponding to the specified flight.
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
    
    all_clusters = sorted([int(cluster_id) for cluster_id in session.flights_by_cluster.keys() if int(cluster_id) != 1])
    
    flight_count = 0
    for cluster_id in all_clusters:
        cluster_flights = session.get_flights_by_cluster([cluster_id])
        for flight in cluster_flights:
            flight_count += 1
            if flight_count == global_flight_number:
                start_idx, end_idx = flight.timebin_start_idx, flight.timebin_end_idx
                boolean_array[start_idx:end_idx] = True
                return boolean_array, cluster_id
    
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

def get_flightID(session, binned_pos, valid_indices, lfp_timestamps_decimated_bins, pos_timestamps):
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

    for cluster_id in all_clusters:
        cluster_flights = session.get_flights_by_cluster([cluster_id])
        for flight in cluster_flights:
            flight_count += 1
            flight_bool, _ = get_flight_boolean_array(session, flight_count)
            
            # Apply valid_indices to the flight boolean array
            labels = flight_bool[valid_indices]
            
            # Label timebins for this flight
            timebin_labels = label_timebins(lfp_timestamps_decimated_bins, labels, pos_timestamps, is_discrete=True)
            
            # Get binned position data for this flight
            flight_pos = binned_pos[timebin_labels > 0]

            if len(flight_pos) == 0:
                continue  # Skip this flight if no valid positions are found

            # Determine feeder visited based on end position
            end_pos = flight_pos[-1]
            feeder = determine_feeder(end_pos)

            # Create flight data for all timebins of this flight
            flight_data = np.column_stack((
                np.full(len(flight_pos), flight_count),
                np.full(len(flight_pos), feeder),
                flight_pos
            ))

            all_flight_data.append(flight_data)

    # Concatenate all flight data
    flightID = np.vstack(all_flight_data)

    return flightID

import numpy as np
from bat.get_data import get_flight_boolean_array
from bat.helpers_bat import label_timebins

def get_flightLFP(session, LFPs, valid_indices, lfp_timestamps_decimated_bins, pos_timestamps):
    """
    Construct a flightLFP array for all flights in the session.
    Input:
    - session: A session object containing flight data and cluster information.
    - LFPs: A numpy array of LFP data with shape (n_timepoints, n_channels)
    - valid_indices: Array of valid indices to use for the LFP data.
    - lfp_timestamps_decimated_bins: Timestamps for the LFP bins.
    - pos_timestamps: Timestamps for the position data.
    Output:
        flightLFP = Data stored in the format listed below
    flightLFP format:
        Column 0: Flight Number
        Columns 1+: LFP data for each channel
    """
    all_flight_data = []
    flight_count = 0

    all_clusters = sorted([int(cluster_id) for cluster_id in session.flights_by_cluster.keys() if int(cluster_id) != 1])

    for cluster_id in all_clusters:
        cluster_flights = session.get_flights_by_cluster([cluster_id])
        for flight in cluster_flights:
            flight_count += 1
            flight_bool, _ = get_flight_boolean_array(session, flight_count)
            
            # Apply valid_indices to the flight boolean array
            labels = flight_bool[valid_indices]
            
            # Label timebins for this flight
            timebin_labels = label_timebins(lfp_timestamps_decimated_bins, labels, pos_timestamps, is_discrete=True)
            
            # Get LFP data for this flight
            flight_lfp = LFPs[timebin_labels > 0]

            if len(flight_lfp) == 0:
                continue  # Skip this flight if no valid LFP data are found

            # Create flight data for all timebins of this flight
            flight_data = np.column_stack((
                np.full(len(flight_lfp), flight_count),
                flight_lfp
            ))

            all_flight_data.append(flight_data)

    # Concatenate all flight data
    flightLFP = np.vstack(all_flight_data)

    return flightLFP