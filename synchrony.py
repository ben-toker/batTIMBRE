import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from helpers import interpolate_nans
from scipy.signal import decimate
from helpers import label_timebins

def sync_and_bin_data(lfp_mat, session, cleaned_pos, fs=25):
    """
    Synchronizes the LFP and positional data, optionally decimates LFP timestamps, and bins positional data accordingly.
    
    Args:
        lfp_mat (dict): The loaded LFP MATLAB data.
        session (FlightRoomSession): The session object containing bat positional data.
        cleaned_pos (numpy.ndarray): The cleaned positional data.
        fs (int): Desired sampling frequency after decimation.
    
    Returns:
        lfp_timestamps_edges (numpy.ndarray): The edges of the LFP time bins.
        binned_pos (numpy.ndarray): The positional data binned according to LFP time bins.
    """
    # Extract LFP timestamps in microseconds
    lfp_timestamps = lfp_mat['global_sample_timestamps_usec']
    print(f"LFP timestamps structure: {lfp_timestamps.shape}")
    
    # Original sampling frequency in Hz
    original_fs = 2500  
    
    # Determine if decimation is needed
    if fs == original_fs:
        print("Using raw data without decimation.")
        lfp_timestamps_dec = lfp_timestamps.squeeze()
    else:
        # Calculate the decimation factor
        decimation_factor = int(original_fs // fs)
        print(f"Decimation factor: {decimation_factor}")
        
        # Decimate LFP timestamps
        lfp_timestamps_dec = decimate(lfp_timestamps.squeeze(), decimation_factor)
        print(f"Decimated LFP timestamps shape: {lfp_timestamps_dec.shape}")
    
    # Lop off negative timestamps and prepare the LFP timestamp edges
    lfp_indices = lfp_timestamps_dec > 0  # Filter out negative timestamps
    lfp_timestamps_dec = lfp_timestamps_dec[lfp_indices]
    lfp_timestamps_dec >= session.cortex_data.cortex_global_sample_timestamps_sec.min() * 1e6
    lfp_timestamps_edges = np.insert(lfp_timestamps_dec, 0, 0)  # Insert 0 at the beginning for proper binning
    print(f"LFP timestamp edges shape: {lfp_timestamps_edges.shape}")

    # Extract and clean positional data timestamps (convert to microseconds)
    pos_timestamps = session.cortex_data.cortex_global_sample_timestamps_sec * 1e6
    valid_indices = pos_timestamps > 0  # Filter out negative timestamps
    pos_timestamps = pos_timestamps[valid_indices]
    print(f"Positional timestamp diff (microseconds): {np.diff(pos_timestamps)}")  
    
    # Clean the corresponding positional data
    cleaned_pos = cleaned_pos[valid_indices]
    print(f"Cleaned positional data shape: {cleaned_pos.shape}")

    # Bin positional data using the provided label_timebins function
    binned_pos_x = label_timebins(lfp_timestamps_edges, cleaned_pos[:, 0], pos_timestamps, is_discrete=False)
    binned_pos_y = label_timebins(lfp_timestamps_edges, cleaned_pos[:, 1], pos_timestamps, is_discrete=False)
    binned_pos_z = label_timebins(lfp_timestamps_edges, cleaned_pos[:, 2], pos_timestamps, is_discrete=False)
    
    # Combine the binned positional data into a single array
    binned_pos = np.column_stack((binned_pos_x, binned_pos_y, binned_pos_z))
    print(f"Binned positional data shape: {binned_pos.shape}")
    
    return lfp_timestamps_edges, binned_pos, pos_timestamps, lfp_indices, valid_indices



def test_synchronization(flightID, flightLFP, fs=25):
    """
    Tests the synchronization between LFP data and positional data for each flight
    by computing cross-correlation for all channels and dimensions, and returns the average lag.
    
    Parameters:
    flightID (ndarray): The flight position data array of shape (samples, 5).
    flightLFP (ndarray): The flight LFP data array of shape (samples, channels + 2).
    fs (float): Sampling frequency in Hz, default is 25.
    """
    num_channels = flightLFP.shape[1] - 2  # Subtract 2 for flight number and phase columns
    num_dimensions = 3  # X, Y, Z positions
    
    all_lags = []
    
    unique_flights = np.unique(flightID[:, 0])
    
    for flight_num in unique_flights:
        flight_mask = flightID[:, 0] == flight_num
        flight_pos = flightID[flight_mask, 2:5]  # X, Y, Z positions
        flight_lfp = flightLFP[flightLFP[:, 0] == flight_num, 2:]  # LFP data
        
        for lfp_channel in range(num_channels):
            for pos_dim in range(num_dimensions):
                lfp_data = np.abs(flight_lfp[:, lfp_channel])
                pos_data = flight_pos[:, pos_dim]
                
                # Remove NaN values
                valid_indices = ~np.isnan(pos_data)
                lfp_data = lfp_data[valid_indices]
                pos_data = pos_data[valid_indices]
                
                # Compute cross-correlation
                max_lag = int(5 * fs)  # 5 seconds max lag
                correlation = signal.correlate(lfp_data, pos_data, mode='same', method='fft')
                lags = np.arange(-max_lag, max_lag+1)
                lags_sec = lags / fs
                
                # Find the lag with maximum correlation
                max_corr_lag = lags_sec[np.argmax(correlation)]
                all_lags.append(max_corr_lag)
    
    # Compute average lag
    average_lag = np.mean(all_lags)
    std_lag = np.std(all_lags)
    
    # Plot histogram of lags
    plt.figure(figsize=(10, 5))
    plt.hist(all_lags, bins=50)
    plt.title('Distribution of Maximum Correlation Lags')
    plt.xlabel('Lag (seconds)')
    plt.ylabel('Frequency')
    plt.axvline(x=average_lag, color='r', linestyle='--', label='Average Lag')
    plt.legend()
    plt.show()
    
    print(f"Average maximum correlation lag: {average_lag:.6f} seconds")
    print(f"Standard deviation of lags: {std_lag:.6f} seconds")
    
    max_lag = int(5 * fs)  # 5 seconds max lag
    avg_correlation = np.zeros(2 * max_lag + 1)
    lags = np.arange(-max_lag, max_lag+1)
    avg_lags_sec = lags / fs

    for flight_num in unique_flights:
        flight_mask = flightID[:, 0] == flight_num
        avg_lfp_data = np.mean(np.abs(flightLFP[flightLFP[:, 0] == flight_num, 2:]), axis=1)
        avg_pos_data = np.mean(flightID[flight_mask, 2:5], axis=1)
        
        # Remove NaN values
        valid_indices = ~np.isnan(avg_pos_data)
        avg_lfp_data = avg_lfp_data[valid_indices]
        avg_pos_data = avg_pos_data[valid_indices]
        
        # Ensure the data have the same length
        min_length = min(len(avg_lfp_data), len(avg_pos_data))
        avg_lfp_data = avg_lfp_data[:min_length]
        avg_pos_data = avg_pos_data[:min_length]
        
        # Compute cross-correlation
        flight_correlation = signal.correlate(avg_lfp_data, avg_pos_data, mode='full')
        
        # Ensure the correlation array is centered and has the correct length
        center = len(flight_correlation) // 2
        start = max(0, center - max_lag)
        end = min(len(flight_correlation), center + max_lag + 1)
        flight_correlation_trimmed = flight_correlation[start:end]
        
        # Pad or trim to ensure consistent length
        if len(flight_correlation_trimmed) < len(avg_correlation):
            flight_correlation_trimmed = np.pad(flight_correlation_trimmed, 
                                                (0, len(avg_correlation) - len(flight_correlation_trimmed)),
                                                mode='constant')
        elif len(flight_correlation_trimmed) > len(avg_correlation):
            flight_correlation_trimmed = flight_correlation_trimmed[:len(avg_correlation)]
        
        avg_correlation += flight_correlation_trimmed

    avg_correlation /= len(unique_flights)
    
    plt.figure(figsize=(10, 5))
    plt.plot(avg_lags_sec, avg_correlation)
    plt.title('Average Cross-correlation between LFP and Position Data (All Flights)')
    plt.xlabel('Lag (seconds)')
    plt.ylabel('Correlation')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.show()

def plot_raw_lfp(LFPs, n_channels=5, start_time=0, end_time=1000, fs=25):
    """
    Plots the raw LFP data for the first n_channels within the specified time window.
    Parameters:
    LFPs (ndarray): The LFP data array of shape (samples, channels).
    n_channels (int): Number of channels to plot.
    start_time (int): Start time index.
    end_time (int): End time index.
    fs (int): Sampling frequency (Hz).
    """
    time = np.arange(start_time, end_time) / fs  # Time axis in seconds
    plt.figure(figsize=(10, 5))

    for i in range(n_channels):
        plt.plot(time, LFPs[start_time:end_time, i]+ i * 200)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (offset for visibility)')
    plt.title('LFP Data Visualization (Reals after Hilbert Transform)')
    plt.show()
    
