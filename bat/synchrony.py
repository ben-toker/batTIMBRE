import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def test_synchronization(LFPs, binned_pos, lfp_timestamps_dec):
    """
    Tests the synchronization between LFP data and positional data by computing cross-correlation
    for all channels and dimensions, and returns the average lag.
    
    Parameters:
    LFPs (ndarray): The LFP data array of shape (samples, channels).
    binned_pos (ndarray): The binned positional data array of shape (samples, dimensions).
    lfp_timestamps_dec (ndarray): Decimated LFP timestamps in microseconds.
    """
    # Convert timestamps to seconds
    timestamps_sec = lfp_timestamps_dec / 1e6
    
    num_channels = LFPs.shape[1]
    num_dimensions = binned_pos.shape[1]
    
    all_lags = []
    
    for lfp_channel in range(num_channels):
        for pos_dim in range(num_dimensions):
            lfp_data = np.abs(LFPs[:, lfp_channel])
            pos_data = binned_pos[:, pos_dim]
            
            # Remove NaN values
            valid_indices = ~np.isnan(pos_data)
            lfp_data = lfp_data[valid_indices]
            pos_data = pos_data[valid_indices]
            
            # Compute cross-correlation
            correlation = signal.correlate(lfp_data, pos_data, mode='full')
            lags = signal.correlation_lags(len(lfp_data), len(pos_data))
            
            # Convert lags to seconds
            time_step = np.mean(np.diff(timestamps_sec))
            lags_sec = lags * time_step
            
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
    
    # Plot average cross-correlation
    avg_lfp_data = np.mean(np.abs(LFPs), axis=1)
    avg_pos_data = np.mean(binned_pos, axis=1)
    
    # Remove NaN values
    valid_indices = ~np.isnan(avg_pos_data)
    avg_lfp_data = avg_lfp_data[valid_indices]
    avg_pos_data = avg_pos_data[valid_indices]
    
    # Ensure the data have the same length
    min_length = min(len(avg_lfp_data), len(avg_pos_data))
    avg_lfp_data = avg_lfp_data[:min_length]
    avg_pos_data = avg_pos_data[:min_length]
    
    # Compute cross-correlation
    avg_correlation = signal.correlate(avg_lfp_data, avg_pos_data, mode='full')
    avg_lags = signal.correlation_lags(len(avg_lfp_data), len(avg_pos_data))
    avg_lags_sec = avg_lags * time_step
    
    plt.figure(figsize=(10, 5))
    plt.plot(avg_lags_sec, avg_correlation)
    plt.title('Average Cross-correlation between LFP and Position Data')
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
        plt.plot(time, LFPs[start_time:end_time, i] + i * 200)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (offset for visibility)')
    plt.title('LFP Data Visualization (Reals after Hilbert Transform)')
    plt.show()