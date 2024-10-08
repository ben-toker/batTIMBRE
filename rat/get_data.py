"""
A set of helper functions for downloading and preprocessing hippocampal data.

@author: Gautam Agarwal
@author: Ben Toker
"""
from scipy.signal import decimate
import numpy as np
import math
import os
from scipy import io
from scipy.stats import mode
import pickle
import hashlib
from multiprocessing import Pool
from functools import partial


def get_behav(mat_file, fs=25,init_fs=1250):
    """
    Organizes information about rat behavior into a matrix
    
    Input:
    mat_file: file containing behavioral data
    fs = desired sampling frequency
    init_fs = initial sampling frequency of the data

    Output:
    lapID = Data stored in the format listed below

    lapID format:
    Column 0: Trial Number
    Column 1: Maze arm (-1/0/1/2) (-1 = not in maze arm)
    Column 2: Correct (0/1)
    Column 3: other/first approach/port/last departure (0/1/2/3)
    Column 4: x position in mm
    Column 5: y position in mm
    """
    dec = int(init_fs / fs)  # decimation factor
    mat = io.loadmat(mat_file, variable_names=['Track'])
    lapID = np.array([np.squeeze(mat['Track']["lapID"][0][0])[::dec]], dtype='float32') - 1
    lapID = np.append(lapID, [np.squeeze(mat['Track']["mazeSect"][0][0])[::dec]], axis=0)
    lapID = np.append(lapID, [np.squeeze(mat['Track']["corrChoice"][0][0])[::dec]], axis=0)
    lapID = np.append(lapID, np.zeros((1, len(lapID[0]))), axis=0)
    lapID = np.append(lapID, decimate(mat['Track']["xMM"][0][0].T, dec), axis=0)
    lapID = np.append(lapID, decimate(mat['Track']["yMM"][0][0].T, dec), axis=0)
    lapID = lapID.T

    # Filter values and construct column 3
    in_arm = np.in1d(lapID[:, 1], np.array(range(4, 10)))  # rat is occupying a maze arm
    in_end = np.in1d(lapID[:, 1], np.array(range(7, 10)))
    # lapID[np.in1d(lapID[:,1], np.array(range(4, 10)), invert = True), 1] = -1
    lapID[in_arm, 1] = (lapID[in_arm, 1] - 1) % 3
    lapID[~in_arm, 1] = -1
    # lapID[lapID[:, 1] == 0, :] = 0

    for i in range(int(np.max(lapID[:, 0]))):
        r = np.logical_and(lapID[:, 0] == i, in_end)  # lapID[:, 3] == 2)
        inds = np.where(np.logical_and(lapID[:, 0] == i, in_arm))[0]
        all_end = np.where(r)[0]
        # if all_end.size > 0: #valid trial where rat goes to end of arm
        lapID[inds[inds < all_end[0]], 3] = 1
        lapID[inds[inds > all_end[-1]], 3] = 3
        lapID[longest_stretch(r), 3] = 2

    # Return structured data
    return lapID


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


def get_spikes(mat_file, fs=25,init_fs=1250): 
    """
    Counts spikes for each neuron in each time bin
    
    Input:
    mat_file = file containing spike data
    fs = desired sampling rate
    init_fs = initial sampling rate of the data
    
    Output:
    sp = binned spikes
    """
    mat = io.loadmat(mat_file, variable_names=['Spike', 'Clu', 'xml'])
    n_channels = mat['xml']['nChannels'][0][0][0][0]
    dec = int(init_fs / fs)
    max_spike_res = np.ceil(np.max(mat['Spike']['res'][0][0]) / dec) + 1
    max_spike_clu = np.max(mat['Spike']['totclu'][0][0]) + 1  # Precompute the bins
    bins_res = np.arange(max_spike_res)
    bins_clu = np.arange(max_spike_clu)
    spike_res = np.squeeze(mat['Spike']['res'][0][0]) // dec
    spike_clu = np.squeeze(mat['Spike']['totclu'][0][0]) - 1

    # Bin both dimensions using histogram2d.
    sp, _, _ = np.histogram2d(spike_res, spike_clu, bins=(bins_res, bins_clu))
    sp = sp.astype(np.uint8)

    mask = mat['Clu']['shank'][0][0][0] <= math.ceil(n_channels / 8)
    sp = sp[:, mask]

    return sp


def get_LFP(lfp_file, n_channels, init_fs, fs=25):
    """
    Decimates LFPs to desired sampling rate
    
    Input:
    lfp_file = raw lfp data file of type .lfp
    init_fs = inital sampling rate of the data
    fs = desired sampling rate (to decimate to)

    Output:
    X = formatted lfp data
    """
    dec = int(init_fs / fs)
    file_size = os.path.getsize(lfp_file)
    data_size = np.dtype('int16').itemsize
    total_elements = file_size // data_size
    n_samples = total_elements // n_channels

    # Clip the rows to remove electrodes implanted in mPFC.
    if n_channels > 256:  # sessions 1 and 2
        n_keep = 255
    else:  # sessions 3 and 4
        n_keep = 192

    # Load and decimate the data (takes more memory!)
    # slice_data = np.memmap(lfp_file, dtype='int16', mode='r', shape=(n_samples, n_channels))
    # X = decimate(slice_data[:, :n_keep], dec, axis=0)

    # Process each channel individually and store in the pre-allocated array (takes less memory)
    final_length = math.ceil(n_samples / dec)
    X = np.zeros((final_length, n_keep), dtype=np.float32)
    for channel in range(n_keep):
        # Load the channel data using memmap
        channel_data = np.memmap(lfp_file, dtype='int16', mode='r', shape=(n_samples, n_channels))[:, channel]
        # Decimate the channel data
        X[:, channel] = decimate(channel_data, dec, axis=0)
        print(channel)

    return X

def decimate_channel(channel_data, dec):
    return channel_data[::dec]

def get_LFP_from_mat(lfp_data, n_channels, init_fs, fs=25, use_cache=False, cache_dir='./lfp_cache'):
    """
    Decimates LFPs to desired sampling rate from a MATLAB file
    """
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = hashlib.md5(lfp_data.tobytes()).hexdigest()
        cache_filename = os.path.join(cache_dir, f"{cache_key}-{n_channels}-{init_fs}-{fs}.npy")
        
        if os.path.exists(cache_filename):
            return np.load(cache_filename, mmap_mode='r')
    
    dec = int(init_fs / fs)
    n_samples = lfp_data.shape[1]
    n_keep = min(255, n_channels) if n_channels > 192 else 192
    final_length = math.ceil(n_samples / dec)
    
    # Use memory mapping for large datasets
    X = np.memmap(os.path.join(cache_dir, 'temp_memmap.dat'), dtype='float32', mode='w+', shape=(final_length, n_keep))
    
    # Parallel processing
    with Pool() as pool:
        decimate_func = partial(decimate_channel, dec=dec)
        results = pool.map(decimate_func, [lfp_data[channel, :] for channel in range(n_keep)])
    
    for channel, result in enumerate(results):
        X[:, channel] = result[:final_length]
    
    if use_cache:
        np.save(cache_filename, X)
    
    return X

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

