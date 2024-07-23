import hdf5storage
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
import os
from scipy.ndimage import gaussian_filter
#import sklearn
import pickle
import glob

import pandas as pd

class FlightRoomSession():
    def __init__(self, data_path, bat_id, date, use_cache=True):
        self.data_path = data_path
        self.date = date
        self.bat_id = bat_id
        self.processed_path = data_path
        
        self.flights_by_cluster = {}
        self.flights_by_start_pos = {}
        self.flights_by_end_pos = {}
        self.flights = []
        self.num_flights = 0
        
        # Load ephys data and cortex data 
        if(os.path.exists(os.path.join(data_path, f'{bat_id}_{date}_cache.pkl')) and use_cache):
            print(f'Loading Flight Room | {bat_id} | {date} from cache...')
            with open(os.path.join(data_path, f'{bat_id}_{date}_cache.pkl'), 'rb') as f:
                self.spike_data, self.cortex_data = pickle.load(f)
        else:
            print("Loading cortex data...")
            self.cortex_data = CortexData(self.data_path, self.bat_id, self.date)
            print("Loading spike data...")
            self.spike_data = SpikeData(self.data_path, self.date, self.bat_id)
            
            with open(os.path.join(data_path, f'{bat_id}_{date}_cache.pkl'), 'wb') as f:
                print(f'Saving Flight Room | {bat_id} | {date} to cache...')
                pickle.dump([self.spike_data, self.cortex_data], f)

        # Construct flights
        for i_flight in range(self.cortex_data.num_flights):
            cluster_id = self.cortex_data.cluster_ids[i_flight]
            s = self.cortex_data.flight_start_idx[i_flight]
            e = self.cortex_data.flight_end_idx[i_flight]
            flight_start_sec = self.cortex_data.cortex_global_sample_timestamps_sec[s]
            flight_end_sec = self.cortex_data.cortex_global_sample_timestamps_sec[e]
            #kalman_estimate = kalman_filter_orientation(self.measured_hd_state[s:e,:], transition_matrix, observation_matrix, observation_covariance)
            #kalman_orientation = kalman_estimate[:,np.array([0,2,4])]
            #kalman_angular_velocity = kalman_estimate[:,np.array([1,3,5])]
            self.cortex_hd = np.full([e-s,3],np.nan)
            self.cortex_hd[:,0] = self.cortex_data.raw_azimuth[s:e]
            self.cortex_hd[:,1] = self.cortex_data.raw_pitch[s:e]
            self.cortex_hd[:,2] = self.cortex_data.raw_roll[s:e]
            angular_velocity = np.full([e-s,3],np.nan)
            flight = Flight(s, 
                            e, 
                            flight_start_sec,
                            flight_end_sec,
                            cluster_id, 
                            self.cortex_data.bat_pos[s:e,:], 
                            self.cortex_hd, 
                            self.cortex_data.bat_pos[s:e,:], 
                            self.cortex_hd, 
                            angular_velocity)
            if(flight.cluster_id not in self.flights_by_cluster):
                self.flights_by_cluster[flight.cluster_id] = []
            if(flight.start_pos[0] not in self.flights_by_start_pos):
                self.flights_by_start_pos[flight.start_pos[0]] = []
            if(flight.end_pos[0] not in self.flights_by_end_pos):
                self.flights_by_end_pos[flight.end_pos[0]] = []

            self.flights.append(flight)
            self.flights_by_cluster[flight.cluster_id].append(flight)
            self.flights_by_start_pos[flight.start_pos[0]].append(flight)
            self.flights_by_end_pos[flight.end_pos[0]].append(flight)
            self.num_flights += 1
    

    def get_flights_by_cluster(self, cluster_ids):
        if(type(cluster_ids) == int):
            return self.flights_by_cluster[cluster_ids]
        else:
            out = []
            for cluster_id in cluster_ids:
                out += self.flights_by_cluster[cluster_id]
            return out
    
    def get_flights_by_start_pos(self, start_pos):
        return self.flights_by_start_pos[start_pos]
    
    def get_flights_by_end_pos(self, end_pos):
        return self.flights_by_end_pos[end_pos]
    
    def get_all_cluster_ids(self):
        return np.sort(list(self.flights_by_cluster.keys()))

    def __repr__(self):
        s1 = f'Flights object containing {self.num_flights} flights. Use get_flights_by_cluster, get_flights_by_start_pos, or get_flights_by_end_pos to access flights.'

        # Cluster IDs sorted by number of flights
        ranked_clusters = sorted(self.flights_by_cluster.keys(), key= lambda x:len(self.flights_by_cluster[x]), reverse=True)

        s2 = f'Number of flights by cluster in descending order: {[(cluster_id, len(self.flights_by_cluster[cluster_id])) for cluster_id in ranked_clusters]}'
        
        return f'Flight Room Session | {self.bat_id} | {self.date}\n' + s1 + '\n' + s2
    
class CortexData():
    def __init__(self, data_path, bat_id, date):
        self.data_path = data_path
        self.bat_id = bat_id
        self.date = date
        self.load_cortex_data()
        self.construct_flights()
    
    def load_cortex_data(self):
        data_path = self.data_path
        bat_id = self.bat_id
        date = self.date

        print("Loading flight paths data...")
        flight_paths_data = hdf5storage.loadmat(os.path.join(data_path, f'{bat_id}_{date}_flight_paths.mat'))
        print("Done!")
        self.cortex_data = flight_paths_data['cortexData'][0][0]
        self.flight_paths = flight_paths_data['flightPaths'][0][0]
        self.cortex_local_ttl_timestamps_usec = self.cortex_data['local_ttl_usec']
        self.cortex_global_sample_timestamps_sec = self.cortex_data['global_sample_ts_usec'][0,:] / 1e6
        
        self.num_ttls = len(self.cortex_local_ttl_timestamps_usec)
        self.ttl_interval = 3 # sec
        
        self.cluster_ids = self.flight_paths[0].T[0]
        self.flight_start_idx = self.flight_paths[1][0]
        self.flight_end_idx = self.flight_paths[2][0]
        self.bat_pos = self.flight_paths[13].T
        self.num_flights = len(self.flight_start_idx)
        
        #is_during = np.logical_and(cortex_global_sample_timestamps_sec < num_ttls*3, cortex_global_sample_timestamps_sec > 0)
        
        self.raw_azimuth = self.cortex_data[9][:,0]
        self.raw_pitch = self.cortex_data[10][:,0]
        self.raw_roll = self.cortex_data[11][:,0]
        self.raw_pos = self.cortex_data['avgMarkerPos']
        
        self.num_cortex_timebins = len(self.cortex_global_sample_timestamps_sec)

        print("Loading analog data...")
        analog_data = hdf5storage.loadmat(os.path.join(data_path, f'ephys/{bat_id}_{date}_analog.mat'))
        print("Done!")
        
        self.gyro_voltage_scaling = 815 # voltage / (rad/s)
        
        self.gyro = analog_data['gyro']/self.gyro_voltage_scaling  # pitch, roll, yaw
        self.analog_global_sample_timestamps_sec = analog_data['global_sample_timestamps_usec'][:,0] / 1e6

        self.accel = analog_data['accel']

        # Free up memory
        del flight_paths_data
        del analog_data

    def construct_flights(self):
        dt = 1/120
        transition_matrix = [[1, dt,  0,  0,  0,  0],
                         [0,  1,  0,  0,  0,  0],
                         [0,  0,  1, dt,  0,  0],
                         [0,  0,  0,  1,  0,  0],
                         [0,  0,  0,  0,  1, dt],
                         [0,  0,  0,  0,  0,  1]]

        observation_covariance = np.eye(6,6)
        observation_covariance[1,1] = 0.01
        observation_covariance[3,3] = 0.01
        observation_covariance[5,5] = 0.01

        observation_matrix = np.eye(6,6)

        # Construct flights
        self.flights = FlightSet()
        for i_flight in range(self.num_flights):
            cluster_id = self.cluster_ids[i_flight]
            s = self.flight_start_idx[i_flight]
            e = self.flight_end_idx[i_flight]
            flight_start_sec = self.cortex_global_sample_timestamps_sec[s]
            flight_end_sec = self.cortex_global_sample_timestamps_sec[e]
            #kalman_estimate = kalman_filter_orientation(self.measured_hd_state[s:e,:], transition_matrix, observation_matrix, observation_covariance)
            #kalman_orientation = kalman_estimate[:,np.array([0,2,4])]
            #kalman_angular_velocity = kalman_estimate[:,np.array([1,3,5])]
            self.cortex_hd = np.full([e-s,3],np.nan)
            self.cortex_hd[:,0] = self.raw_azimuth[s:e]
            self.cortex_hd[:,1] = self.raw_pitch[s:e]
            self.cortex_hd[:,2] = self.raw_roll[s:e]
            angular_velocity = np.full([e-s,3],np.nan)
            flight = Flight(s, 
                            e, 
                            flight_start_sec,
                            flight_end_sec,
                            cluster_id, 
                            self.bat_pos[s:e,:], 
                            self.cortex_hd, 
                            self.bat_pos[s:e,:], 
                            self.cortex_hd, 
                            angular_velocity)
            self.flights.add_flight(flight)


class Flight():
    # Initialize class variables
    def __init__(self, 
                 timebin_start_idx,
                 timebin_end_idx,  
                 start_time,
                 end_time,
                 cluster_id,
                 raw_position,
                 raw_orientation,
                 filtered_position,
                 filtered_orientation,
                 filtered_angular_velocity):
        self.timebin_start_idx = timebin_start_idx
        self.timebin_end_idx = timebin_end_idx
        self.start_time = start_time
        self.end_time = end_time
        self.cluster_id = cluster_id
        self.raw_position = raw_position
        self.raw_orientation = raw_orientation
        self.filtered_position = filtered_position
        self.filtered_orientation = filtered_orientation
        self.filtered_angular_velocity = filtered_angular_velocity

        first_non_nan_idx = np.where(~np.isnan(filtered_position))[0][0]
        last_non_nan_idx = np.where(~np.isnan(filtered_position))[0][-1]

        self.start_pos = self.filtered_position[first_non_nan_idx,:]
        self.end_pos = self.filtered_position[last_non_nan_idx,:]

        
    # Repr function that displays cluster_id
    def __repr__(self):
        return f'{self.cluster_id} | {(self.timebin_end_idx - self.timebin_start_idx)/120}'

class SpikeData():
    def __init__(self, data_path, date, bat_id):
        self.data_path = data_path
        self.date = date
        self.bat_id = bat_id
        #self.ephys_data = hdf5storage.loadmat(os.path.join(self.data_path, f'ephys/{self.bat_id}_{self.date}_spikes_python.mat'))['spikeDataStruct'][0]

        data = hdf5storage.loadmat(os.path.join(data_path, f'ephys/{bat_id}_{date}_spikes_python.mat'))
        self.ephys_data = data['spikeDataStruct'][0]
        self.ttl_timestamps_sec = data['local_ttl_timestamps_sec'].T.flatten()
        self.num_ttls = len(self.ttl_timestamps_sec)
        self.session_start_time = 0
        self.session_end_time = self.num_ttls * 3

        self.num_probes = len(self.ephys_data)

        self.num_cells_per_probe = []
        self.num_cells = 0

        self.single_units = {} # Single units probes x cells
        self.all_single_units = {} # Single units all probes combined
        idx = 0
        for i_probe in range(self.num_probes):
            self.single_units[i_probe] = {}
            if(self.ephys_data[i_probe].shape[1] == 0): # No cells will result in self.ephys_data[i_probe].shape = (1,0)
                numCells = 0
            else:
                numCells = len(self.ephys_data[i_probe])
            self.num_cells_per_probe.append(numCells)
            self.num_cells += numCells
            for i_cell in range(numCells):
                self.single_units[i_probe][i_cell] = {}
                spikeTimes_sec = self.ephys_data[i_probe][i_cell]['spikeTimes_usec'][0].flatten()/1e6
                spikeTimes_sec = spikeTimes_sec[spikeTimes_sec > 0]
                spikeTimes_sec = spikeTimes_sec[spikeTimes_sec < self.session_end_time]

                # Remove double counted spikes
                # See https://github.com/MouseLand/Kilosort/issues/29
                isi = np.diff(spikeTimes_sec)            
                spikeTimes_sec = np.delete(spikeTimes_sec, np.where(isi< 10/30000)[0] + 1)

                
                self.single_units[i_probe][i_cell]['depth'] = self.ephys_data[i_probe][i_cell]['depth'][0][0]
                self.single_units[i_probe][i_cell]['cluster_id'] = self.ephys_data[i_probe][i_cell]['cluster_id'][0][0]
                self.single_units[i_probe][i_cell]['spikeTimes_sec'] = spikeTimes_sec
                self.single_units[i_probe][i_cell]['category'] = None
                self.single_units[i_probe][i_cell]['probe_id'] = i_probe
                self.single_units[i_probe][i_cell]['bat_id'] = self.bat_id
                self.single_units[i_probe][i_cell]['date'] = self.date
                self.single_units[i_probe][i_cell]['contam_pct'] = self.ephys_data[i_probe][i_cell]['ContamPct'][0][0]
                self.all_single_units[idx] = self.single_units[i_probe][i_cell]
                idx += 1

    def estimate_firing_rates(self, timebin_size_sec = 0.01, sigma_sec = 0.05, smooth=True, zscore=True, probes_incl = None):
        """
        Bin spikes into spk_count_timebins, then convolve with gaussian kernel to obtain estimated firing rate.

        Parameters
        ----------
        timebins
            Timebins for calculating spike counts. E.g) 10 ms
        """
        
        timebins = np.arange(self.session_start_time, self.session_end_time, timebin_size_sec)


        
        # Estimate firing rate for all single units
        idx = 0
        if(probes_incl is None):
            probes_incl = range(self.num_probes)

        # Count number of cells in incl probes
        num_incl_cells = 0
        for i_probe in probes_incl:
            num_incl_cells += self.num_cells_per_probe[i_probe]
        spike_rates = np.full([num_incl_cells, len(timebins)-1], np.nan)
        for i_probe in probes_incl:
            for i_cell in range(self.num_cells_per_probe[i_probe]):
                unit = self.single_units[i_probe][i_cell]

                # Estimate firing rate for single unit
                timebin_size = timebins[1] - timebins[0]
                binnedSpikeCount = np.histogram(unit['spikeTimes_sec'], timebins)[0]
                firing_rate = binnedSpikeCount / timebin_size

                # Convolve instantaneous firing rate with gaussian kernel
                if(smooth):
                    firing_rate = gaussian_filter(firing_rate, sigma=int(sigma_sec / timebin_size)) # 5 sigma at 10ms time bins = 50 ms sigma. See Gardner et al. (2022)
                if(zscore):
                    firing_rate = scipy.stats.zscore(firing_rate)

                spike_rates[idx,:] = firing_rate
                idx += 1
        return spike_rates, timebins
    
    def filter_units(self, depths):
        """
        Filter single units by depth. Returns a logical array for indexing spike_rates from estimate_firing_rates

        Parameters
        ----------
        depth : 2D np.array | (num_probes, 2)
            Each row contains the min and max depth for each probe

        Returns
        -------
        Logical array | (self.num_cells,)
            True if unit is within depth range of its probe
        """

        logical_arr = [self.single_units[i_probe]['depth'] > depths[i_probe,0] and self.single_units[i_probe]['depth'] < depths[i_probe,1] for i_probe in range(self.num_probes)]
        return np.concatenate(logical_arr)
        
class FlightSet():
    # Object containing all flights for a given bat and date. Allow for grouping of flights by cluster_id, start bat_pos, or end bat_pos
    def __init__(self):
        self.flights_by_cluster = {}
        self.flights_by_start_pos = {}
        self.flights_by_end_pos = {}
        
        self.num_flights = 0

    def add_flight(self, flight):
        if(flight.cluster_id not in self.flights_by_cluster):
            self.flights_by_cluster[flight.cluster_id] = []
        if(flight.start_pos[0] not in self.flights_by_start_pos):
            self.flights_by_start_pos[flight.start_pos[0]] = []
        if(flight.end_pos[0] not in self.flights_by_end_pos):
            self.flights_by_end_pos[flight.end_pos[0]] = []

        self.flights_by_cluster[flight.cluster_id].append(flight)
        self.flights_by_start_pos[flight.start_pos[0]].append(flight)
        self.flights_by_end_pos[flight.end_pos[0]].append(flight)
        self.num_flights += 1

    def get_flights_by_cluster(self, cluster_ids):
        if(type(cluster_ids) == int):
            return self.flights_by_cluster[cluster_ids]
        else:
            out = []
            for cluster_id in cluster_ids:
                out += self.flights_by_cluster[cluster_id]
            return out
    
    def get_flights_by_start_pos(self, start_pos):
        return self.flights_by_start_pos[start_pos]
    
    def get_flights_by_end_pos(self, end_pos):
        return self.flights_by_end_pos[end_pos]
    
    def __repr__(self) -> str:
        s1 = f'Flights object containing {self.num_flights} flights. Use get_flights_by_cluster, get_flights_by_start_pos, or get_flights_by_end_pos to access flights.'

        # Cluster IDs sorted by number of flights
        ranked_clusters = sorted(self.flights_by_cluster.keys(), key= lambda x:len(self.flights_by_cluster[x]), reverse=True)

        s2 = f'Number of flights by cluster in descending order: {[(cluster_id, len(self.flights_by_cluster[cluster_id])) for cluster_id in ranked_clusters]}'
        return s1 + '\n' + s2