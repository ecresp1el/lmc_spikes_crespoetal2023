import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class UnitDataAnalysis:
    def __init__(self, unit_data):
        """
        Initializes the UnitDataAnalysis class with a unit data DataFrame.

        Parameters:
        unit_data (pd.DataFrame): The unit data DataFrame obtained from the get_unit_table method.
        """
        if not isinstance(unit_data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        
        self.unit_data = unit_data

    def convert_samples2seconds(self):
        """
        This method converts sample times to seconds using the sampling frequency.
        It operates on the `self.unit_data` attribute of the class, adding new columns
        with the converted times.
        """
        
        # Check that self.unit_data is properly initialized
        if self.unit_data is None or not isinstance(self.unit_data, pd.DataFrame):
            raise ValueError("self.unit_data is not properly initialized")
        
        
        #perform the conversion and add the new columns to the DataFrame 
        self.unit_data['SpikeTimes_in_seconds'] = self.unit_data['SpikeTimes_all'] / self.unit_data['Sampling_Frequency']
        self.unit_data['ChemStimTime_in_seconds'] = self.unit_data['ChemStimTime_samples'] / self.unit_data['Sampling_Frequency']
        
    def calculate_and_plot_firing_rate_matrix(self, bin_size=1.0):
        # Step 1: Determine the Bin Edges
        total_duration = self.unit_data['Recording_Duration'].max()  # Assuming 'Recording_Duration' is in seconds
        bin_edges = np.arange(0, total_duration, bin_size)
        
        # Step 2 & 3: Calculate Spike Counts and Firing Rates
        num_units = len(self.unit_data)
        firing_rate_matrix = np.full((num_units, len(bin_edges)-1), np.nan)  # Initialize with NaN values
        
        for i, (unit_id, unit_data) in enumerate(self.unit_data.iterrows()):
            spike_times = unit_data['SpikeTimes_in_seconds']
            recording_duration = unit_data['Recording_Duration']
            
            # Adjust the bin edges for the current unit based on its recording duration
            unit_bin_edges = np.arange(0, recording_duration, bin_size)
            
            spike_counts, _ = np.histogram(spike_times, bins=unit_bin_edges)
            firing_rate_matrix[i, :len(unit_bin_edges)-1] = spike_counts / bin_size
        
        # Step 4: Normalize the Firing Rates
        max_firing_rates = np.nanmax(firing_rate_matrix, axis=1)
        normalized_firing_rate_matrix = firing_rate_matrix / max_firing_rates[:, np.newaxis]
        
        # Step 5: Plot the Matrix
        plt.imshow(normalized_firing_rate_matrix, aspect='auto', cmap='RdPu', vmax=1)
        plt.colorbar(label='Normalized Firing Rate')
        plt.xlabel('Time Bin')
        plt.ylabel('Unit')
        plt.title('Normalized Firing Rate Matrix')
        plt.show()

        