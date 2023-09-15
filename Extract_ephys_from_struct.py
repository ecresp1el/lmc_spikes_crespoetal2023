import mat73 
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import pandas as pd
import xarray as xr

class ExtractEphysData:
    """
    A class to facilitate the extraction of electrophysiological data from .mat files.

    Args:
        matfile_directory (str): The directory where the .mat file is located.

    Attributes:
        mat (dict): The loaded .mat file represented as a dictionary.
        all_data (dict): A dictionary containing all data from the .mat file.
        group_names (list of str): A list of group names extracted from the all_data attribute.
        recordings (dict): A dictionary mapping group names to lists of recording names.
        unit_id_map (dict): A dictionary to store unit ID mappings (initialized as empty).
    """
    
    def __init__(self, matfile_directory):
        """
        Initializes the ExtractEphysData class by loading a .mat file using the mat73 library.

        Args:
            matfile_directory (str): The directory path to the .mat file.
        """
        mat = mat73.loadmat(matfile_directory, use_attrdict=True)
        self.all_data = mat['all_data'] 
        self.group_names = list(self.all_data.keys()) # get the group names from the all_data attribute and store them in the group_names attribute
        self.recordings = {group: list(recordings.keys()) for group, recordings in self.all_data.items()} # get the recording names for each group and store them in the recordings attribute
        self.generate_unit_id_map()
        self.trial_intensity_dataframes = {}  # Initialize an empty dictionary to store DataFrames

        # Perform the dict keys check early on and store the results as an attribute 
        # results of the check_dict_keys method are stored in the self.dict_keys_check_results attribute, 
        # which you can reference at any point in your analysis to know which unit IDs passed the check.
        #self.stimulus_tables = {}  # Initialize stimulus_tables as an empty dictionary
        #self.construct_stimulus_table()  # Construct stimulus tables for all recordings at initialization


    def get_group_names(self):
        """
        Retrieves all group names available in the data.
        
        Returns:
            list: A list of group names.
        """
        return self.group_names

    def get_recording_names(self, group_name):
        """
        Retrieves all recording names available for a specified group.
        
        Args:
            group_name (str): The name of the group to get the recording names for.
        
        Returns:
            list of str: A list containing all recording names for the specified group. 
                         Returns an empty list if the group name is not found.
        """
        # get method is a dictionary method that returns the value for a given key if the key is present in the dictionary. 
        # If the key is not present, the method returns a default value instead of raising a KeyError exception.
        return self.recordings.get(group_name, []) 
    
    def generate_unit_id_map(self):
        """
        Generates a unique unit ID map that associates a unique identifier with each cell ID based on the group name, 
        recording name, and cell ID. This ensures no conflicts between cell IDs across different recordings or groups.

        The unique unit ID is created by generating a hex hash of the concatenated group name, recording name, and cell ID.

        This method should be called during initialization to create the unit ID map for further analysis.
        """
        self.unit_id_map = {}

        for group_name in self.group_names:
            for recording_name in self.recordings[group_name]:
                for cell_id in self.all_data[group_name][recording_name].keys():
                    # Create a unique unit ID by generating a hex hash of the concatenated strings
                    unique_unit_id = hashlib.sha256(f"{group_name}_{recording_name}_{cell_id}".encode()).hexdigest()
                    self.unit_id_map[unique_unit_id] = {
                        "group": group_name,
                        "recording": recording_name,
                        "cell_id": cell_id,
                        "path": [group_name, recording_name, cell_id],  # Storing the path to the data
                    }
                    
    def get_unit_data(self, unique_unit_id):
        """
        Retrieves the data associated with a specific unique unit ID.

        Args:
            unique_unit_id (str): The unique identifier for a unit.

        Returns:
            dict: The data associated with the unit, or None if the unit ID is not found.
        """
        unit_info = self.unit_id_map.get(unique_unit_id)
        if unit_info:
            # Using the path to get the data from the all_data attribute
            data = self.all_data
            for key in unit_info["path"]:
                data = data.get(key)
                if data is None:
                    return None
            return data 
        
    def get_metric(self, unique_unit_id, metric_name):
        """
        Retrieves a specific metric for a unit.

        Args:
            unique_unit_id (str): The unique identifier for a unit.
            metric_name (str): The name of the metric to retrieve.

        Returns:
            The requested metric, or None if the unit ID or metric name is not found.
            
        Raises:
            ValueError: If the metric name is not valid.
        """
        valid_metric_names = {
            'Amplitude', 'Cell_Type', 'ChemStimTime_note', 'ChemStimTime_s', 
            'ChemStimTime_samples', 'FR_time_cutoff_after_stim_ms', 'FRs_baseline', 
            'FRs_baseline_vec', 'FRs_stim', 'FanoFactor_baseline', 'FanoFactor_stim', 
            'FirstSpikeLatency', 'FirstSpikeLatency_Reliability', 'FirstSpikeLatency_pdf_x', 
            'FirstSpikeLatency_pdf_y', 'FirstSpikeLatency_perTrial', 'Header', 'ISI_baseline_CV', 
            'ISI_baseline_vec', 'ISI_pdf_peak_xy', 'ISI_pdf_x', 'ISI_pdf_y', 
            'ISI_violations_percent', 'IsSingleUnit', 'MeanFR_baseline', 'MeanFR_inst_baseline', 
            'MeanFR_inst_stim', 'MeanFR_stim', 'MeanFR_total', 'Mean_Waveform', 'ModulationIndex', 
            'Normalized_Template_Waveform', 'PSTHs_conv', 'PSTHs_raw', 'Peak1ToTrough_ratio', 
            'Peak2ToTrough_ratio', 'PeakEvokedFR', 'PeakEvokedFR_Latency', 'PeakToPeak_ratio', 
            'Post', 'Pre', 'Recording_Duration', 'Sampling_Frequency', 'SpikeHalfWidth', 
            'SpikeTimes_all', 'SpikeTimes_baseline', 'SpikeTimes_stim', 'SpikeTimes_trials', 
            'SpikeTrains_baseline', 'SpikeTrains_baseline_ms', 'SpikeTrains_for_PSTHs', 
            'SpikeTrains_stim', 'SpikeTrains_stim_ms', 'SpikeTrains_trials', 
            'SpikeTrains_trials_ms', 'StimProb', 'StimResponsivity', 'Stim_Intensity', 
            'Stim_Offsets_samples', 'Stim_Onsets_samples', 'Template_Channel', 
            'Template_Channel_Position', 'TroughToPeak_duration', 'UnNormalized_Template_Waveform', 
            'peak1_normalized_amplitude'
        }
        
        if metric_name not in valid_metric_names:
            raise ValueError(f"Invalid metric name: '{metric_name}'. Must be one of {valid_metric_names}")

        unit_data = self.get_unit_data(unique_unit_id)
        if unit_data:
            return unit_data.get(metric_name)
        
    def iterate_unit_ids(self, func):
        """
        Iterates over all unit IDs and applies a function to each unit's data.

        Args:
            func (callable): A function to apply to each unit's data. 
                             The function should take a unit ID and a unit data dictionary as parameters.

        Returns:
            A dictionary with unit IDs as keys and the results of applying the function as values.
        """
        results = {}
        for unit_id in self.unit_id_map:
            unit_data = self.get_unit_data(unit_id)
            results[unit_id] = func(unit_id, unit_data)
        return results

    @staticmethod
    def get_amplitude(unit_id, unit_data):
        """
        Retrieves the 'Amplitude' metric from the unit data dictionary for a given unit ID.

        This method is designed to be used as a helper function with the iterate_unit_ids method, 
        to facilitate the extraction of amplitude data for each unit in a batch operation.

        Args:
            unit_id (str): The unique identifier for a unit. This ID is used to track the unit 
                        during the iteration process, but is not used within this method.
            unit_data (dict): A dictionary containing the data for a single unit. This dictionary 
                            is expected to have a key 'Amplitude' whose value is to be retrieved.

        Returns:
            float or None: The amplitude value associated with the unit. If the 'Amplitude' key 
                        does not exist in the unit_data dictionary, None is returned.

        Raises:
            TypeError: If unit_data is not a dictionary.

        Example:
            >>> unit_data = {'Amplitude': 1.23, 'OtherMetric': 4.56}
            >>> ExtractEphysData.get_amplitude('some_unit_id', unit_data)
            1.23
        """
        if not isinstance(unit_data, dict):
            raise TypeError("unit_data must be a dictionary.")
        
        return unit_data.get('Amplitude')

    def calculate_average_amplitude(self):
        """
        Calculate the average amplitude for all units in the data.

        This method iterates through all unit IDs, extracts the amplitude metric for each unit, 
        and calculates the average amplitude across all units.

        Returns:
            dict: A dictionary where keys are unit IDs (str), and values are the corresponding
            average amplitudes (float). If a unit has no amplitude data or is not found in the data,
            it will not be included in the result.

        Examples:
        >>> eed = ExtractEphysData('path/to/matfile.mat')
        >>> average_amplitudes = eed.calculate_average_amplitude()
        >>> print(average_amplitudes)
        {'unit_id1': 1.23, 'unit_id2': 2.45, ...}

        Notes:
        - This method relies on the 'Amplitude' metric being present in the unit data dictionary.
        - The result may contain fewer entries if some units lack amplitude data or do not exist in the data.
        """
        average_amplitudes = {} # initialize an empty dictionary to store the average amplitudes 

        for unit_id in self.unit_id_map: # iterate over all unit IDs using the unit_id_map attribute
            unit_data = self.get_unit_data(unit_id) # retrieve the unit data for the current unit ID with the get_unit_data method 
            if unit_data: # check if the unit data is not None
                amplitude = unit_data.get('Amplitude') # retrieve the amplitude metric from the unit data dictionary with the get method
                if amplitude is not None: # check if the amplitude is not None 
                    average_amplitudes[unit_id] = amplitude # add the unit ID and amplitude to the average_amplitudes dictionary

        return average_amplitudes
    
    def get_stimulation_data(self):
        """
        Retrieve stimulation data for all units in the data.

        This method iterates through all unit IDs, extracts the 'Pre' and 'Post' stimulation data for each unit, 
        and combines them into a dictionary where keys are unit IDs (str), and values are dictionaries containing 
        the 'Pre' and 'Post' stimulation data. If a unit has no stimulation data or is not found in the data, it will 
        not be included in the result.

        Returns:
            dict: A dictionary where keys are unit IDs (str), and values are dictionaries containing 'Pre' and 'Post' 
            stimulation data. Each 'Pre' and 'Post' dictionary contains keys 'Intensity' and 'SpikeTrain' whose 
            values are the corresponding data.

        Examples:
        >>> eed = ExtractEphysData('path/to/matfile.mat')
        >>> stimulation_data = eed.get_stimulation_data()
        >>> print(stimulation_data)
        {'unit_id1': {'Pre': {'Intensity': [...], 'SpikeTrain': [...]}, 'Post': {'Intensity': [...], 'SpikeTrain': [...]}},
         'unit_id2': {'Pre': {'Intensity': [...], 'SpikeTrain': [...]}, 'Post': {'Intensity': [...], 'SpikeTrain': [...]}},
         ...}

        Notes:
        - This method relies on the 'Pre' and 'Post' stimulation data being present in the unit data dictionary.
        - The result may contain fewer entries if some units lack stimulation data or do not exist in the data.
        """
        stimulation_data = {} # initialize an empty dictionary to store the stimulation data 

        for unit_id in self.unit_id_map: # iterate over all unit IDs using the unit_id_map attribute
            unit_data = self.get_unit_data(unit_id) # retrieve the unit data for the current unit ID with the get_unit_data method 
            if unit_data: # check if the unit data is not None
                pre_intensity = unit_data.get('Pre', {}).get('Stim_Intensity') # retrieve the 'Pre' intensity data
                post_intensity = unit_data.get('Post', {}).get('Stim_Intensity') # retrieve the 'Post' intensity data
                pre_spike_trains = unit_data.get('Pre', {}).get('SpikeTrains_trials') # retrieve the 'Pre' spike train data
                post_spike_trains = unit_data.get('Post', {}).get('SpikeTrains_trials') # retrieve the 'Post' spike train data

                # Create a dictionary to store the 'Pre' and 'Post' data
                unit_stim_data = {}
                if pre_intensity is not None:
                    unit_stim_data['Pre'] = {'Intensity': pre_intensity, 'SpikeTrain': pre_spike_trains}
                if post_intensity is not None:
                    unit_stim_data['Post'] = {'Intensity': post_intensity, 'SpikeTrain': post_spike_trains}

                # Add the unit ID and combined 'Pre' and 'Post' data to the stimulation_data dictionary
                stimulation_data[unit_id] = unit_stim_data

        return stimulation_data

    def reorganize_stimulation_data(self, stimulation_data):
        """
        Reorganize stimulation data into a different format.

        This method takes the stimulation data in the format generated by 
        'get_stimulation_data' and reorganizes it into a dictionary where the keys 
        are the unique unit IDs. Each unit's data contains keys pointing to 
        'Intensity', 'SpikeTrain', and 'Epoch' with values being the combined 'Pre' 
        and 'Post' data. The 'Epoch' key stores information about whether the data 
        belongs to the 'Pre' or 'Post' stimulation period.

        Parameters
        ----------
        stimulation_data : dict
            The stimulation data obtained from 'get_stimulation_data', which contains 
            'Pre' and 'Post' stimulation data for different units.

        Returns
        -------
        dict
            A dictionary where keys are unique unit IDs, and each unit's data 
            contains 'Intensity', 'SpikeTrain', and 'Epoch' keys with values being 
            the combined 'Pre' and 'Post' data.

        Examples
        --------
        >>> eed = ExtractEphysData('path/to/matfile.mat')
        >>> stimulation_data = eed.get_stimulation_data()
        >>> reorganized_data = eed.reorganize_stimulation_data(stimulation_data)
        >>> print(reorganized_data)
        {'unit_id1': {'Intensity': [...], 'SpikeTrain': [...], 'Epoch': [...]}, 
        'unit_id2': {'Intensity': [...], 'SpikeTrain': [...], 'Epoch': [...]}}
        """
        reorganized_data = {}

        for unit_id, unit_stim_data in stimulation_data.items():
            pre_intensity = unit_stim_data['Pre']['Intensity']
            post_intensity = unit_stim_data['Post']['Intensity']
            pre_spike_trains = unit_stim_data['Pre']['SpikeTrain']
            post_spike_trains = unit_stim_data['Post']['SpikeTrain']

            combined_data = {
                'Intensity': np.concatenate((pre_intensity, post_intensity)),
                'SpikeTrain': np.concatenate((pre_spike_trains, post_spike_trains)),
                'Epoch': np.concatenate((['Pre'] * len(pre_intensity), ['Post'] * len(post_intensity)))  # new line
            }


            reorganized_data[unit_id] = combined_data

        return reorganized_data

    def create_trial_intensity_dataframe(self, reorganized_data):
        """
        Create DataFrames with trial IDs and mapped intensity labels for each unit's data.

        This method takes a dictionary containing reorganized stimulation data for 
        multiple units and constructs DataFrames with trial IDs, corresponding 
        intensity labels, and epoch information ('Pre' or 'Post') for each unit. The 
        trial IDs are generated as 'Trial_1', 'Trial_2', ..., 'Trial_N', where N is 
        the number of trials for each unit.

        The intensity labels are mapped as follows:
            - 1 corresponds to 'Zero'
            - 2 corresponds to 'Low'
            - 3 corresponds to 'Mid'
            - 4 corresponds to 'Max'

        Parameters
        ----------
        reorganized_data : dict
            A dictionary where keys are unique unit IDs, and each unit's data 
            contains 'Intensity' and 'SpikeTrain' keys with values being the 
            combined 'Pre' and 'Post' data. Also contains 'Epoch' key with 'Pre' or 
            'Post' labels for each trial.

        Returns
        -------
        None

        Examples
        --------
        >>> eed = ExtractEphysData('path/to/matfile.mat')
        >>> reorganized_data = {'unit_id1': {'Intensity': [1, 2, 3], 'SpikeTrain': [...], 'Epoch': ['Pre', 'Post', 'Pre']},
                            'unit_id2': {'Intensity': [2, 3, 4], 'SpikeTrain': [...], 'Epoch': ['Pre', 'Post', 'Post']}}
        >>> eed.create_trial_intensity_dataframe(reorganized_data)
        >>> print(eed.trial_intensity_dataframes)
        {'unit_id1': DataFrame with Trial_ID, Intensity, and Epoch columns,
        'unit_id2': DataFrame with Trial_ID, Intensity, and Epoch columns}

        Notes
        -----
        - The method creates a separate DataFrame for each unique unit ID and stores 
        them as attributes with the unit ID as the key in the 
        'trial_intensity_dataframes' dictionary.
        """

        for unit_id, unit_data in reorganized_data.items():
            # Extract Intensity data from the unit's data
            intensity_data = unit_data['Intensity']
            epoch_data = unit_data['Epoch']  # new line

            # Create trial IDs based on the length of Intensity data
            trial_ids = [f'Trial_{i + 1}' for i in range(len(intensity_data))]

            # Map Intensity values to labels
            intensity_labels = {
                1: 'Zero',
                2: 'Low',
                3: 'Mid',
                4: 'Max'
            }
            intensity_data_labels = [intensity_labels[i] for i in intensity_data]

            # Create a DataFrame
            df = pd.DataFrame({'Trial_ID': trial_ids, 'Intensity': intensity_data_labels, 'Epoch': epoch_data})  # modified line

            # Store the DataFrame in the dictionary with the unit ID as the key
            self.trial_intensity_dataframes[unit_id] = df

    def create_xarray(self, converted_data):
        """
        Create xarrays for all units using the converted_data and trial_intensity_dataframes attributes.

        Args:
            converted_data (dict): A dictionary containing the reorganized stimulation data for all units.
            trial_intensity_dataframes (dict): A dictionary containing trial intensity dataframes for all units.

        Returns:
            dict: A dictionary where keys are unit IDs (str), and values are xarray DataArrays containing the data.

        Notes:
        - This method assumes that the 'Sample' dimension corresponds to the number of columns in the SpikeTrains.
        """
        xarrays = {} # Initialize an empty dictionary to store xarrays
    
        
        for unit_id, data in converted_data.items():
            intensity = data['Intensity'].astype(int)
            spike_train = data['SpikeTrain']
            
            # Fetch the trial IDs for the current unit
            trial_ids = self.trial_intensity_dataframes[unit_id]['Trial_ID']
            
            # Debug print statements to check the alignment
            print(f"Unit ID: {unit_id}")
            print(f"Spike train shape: {spike_train.shape}")
            print(f"Trial_IDs from dataframes: {trial_ids}")
            
            # Create a DataArray with 'Sample' as the dimension (number of columns in SpikeTrains)
            xarray = xr.DataArray(
                spike_train,
                dims=['Trial_ID', 'Sample'],
                coords={'Trial_ID': trial_ids}
            )
            
            # Assign the 'Intensity' values as an attribute
            xarray.attrs['Intensity'] = intensity.astype(int) # Convert to integer type here
            
            xarrays[unit_id] = xarray # Store the xarray in the dictionary with the unit ID as the key
        
        return xarrays         

    def get_psths(self, unit_ids=None):
        """
        Retrieve PSTH data for specified units or all units in the data.

        This method iterates through the specified unit IDs (if provided) or all unit IDs, 
        extracts the 'Pre' and 'Post' PSTH data for each unit, and combines them into a dictionary 
        where keys are unit IDs (str), and values are dictionaries containing the 'Pre' and 'Post' PSTH data. 
        If a unit has no PSTH data or is not found in the data, it will not be included in the result.

        Args:
            unit_ids (list or None): A list of specific unit IDs to retrieve PSTH data for. If None, PSTH data
                for all units will be retrieved. Default is None.

        Returns:
            dict: A dictionary where keys are unit IDs (str), and values are dictionaries containing 'Pre' and 'Post' 
            PSTH data. Each 'Pre' and 'Post' dictionary contains keys 'PSTH_raw' whose values are the corresponding data.

        Examples:
        >>> eed = ExtractEphysData('path/to/matfile.mat')
        >>> psth_data = eed.get_psths(unit_ids=['unit_id1', 'unit_id2'])
        >>> print(psth_data)
        {'unit_id1': {'Pre': {'PSTH_raw': [...]}, 'Post': {'PSTH_raw': [...]}},
        'unit_id2': {'Pre': {'PSTH_raw': [...]}, 'Post': {'PSTH_raw': [...]}}}

        Notes:
        - This method relies on the 'Pre' and 'Post' PSTH data being present in the unit data dictionary.
        - The result may contain fewer entries if some units lack PSTH data or do not exist in the data.
        """
        psth_data = {}  # initialize an empty dictionary to store the PSTH data

        # Determine the list of unit IDs to process based on the input or all unit IDs
        if unit_ids is None:
            unit_ids_to_process = list(self.unit_id_map.keys())
        else:
            unit_ids_to_process = unit_ids

        for unit_id in unit_ids_to_process:  # iterate over the specified or all unit IDs
            unit_data = self.get_unit_data(unit_id)  # retrieve the unit data for the current unit ID with the get_unit_data method
            if unit_data:  # check if the unit data is not None
                pre_psths = unit_data.get('Pre', {}).get('PSTHs_raw')  # retrieve the 'Pre' PSTH data
                post_psths = unit_data.get('Post', {}).get('PSTHs_raw')  # retrieve the 'Post' PSTH data

                # Create a dictionary to store the 'Pre' and 'Post' PSTH data
                unit_psth_data = {}
                if pre_psths is not None:
                    unit_psth_data['Pre'] = {'PSTH_raw': pre_psths}
                if post_psths is not None:
                    unit_psth_data['Post'] = {'PSTH_raw': post_psths}

                # Add the unit ID and combined 'Pre' and 'Post' PSTH data to the psth_data dictionary
                psth_data[unit_id] = unit_psth_data

        return psth_data

    def query_xarrays(self, xarrays, unit_id, intensity=None, epoch=None):
        """
        Query xarrays based on specified criteria such as intensity and epoch.

        Parameters:
        - xarrays (dict): A dictionary where keys are unit IDs (str), and values are xarray DataArrays containing the data.
        - unit_id (str): The ID of the unit to be queried.
        - intensity (str, optional): The stimulation intensity to be used for filtering. Possible values are 'Zero', 'Low', 'Mid', 'Max'. Defaults to None.
        - epoch (str, optional): The epoch to be used for filtering. Possible values are 'Pre', 'Post'. Defaults to None.

        Returns:
        - xarray.DataArray: An xarray DataArray containing the queried data.
        """

        # Step 1: Access the relevant DataFrame using the unit_id
        df = self.trial_intensity_dataframes[unit_id]

        # Step 2: Build the query string based on provided parameters
        query_str = ' & '.join([f'{col} == "{val}"' for col, val in zip(['Intensity', 'Epoch'], [intensity, epoch]) if val])

        # Step 3: Filter the DataFrame using the query string to get relevant Trial_IDs
        filtered_df = df.query(query_str) if query_str else df

        # Step 4: Get the list of relevant Trial_IDs
        trial_ids = filtered_df['Trial_ID'].tolist()

        # Step 5: Access and filter the xarray DataArray using the Trial_IDs
        xarray_data = xarrays[unit_id]
        filtered_xarray_data = xarray_data.sel(Trial_ID=trial_ids)
        
        # Step 6: Update the intensity attribute to reflect the filtering
        if intensity:
            intensity_mapping = {'Zero': 1, 'Low': 2, 'Mid': 3, 'Max': 4}
            filtered_xarray_data.attrs['Intensity'] = np.full_like(filtered_xarray_data.attrs['Intensity'], intensity_mapping[intensity])

        return filtered_xarray_data

    
    def query_units(self, xarrays, unit_ids, intensity=None, epoch=None):
        """
        Batch query xarrays based on specified criteria such as intensity and epoch across multiple units.

        Parameters:
        - xarrays (dict): A dictionary where keys are unit IDs (str), and values are xarray DataArrays containing the data.
        - unit_ids (list of str): A list of unit IDs to be queried.
        - intensity (str, optional): The stimulation intensity to be used for filtering. Possible values are 'Zero', 'Low', 'Mid', 'Max'. Defaults to None.
        - epoch (str, optional): The epoch to be used for filtering. Possible values are 'Pre', 'Post'. Defaults to None.

        Returns:
        - dict: A dictionary where keys are unit IDs and values are the queried xarray DataArrays for each unit.

        Examples:
        ```
        # Example 1: Querying multiple units with the same criteria
        queried_data = EED.query_units(xarrays, ['unit1', 'unit2'], intensity='Mid', epoch='Post')
        ```
        """
        # Initializing an empty dictionary to store the queried data arrays for each unit
        queried_data = {}
        
        # Looping over each unit ID to query the data based on the specified criteria
        for unit_id in unit_ids:
            queried_data[unit_id] = self.query_xarrays(xarrays, unit_id, intensity=intensity, epoch=epoch)
        
        return queried_data
    
    def convert_sample_to_time(self, xarrays):
        new_xarrays = {}
        for unit_id, xarray in xarrays.items():
            # Get the sampling frequency for the unit
            sampling_freq = self.get_metric(unit_id, 'Sampling_Frequency')
            
            # Determine the new time points with 1 ms bins
            time_points = np.linspace(0, 1500, 1500)
            
            # Downsample the data to 1 ms bins
            bin_width_in_samples = int(sampling_freq / 1000)  # Number of samples in 1 ms
            downsampled_data = xarray.data.reshape((xarray.shape[0], -1, bin_width_in_samples)).sum(axis=2)

            # Create a new xarray with the downsampled data and new time points
            new_xarray = xr.DataArray(
                downsampled_data,
                dims=['Trial_ID', 'Time'],
                coords={'Trial_ID': xarray['Trial_ID'], 'Time': time_points}
            )
            
            # Copy the attributes from the original xarray
            new_xarray.attrs = xarray.attrs
            
            # Store the new xarray in the dictionary
            new_xarrays[unit_id] = new_xarray
        
        return new_xarrays


