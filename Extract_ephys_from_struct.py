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

        This method takes the stimulation data in the format generated by 'get_stimulation_data' and reorganizes it
        into a dictionary where the keys are the unique unit IDs. Each unit's data contains a key pointing to 'Intensity'
        and 'SpikeTrain' with values being the combined 'Pre' and 'Post' data.

        Args:
            stimulation_data (dict): The stimulation data obtained from 'get_stimulation_data'.

        Returns:
            dict: A dictionary where keys are unique unit IDs, and each unit's data contains 'Intensity' and 'SpikeTrain'
            keys with values being the combined 'Pre' and 'Post' data.

        Examples:
        >>> eed = ExtractEphysData('path/to/matfile.mat')
        >>> stimulation_data = eed.get_stimulation_data()
        >>> reorganized_data = eed.reorganize_stimulation_data(stimulation_data)
        >>> print(reorganized_data)
        {'unit_id1': {'Intensity': [...], 'SpikeTrain': [...]}, 'unit_id2': {'Intensity': [...], 'SpikeTrain': [...]}}
        """
        reorganized_data = {}

        for unit_id, unit_stim_data in stimulation_data.items():
            pre_intensity = unit_stim_data['Pre']['Intensity']
            post_intensity = unit_stim_data['Post']['Intensity']
            pre_spike_trains = unit_stim_data['Pre']['SpikeTrain']
            post_spike_trains = unit_stim_data['Post']['SpikeTrain']

            combined_data = {
                'Intensity': np.concatenate((pre_intensity, post_intensity)),
                'SpikeTrain': np.concatenate((pre_spike_trains, post_spike_trains))
            }

            reorganized_data[unit_id] = combined_data

        return reorganized_data

    def create_trial_intensity_dataframe(self, reorganized_data):
        """
        Create DataFrames with trial IDs and mapped intensity labels for each unit's data.

        This method takes a dictionary containing reorganized stimulation data for multiple units and
        constructs DataFrames with trial IDs and corresponding intensity labels for each unit. The trial IDs are
        generated as 'Trial_1', 'Trial_2', ..., 'Trial_N', where N is the number of trials for each unit.

        The intensity labels are mapped as follows:
            - 1 corresponds to 'Zero'
            - 2 corresponds to 'Low'
            - 3 corresponds to 'Mid'
            - 4 corresponds to 'Max'

        The resulting DataFrames are stored as attributes with the unit ID as the key in the
        'trial_intensity_dataframes' dictionary.

        Args:
            reorganized_data (dict): A dictionary where keys are unique unit IDs, and each unit's data
                contains 'Intensity' and 'SpikeTrain' keys with values being the combined 'Pre' and 'Post' data.

        Returns:
            None

        Examples:
        >>> eed = ExtractEphysData('path/to/matfile.mat')
        >>> reorganized_data = {'unit_id1': {'Intensity': [1, 2, 3], 'SpikeTrain': [...]},
        ...                     'unit_id2': {'Intensity': [2, 3, 4], 'SpikeTrain': [...]}}

        >>> eed.create_trial_intensity_dataframe(reorganized_data)
        >>> print(eed.trial_intensity_dataframes)
        {'unit_id1': DataFrame with Trial_ID and Intensity columns,
         'unit_id2': DataFrame with Trial_ID and Intensity columns}

        Notes:
        - The method creates a separate DataFrame for each unique unit ID and stores them as attributes
          with the unit ID as the key in the 'trial_intensity_dataframes' dictionary.
        """

        for unit_id, unit_data in reorganized_data.items():
            # Extract Intensity data from the unit's data
            intensity_data = unit_data['Intensity']

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
            df = pd.DataFrame({'Trial_ID': trial_ids, 'Intensity': intensity_data_labels})

            # Store the DataFrame in the dictionary with the unit ID as the key
            self.trial_intensity_dataframes[unit_id] = df

    def create_xarray(self, converted_data, trial_intensity_dataframes):
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
            intensity = data['Intensity']
            spike_train = data['SpikeTrain']
            
            # Create a DataArray with 'Sample' as the dimension (number of columns in SpikeTrains)
            xarray = xr.DataArray(
                spike_train,
                dims=['Trial_ID', 'Sample'],
                coords={'Trial_ID': trial_intensity_dataframes[unit_id]['Trial_ID']}
            )
            
            # Assign the 'Intensity' values as an attribute
            xarray.attrs['Intensity'] = intensity
            
            xarrays[unit_id] = xarray # Store the xarray in the dictionary with the unit ID as the key
        
        return xarrays            
                
            




    def get_cellid_names(self):
        """
        Returns a list of unique unit IDs for the current group and recording.
        Returns:
            list: A list of unique unit IDs for the current group and recording.
        """
        unit_ids = []
        group_name = self.group_name  # Accessing group_name attribute
        recording_name = self.recording_name  # Accessing recording_name attribute
        for cellid_name in self.all_data[group_name][recording_name].keys():
            # Creating a unique unit ID using group name, recording name, and cell ID
            unique_unit_id = hashlib.md5(f"{group_name}_{recording_name}_{cellid_name}".encode()).hexdigest()
            unit_ids.append(unique_unit_id)

            # Populate the unit_id_map attribute
            self.unit_id_map[unique_unit_id] = {
                "group_name": group_name,
                "recording_name": recording_name,
                "cellid_name": cellid_name
            }
        return unit_ids

    
    def load_matfiles_printdata(self):
        """
        Load the mat files and print the data structure including the total number of units per recording within each group.

        Returns:
            None
        """
        # Get the group names
        group_names = self.get_group_names()
        
        # Iterate through the group names and print them
        for group_name in group_names:
            print(f"Group name: {group_name}")
            
            # Get the recording names for the current group
            recording_names = self.get_recording_names(group_name)
            
            # Iterate through the recording names and print them along with the total number of units
            for recording_name in recording_names:
                # Get the unique unit IDs for the current group and recording
                unit_ids = self.get_cellid_names(group_name, recording_name)
                
                # Print the recording name and the total number of units in this recording
                print(f"  Recording name: {recording_name} - Total units: {len(unit_ids)}")    
    
    def get_unit_summary(self, unit_id):
        """
        Get a summary of the data available for a specific unit ID, including the group and recording it belongs to, 
        and whether it passed the dict keys check.

        Args:
            unit_id (str): The unique unit ID.

        Returns:
            dict: A dictionary containing the summary information for the unit ID.
        """
        # Type checking
         #check that the unit_id parameter is a string, raising a TypeError with a descriptive message if not.
        if not isinstance(unit_id, str):
            raise TypeError("unit_id must be a string") 
        
        # Existence checking
        # check unit_id exists in your data structure using the get_original_cellid method, raising a ValueError with a descriptive message if not.
        mapping = self.get_original_cellid(unit_id) 
        if mapping is None: #
            raise ValueError(f"Unit ID '{unit_id}' does not exist in the data structure")
        
        group, recording, original_cell_id = mapping # unpacking the tuple
        
        # Check if the unit passed the dict keys check
        # retrieve whether the unit passed the dict keys check from the dict_keys_check_results attribute
        passed_dict_keys_check = self.dict_keys_check_results.get(unit_id, False)
        
        # Create a summary dictionary
        summary = {
            "Unit ID": unit_id,
            "Group": group,
            "Recording": recording,
            "Original Cell ID": original_cell_id,
            "Passed Dict Keys Check": passed_dict_keys_check
        }
        
        return summary
               

    
    def extract_ephys_data(self, group_name, recording_name, unit_id):
        """
        Extracts the ephys data for a specific unit ID.

        Args:
            group_name (str): The name of the group.
            recording_name (str): The name of the recording.
            unit_id (str): The unique unit ID.

        Returns:
            dict: The ephys data for the specified unit ID.
        """
        # Decoding the unit ID to get the original cell ID name
        for cellid_name in self.mat['all_data'][group_name][recording_name].keys():
            if hashlib.md5(f"{group_name}_{recording_name}_{cellid_name}".encode()).hexdigest() == unit_id:
                # Extract the data using the original cell ID name
                data = self.mat['all_data'][group_name][recording_name][cellid_name]
                return data
        raise ValueError(f"Unit ID {unit_id} not found.")


    def get_all_unit_ids(self):
    
        """
        Iterates over all groups and recordings to print the unique unit IDs.

        Returns:
            None
        """
        all_unit_ids = [] # initialize an empty list to store all unit IDs 
        
        for group_name in self.get_group_names():
            for recording_name in self.get_recording_names(group_name):
                for cell_id in self.get_cellid_names(group_name, recording_name):
                    all_unit_ids.append(cell_id)
        
            
        return all_unit_ids
            
    def get_original_cellid(self, unit_id):
        """
        Retrieves the original cell ID corresponding to a unique unit ID.

        Args:
            unit_id (str): The unique unit ID.

        Returns:
            tuple: A tuple containing the group name, recording name, and original cell ID, or None if not found.
        """
        try:
            # Get all group names
            group_names = self.get_group_names()
            
            # Loop over all groups
            for group_name in group_names:

                # Get all recording names for the current group
                recording_names = self.get_recording_names(group_name)

                # Loop over all recordings in the current group
                for recording_name in recording_names:

                    # Get all cell ID names for the current recording
                    cellid_names = list(self.mat['all_data'][group_name][recording_name].keys())

                    # Loop over all cell ID names in the current recording
                    for cellid_name in cellid_names:

                        # Generate the unique unit ID for the current cell ID name
                        generated_unit_id = hashlib.md5(f"{group_name}_{recording_name}_{cellid_name}".encode()).hexdigest()

                        # Check if the generated unit ID matches the input unit ID
                        if generated_unit_id == unit_id:
                            return (group_name, recording_name, cellid_name)
            
            # If no match is found, return None
            return None
        except Exception as e: 
            print(f"An error occurred: {e}")
            return None
                    
    def get_data(self, unit_id=None, data_type='pre'):
        """
        Retrieves the specified type of data (pre or post) for a specified unit ID. 
        If no unit ID is provided, retrieves data for all unit IDs.

        Args:
            unit_id (str, optional): The unique unit ID. Defaults to None.
            data_type (str, optional): The type of data to retrieve ('pre' or 'post'). Defaults to 'pre'.

        Returns:
            dict: A dictionary with unit IDs as keys and data as values, or None if not found.
        """
        result = {}
        try:
            if unit_id:
                mapping = self.get_original_cellid(unit_id)
                if mapping:
                    group_name, recording_name, cellid_name = mapping
                    data = next((v for k, v in self.mat['all_data'][group_name][recording_name][cellid_name].items() if k.lower() == data_type.lower()), None)
                    if data is not None:
                        result[unit_id] = data
            else:
                group_names = self.get_group_names()
                for group_name in group_names:
                    recording_names = self.get_recording_names(group_name)
                    for recording_name in recording_names:
                        unit_ids = self.get_cellid_names(group_name, recording_name)
                        for unit_id in unit_ids:
                            cellid_name = self.get_original_cellid(unit_id)[2]
                            data = next((v for k, v in self.mat['all_data'][group_name][recording_name][cellid_name].items() if k.lower() == data_type.lower()), None)
                            if data is not None:
                                result[unit_id] = data
            return result
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    #create aliases for get_pre_data and get_post_data for convenience
    def get_pre_data(self, unit_id=None):
        """_summary_

        An alias for the get_data method with the data_type parameter set to 'pre'. 
        It facilitates the retrieval of 'pre' data.
        """
        
        return self.get_data(unit_id, data_type='pre')

    def get_post_data(self, unit_id=None):
        """_summary_

        An alias for the get_data method with the data_type parameter set to 'post'. 
        It facilitates the retrieval of 'post' data.
        """
        return self.get_data(unit_id, data_type='post')
    
    def check_dict_keys(self):
        """
        Check if the specified dict keys are present at both the 'pre' and 'post' levels for all unit IDs 
        and verifies that both levels have the same number of keys.

        Returns:
            dict: A dictionary with unit IDs as keys and a boolean as value indicating whether the conditions are met.
        """

        # Define the set of specified dict keys (all lowercase for case-insensitive comparison)
        specified_keys = {
            'frs_baseline', 'frs_baseline_vec', 'frs_stim', 'fanofactor_baseline', 'fanofactor_stim', 
            'firstspikelatency', 'firstspikelatency_reliability', 'firstspikelatency_pdf_x', 
            'firstspikelatency_pdf_y', 'firstspikelatency_pertrial', 'isi_baseline_cv', 
            'isi_baseline_vec', 'isi_pdf_peak_xy', 'isi_pdf_x', 'isi_pdf_y', 'meanfr_baseline', 
            'meanfr_inst_baseline', 'meanfr_inst_stim', 'meanfr_stim', 'modulationindex', 
            'psths_conv', 'psths_raw', 'peakevokedfr', 'peakevokedfr_latency', 'spiketimes_baseline', 
            'spiketimes_stim', 'spiketimes_trials', 'spiketrains_baseline', 'spiketrains_baseline_ms', 
            'spiketrains_for_psths', 'spiketrains_stim', 'spiketrains_stim_ms', 'spiketrains_trials', 
            'spiketrains_trials_ms', 'stimprob', 'stimresponsivity', 'stim_intensity', 
            'stim_offsets_samples', 'stim_onsets_samples'
        }
        

        # Get all unit IDs
        all_unit_ids = self.get_all_unit_ids()

        results = {}

        for unit_id in all_unit_ids:
            # Retrieve and lowercase the keys from the 'pre' and 'post' dictionaries
            pre_data = self.get_pre_data(unit_id)
            post_data = self.get_post_data(unit_id)

            if pre_data is None or post_data is None:
                results[unit_id] = False
                continue

            pre_keys = {key.lower() for key in pre_data.keys()}
            post_keys = {key.lower() for key in post_data.keys()}

            # Check if all specified keys are present in both dictionaries
            if not (specified_keys.issubset(pre_keys) and specified_keys.issubset(post_keys)):
                results[unit_id] = False
                continue

            # Verify that both dictionaries have the same number of keys
            if len(pre_keys) != len(post_keys):
                results[unit_id] = False
                continue

            results[unit_id] = True

        return results
    
    def get_unit_level_data(self, unit_id=None):
        """
        Retrieves the data for a specified unit ID up to the level just before the 'pre' and 'post' data.
        If no unit ID is provided, retrieves data for all unit IDs.

        Args:
            unit_id (str, optional): The unique unit ID. Defaults to None t

        Returns:
            dict: A dictionary with unit IDs as keys and data as values, or None if not found.
        """
        result = {}
        try:
            if unit_id:
                mapping = self.get_original_cellid(unit_id)
                if mapping:
                    group_name, recording_name, cellid_name = mapping
                    data = self.mat['all_data'][group_name][recording_name][cellid_name]
                    if data is not None:
                        result[unit_id] = data
            else:
                group_names = self.get_group_names()
                for group_name in group_names:
                    recording_names = self.get_recording_names(group_name)
                    for recording_name in recording_names:
                        unit_ids = self.get_cellid_names(group_name, recording_name)
                        for unit_id in unit_ids:
                            cellid_name = self.get_original_cellid(unit_id)[2]
                            data = self.mat['all_data'][group_name][recording_name][cellid_name]
                            if data is not None:
                                result[unit_id] = data
            return result
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    def get_unit_data_keys(self, unit_id):
        """
        Retrieves the keys present at the level just before the 'pre' and 'post' data for a specified unit ID.

        Args:
            unit_id (str): The unique unit ID.

        Returns:
            list: A list containing the keys present at the specified level for the given unit ID, or None if an error occurs.
        """
        try:
            # Get the mapping to the original identifiers
            mapping = self.get_original_cellid(unit_id)
            if mapping:
                group_name, recording_name, cellid_name = mapping

                # Get the data at the level just before 'pre' and 'post'
                data = self.mat['all_data'][group_name][recording_name][cellid_name]

                # Get the key names at this level
                key_names = list(data.keys())

                return key_names
            else:
                print(f"No mapping found for unit ID: {unit_id}")
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    def get_unit_table(self):
        """
        Creates a DataFrame where each row corresponds to a unit, indexed by the unit IDs, 
        and the columns contain data for each unit up to the level just before the 'pre' and 'post' data.
        
        Returns:
            pd.DataFrame: The DataFrame containing the data for each unit.
        """
        # Get all unit IDs using the existing method
        unit_ids = self.get_all_unit_ids()
        
        # Create an empty list to store the data for each unit
        data_list = []
        
        # Specify the keys you are interested in (excluding 'Pre' and 'Post')
        keys_of_interest = [
                'Amplitude', 'Cell_Type', 'ChemStimTime_note', 'ChemStimTime_s', 'ChemStimTime_samples', 
                'FR_time_cutoff_after_stim_ms', 'FRs_baseline', 'FRs_baseline_vec', 'FRs_stim', 'FanoFactor_baseline', 
                'FanoFactor_stim', 'FirstSpikeLatency', 'FirstSpikeLatency_Reliability', 'FirstSpikeLatency_pdf_x', 
                'FirstSpikeLatency_pdf_y', 'FirstSpikeLatency_perTrial', 'Header', 'ISI_baseline_CV', 'ISI_baseline_vec', 
                'ISI_pdf_peak_xy', 'ISI_pdf_x', 'ISI_pdf_y', 'ISI_violations_percent', 'IsSingleUnit', 
                'MeanFR_baseline', 'MeanFR_inst_baseline', 'MeanFR_inst_stim', 'MeanFR_stim', 'MeanFR_total', 
                'Mean_Waveform', 'ModulationIndex', 'Normalized_Template_Waveform', 'PSTHs_conv', 'PSTHs_raw', 
                'Peak1ToTrough_ratio', 'Peak2ToTrough_ratio', 'PeakEvokedFR', 'PeakEvokedFR_Latency', 'PeakToPeak_ratio', 
                'Recording_Duration', 'Sampling_Frequency', 'SpikeHalfWidth', 'SpikeTimes_all', 'SpikeTimes_baseline', 'SpikeTimes_stim', 
                'SpikeTimes_trials', 'SpikeTrains_baseline', 'SpikeTrains_baseline_ms', 'SpikeTrains_for_PSTHs', 'SpikeTrains_stim', 
                'SpikeTrains_stim_ms', 'SpikeTrains_trials', 'SpikeTrains_trials_ms', 'StimProb', 'StimResponsivity', 'Stim_Intensity', 
                'Stim_Offsets_samples', 'Stim_Onsets_samples', 'Template_Channel', 'Template_Channel_Position', 'TroughToPeak_duration', 
                'UnNormalized_Template_Waveform', 'peak1_normalized_amplitude'
        ]
        
        # Iterate over all unit IDs
        for unit_id in unit_ids:
            # Get the mapping to the original identifiers
            mapping = self.get_original_cellid(unit_id)
            if mapping:
                group_name, recording_name, cellid_name = mapping

                # Get the data for the current unit up to the level just before the 'pre' and 'post' data
                unit_data = self.mat['all_data'][group_name][recording_name][cellid_name]
            
                # Create a dictionary to hold the selected data for the current unit
                selected_data = {key: unit_data.get(key, None) for key in keys_of_interest}
            
                # Add group name and original cell ID to the selected data dictionary
                selected_data['Group'] = group_name
                selected_data['OriginalCellID'] = cellid_name
            
                # Append the selected data for the current unit to the list
                data_list.append(selected_data)
            else:
                print(f"No mapping found for unit ID: {unit_id}")
        
        # Create a DataFrame from the list of data, with the unit IDs as the index
        unit_data_df = pd.DataFrame(data_list, index=unit_ids)
        
        return unit_data_df
    
    def construct_stimulus_table(self, recording_names=None):

        # If no recording names are provided, get all recording names
        if recording_names is None:
            group_names = self.get_group_names()
            recording_names = [self.get_recording_names(group_name) for group_name in group_names]
            recording_names = [item for sublist in recording_names for item in sublist]  # Flatten the list
        
        # Allow for a single recording name to be passed as a string
        if isinstance(recording_names, str):
            recording_names = [recording_names]
            
        # Loop over the recording names and construct the stimulus table for each
        for recording_name in recording_names:
            
            # Get all unit IDs
            all_unit_ids = self.get_all_unit_ids()

            # Create a list to store unit IDs associated with the specified recording
            unit_ids_for_recording = []

            # Iterate through all unit IDs to find those associated with the specified recording
            for unit_id in all_unit_ids:
                # Get the unit summary for the current unit ID
                unit_summary = self.get_unit_summary(unit_id)
                
                # Check if the unit is associated with the specified recording
                if unit_summary['Recording'] == recording_name:
                    unit_ids_for_recording.append(unit_id)
            
            # Check if we have any unit IDs for the specified recording
            if not unit_ids_for_recording: # if the list is empty
                print(f"No unit IDs found for the recording: {recording_name}")
                continue

            # Use the first unit ID to get the pre and post epoch data
            unit_id = unit_ids_for_recording[0] # get the first unit ID in the list
            
            # Get the pre and post epoch data
            pre_data = self.get_pre_data(unit_id) # get the pre epoch data for the current unit ID
            post_data = self.get_post_data(unit_id) # get the post epoch data for the current unit ID
            
            # Initialize lists to store stimulus details
            trial_ids = []
            onsets = []
            offsets = []
            intensities = []
            epochs = []   
            
            # Extract stimulus details from pre epoch data
            pre_stim_data = pre_data[unit_id]
            post_stim_data = post_data[unit_id]

            for i, (onset, offset, intensity) in enumerate(zip(pre_stim_data['Stim_Onsets_samples'], pre_stim_data['Stim_Offsets_samples'], pre_stim_data['Stim_Intensity'])):
                trial_ids.append(f"trial{i+1}")
                onsets.append(onset)
                offsets.append(offset)
                intensities.append(intensity)
                epochs.append('Pre')

            # Extract stimulus details from post epoch data
            for i, (onset, offset, intensity) in enumerate(zip(post_stim_data['Stim_Onsets_samples'], post_stim_data['Stim_Offsets_samples'], post_stim_data['Stim_Intensity']), start=len(onsets)):
                trial_ids.append(f"trial{i+1}")
                onsets.append(onset)
                offsets.append(offset)
                intensities.append(intensity)
                epochs.append('Post')

            # Create a dataframe to store the stimulus details
            stimulus_table = pd.DataFrame({
                'Stim_Onset_samples': onsets,
                'Stim_Offset_samples': offsets,
                'Stim_Intensity': intensities,
                'Epoch': epochs,
            })
            
            # Set the 'Trial_ID' column as the index
            stimulus_table.index = trial_ids
            stimulus_table.index.name = 'Trial_ID'

            # Convert 'Stim_Onset_samples' and 'Stim_Offset_samples' to integers
            stimulus_table['Stim_Onset_samples'] = stimulus_table['Stim_Onset_samples'].astype(int)
            stimulus_table['Stim_Offset_samples'] = stimulus_table['Stim_Offset_samples'].astype(int)

            # Convert 'Stim_Intensity' to integers (if they are not already)
            stimulus_table['Stim_Intensity'] = stimulus_table['Stim_Intensity'].astype(int)
            
            # Map the Stim_Intensity values to descriptive labels
            stimulus_table['Stim_Intensity'] = stimulus_table['Stim_Intensity'].replace({
                1: 'zero',
                2: 'low',
                3: 'mid',
                4: 'max'
            })      
            
            # Get the original cell IDs for the unit IDs associated with the recording
            original_cell_ids = [self.get_unit_summary(unit_id)['Original Cell ID'] for unit_id in unit_ids_for_recording]

            # Add columns to the stimulus table to store the unit IDs and original cell IDs
            stimulus_table['Unit_IDs'] = [unit_ids_for_recording] * len(stimulus_table)
            stimulus_table['Original_Cell_IDs'] = [original_cell_ids] * len(stimulus_table)

            # Add the completed stimulus table to the stimulus_tables dictionary
            self.stimulus_tables[recording_name] = stimulus_table
    
    def get_recording_name_from_unit_id(self, unit_id):
        """
        Get the recording name associated with a given unit ID.

        Parameters:
        - unit_id (str): The ID of the unit.

        Returns:
        - str: The name of the recording the unit belongs to.
        """
        
        # Get the unit summary for the specified unit ID
        unit_summary = self.get_unit_summary(unit_id)
        
        # Get and return the recording name from the unit summary
        # (You would need to know the exact key where the recording name is stored in the unit summary dictionary)
        recording_name = unit_summary['Recording']
        
        return recording_name
  
    def get_spike_times(self, unit_id, trial_type=None, epoch=None):
        """
        Get the spike times for a specified unit, optionally filtered by trial type and epoch.

        Parameters:
        - unit_id (str): The ID of the unit whose spike times you want to analyze.
        - trial_type (str, optional): The type of trial ('zero', 'low', 'mid', 'max') to filter by. Defaults to None, which includes all trial types.
        - epoch (str, optional): The epoch ('Pre' or 'Post') to filter by. Defaults to None, which includes both epochs.

        Returns:
        - dict: A dictionary where keys are the trial IDs and values are arrays/lists of spike times (in seconds) for each trial.
        """
        
        # Get the spike times and sampling frequency for the specified unit
        unit_data_dict = self.get_unit_level_data(unit_id)
        unit_data = unit_data_dict[unit_id]
        
        spike_times_all = unit_data['SpikeTimes_all']
        sampling_frequency = unit_data['Sampling_Frequency']
        # Print the value of sampling_frequency to verify it
        print(f"Sampling frequency: {sampling_frequency}")
        
        # Convert spike times from samples to seconds
        spike_times_all = spike_times_all / sampling_frequency
        
        # Print the entire spike_times_all array to verify the spike times
        print(spike_times_all)
        
        # Get the stimulus table for the recording the unit belongs to
        recording_name = self.get_recording_name_from_unit_id(unit_id)
        
        # Check if the stimulus table for the recording is already available in the stimulus_tables attribute
        if recording_name in self.stimulus_tables:
            stimulus_table = self.stimulus_tables[recording_name]
        else:
            # If not available, construct the stimulus table and store it in the stimulus_tables attribute
            stimulus_table = self.construct_stimulus_table(recording_name)
            self.stimulus_tables[recording_name] = stimulus_table
            # Check if the conversion from samples to seconds has already been done to prevent multiple conversions
        
        if not hasattr(self, 'conversion_done'):
            stimulus_table['Stim_Onset_samples'] = stimulus_table['Stim_Onset_samples'] / sampling_frequency
            stimulus_table['Stim_Offset_samples'] = stimulus_table['Stim_Offset_samples'] / sampling_frequency
            self.conversion_done = True
        
        # Filter the stimulus table based on the trial_type and epoch parameters
        if trial_type:
            stimulus_table = stimulus_table[stimulus_table['Stim_Intensity'] == trial_type]
        if epoch:
            stimulus_table = stimulus_table[stimulus_table['Epoch'] == epoch]
        
        # Create a dictionary to store the spike times for each trial
        trial_spike_times = {}
        
        # Loop through each trial and extract the relevant spike times
        for trial_id, trial_data in stimulus_table.iterrows():
            onset = trial_data['Stim_Onset_samples']
            offset = trial_data['Stim_Offset_samples']
            
            # Print onset and offset times for each trial
            print(f"Onset time for {trial_id}: {onset}")
            print(f"Offset time for {trial_id}: {offset}")
            
            # Get the spike times that fall within the onset and offset times of the trial
            trial_spike_times[trial_id] = spike_times_all[(spike_times_all >= onset) & (spike_times_all <= offset)]
        
        return trial_spike_times



class ResponseDistributionPlotter:
    def __init__(self, data):
        """
        Initialize the ResponseDistributionPlotter with the data dictionary obtained from the calculate_mean_responses function.

        Parameters:
        data (dict): The data dictionary containing the pooled mean responses for each group.
        """
        self.data = data

    
    def plot_distribution(self, group_name, epoch, stim_level, bins=10, overlay=False, phase='early'):
        """
        Plot the distribution of mean responses for a specific group, epoch, and stimulation level.

        Parameters:
        group_name (str): The name of the group to plot.
        epoch (str, optional): The epoch to plot ('Pre', 'Post', or None). If None, both 'Pre' and 'Post' are plotted together. Defaults to None.
        stim_level (str): The stimulation level to plot ('Zero', 'Low', 'Mid', 'Max', or 'Pooled'). Defaults to 'Zero'.
        bins (int): The number of bins to use in the histogram. Defaults to 30.
        overlay (bool): Whether to overlay the 'Pre' and 'Post' histograms on a single plot. Defaults to False.
        phase (str): The phase to plot data for ('early' or 'late'). Default is 'early'.

        Returns:
        None: The function plots the distribution and does not return any value.
        """
        
        # Get the data for the specified group, epoch, and stimulation level
        group_data = self.data[group_name][epoch]
        data_to_plot = [unit_data[f'{epoch}_{stim_level}_{phase}'] for unit_data in group_data]

        # Create a new figure
        plt.figure()

        # Define colors for each epoch
        colors = {'Pre': 'grey', 'Post': 'blue'}

        # Plot the histogram
        plt.hist(data_to_plot, bins=bins, alpha=0.5, label=f'{epoch}_{stim_level}_{phase}', histtype='step', color=colors[epoch])

        # Set the plot title and labels
        plt.title(f'{group_name} {epoch} {stim_level} {phase} Response Distribution')
        plt.xlabel('Mean Response')
        plt.ylabel('Frequency')

        # Add a legend
        plt.legend()

        # Show the plot
        if not overlay:
            plt.show()
        
    
    def plot_box_and_whisker(self, group_name, epoch=None, stim_level='Zero', overlay=False):
        """
        Plot box and whisker plots of mean responses for a specific group, epoch, and stimulation level.

        Parameters:
        group_name (str): The name of the group to plot.
        epoch (str, optional): The epoch to plot ('Pre', 'Post', or None). If None, both 'Pre' and 'Post' are plotted together. Defaults to None.
        stim_level (str): The stimulation level to plot ('Zero', 'Low', 'Mid', 'Max', or 'Pooled'). Defaults to 'Zero'.
        overlay (bool): Whether to overlay the 'Pre' and 'Post' box plots on a single plot. Defaults to False.

        Returns:
        None: The function plots the box and whisker plots and does not return any value.
        """
        epochs = [epoch] if epoch else ['Pre', 'Post']
        
        data_to_plot = []
        labels = []
        
        for epoch in epochs:
            # Get the mean responses for the specified group, epoch, and stimulation level
            mean_responses = self.data[group_name][stim_level][epoch]
            
            data_to_plot.append(mean_responses)
            labels.append(epoch)
        
        # Set plot title and labels
        plt.title(f'{group_name} - {stim_level} Stimulation')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Response')
        
        # Create box and whisker plot
        plt.boxplot(data_to_plot, labels=labels)
        
        # Display the plot
        plt.show()

def calculate_mean_responses(EED, group_name=None):
    """
    Calculate the mean responses during the early phase (0-50 ms post-stimulus) and late phase (100-700 ms post-stimulus) for each unit and pool them together per group.

    Parameters:
    EED (object): The object containing the electrophysiology data.
    group_name (str, optional): The name of the group to analyze. Defaults to None, in which case all groups are analyzed.

    Returns:
    dict: A dictionary containing the pooled mean responses for each group.
    """
    
    # Get list of group names
    group_names = [group_name] if group_name else EED.group_names
    
    # Dictionary to store the pooled mean responses for each group
    pooled_mean_responses = {}
    
    # Loop through each group
    for group in group_names:
        recording_names = EED.get_recording_names(group)
        
        # List to store the mean responses for all units in the current group
        group_mean_responses = {'Pre': [], 'Post': []}
        
        # Loop through each recording
        for recording in recording_names:
            cellid_names = EED.get_cellid_names(group, recording)
            
            # Loop through each cell ID
            for cell_id in cellid_names:
                # Get the pre and post stim data
                data = EED.get_pre_post_data(group, recording, cell_id)
                
                # Define stimulation levels and pooled stimulation levels
                stim_levels = ['Zero', 'Low', 'Mid', 'Max', 'Pooled']
                
                # Dictionary to store the mean responses for the current unit
                unit_mean_responses = {'Recording': recording, 'CellID': cell_id}
                
                # Loop through each epoch (pre and post) and stimulation level to calculate mean responses
                for epoch in ['Pre', 'Post']:
                    for stim_level in stim_levels:
                        if stim_level == 'Pooled':
                            stim_indices = [1, 2, 3]
                        else:
                            stim_indices = [stim_levels.index(stim_level)]
                        
                        # Get spike trains for the current stimulation level
                        spiketrains = np.concatenate([data[epoch]['SpikeTrains_for_PSTHs'][i] for i in stim_indices], axis=0)
                        
                        # Extract spike data for the early phase (0-50 ms post-stimulus)
                        early_phase = spiketrains[:, 500:550]  # Adjust indices as necessary
                        
                        # Extract spike data for the late phase (100-700 ms post-stimulus)
                        late_phase = spiketrains[:, 600:1200]  # Adjust indices as necessary
                        
                        # Calculate the total number of spikes in each trial during the early phase
                        mean_response_early = early_phase.sum(axis=1).mean()
                        
                        # Calculate the total number of spikes in each trial during the late phase
                        mean_response_late = late_phase.sum(axis=1).mean()
                        
                        # Add the mean response to the dictionary
                        unit_mean_responses[f'{epoch}_{stim_level}_early'] = mean_response_early
                        unit_mean_responses[f'{epoch}_{stim_level}_late'] = mean_response_late
                
                # Add the mean responses for the current unit to the list
                group_mean_responses['Pre'].append(unit_mean_responses)
                group_mean_responses['Post'].append(unit_mean_responses)
        
        # Add the mean responses for the current group to the dictionary
        pooled_mean_responses[group] = group_mean_responses
    
    return pooled_mean_responses



