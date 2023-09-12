import mat73 
import matplotlib.pyplot as plt
import numpy as np
import hashlib

class ExtractEphysData:
    """_summary_
    Create a class that will allow me to extract the ephys data from the matfiles 
    Args:
        matfile_directory (str): the directory of the matfile
        
    Attributes:
        mat (matfile): the matfile
        # group_names attribute has been removed
        
    Returns:
        _type_: _description_
        #TODO: add the return types and descriptions for each function
    """
    
    def __init__(self, matfile_directory):
        # use mat73.loadmat to load mat files
        mat = mat73.loadmat(matfile_directory, use_attrdict=True)
        
        # store the matfile
        self.mat = mat 
        
        # The group_names attribute initialization has been moved to a separate method

    def get_group_names(self):
        """
        Returns a list of group names present in the matfile.
        
        Returns:
            list: A list of group names.
        """
        # Getting the group names dynamically whenever the method is called
        group_names = list(self.mat['all_data'].keys())
        return group_names

    def get_recording_names(self, group_name):
        """
        Returns a list of recording names for a given group name.
        
        Args:
            group_name (str): The name of the group to retrieve recording names from.
            
        Returns:
            list: A list of recording names for the given group name.
        """
        # implementation code here
        recording_names = []
        for recording in self.mat['all_data'][group_name]:
            recording_names.append(recording)
        return recording_names 

    def get_cellid_names(self, group_name, recording_name):
        """
        Returns a list of unique unit IDs for a specific group and recording.

        Args:
            group_name (str): The name of the group to retrieve unit IDs from.
            recording_name (str): The name of the recording to retrieve unit IDs from.

        Returns:
            list: A list of unique unit IDs for the given group and recording.
        """
        unit_ids = []
        for cellid_name in self.mat['all_data'][group_name][recording_name].keys():
            # Creating a unique unit ID using group name, recording name, and cell ID
            unique_unit_id = hashlib.md5(f"{group_name}_{recording_name}_{cellid_name}".encode()).hexdigest()
            unit_ids.append(unique_unit_id)
        return unit_ids
    
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
    
    def print_all_unit_ids(self):
        """
        Iterates over all groups and recordings to print the unique unit IDs.

        Returns:
            None
        """
        try:
            # Get all group names
            group_names = self.get_group_names()
            
            # Loop over all groups
            for group_name in group_names:
                print(f"Group name: {group_name}")

                # Get all recording names for the current group
                recording_names = self.get_recording_names(group_name)

                # Loop over all recordings in the current group
                for recording_name in recording_names:
                    print(f"  Recording name: {recording_name}")

                    # Get all unique unit IDs for the current recording
                    unit_ids = self.get_cellid_names(group_name, recording_name)

                    # Loop over all unit IDs in the current recording and print them
                    for i, unit_id in enumerate(unit_ids):
                        print(f"    Unit {i+1}: {unit_id}")
        except Exception as e:
            print(f"An error occurred: {e}")
            
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
                    
    

    def get_pre_post_data(self, group_name=None, recording_name=None, cellid_name=None):
        
        # Create an empty dictionary to store the pre and post data with 'Pre' and 'Post' as keys
        pre_post_data = {} 
        
        # If the user specifies a group name, recording name, and cellid name, then return the pre and post data for that cellid 
        if group_name is not None and recording_name is not None and cellid_name is not None: 
            
            # Use the extract_ephys_data method to get the data
            cell_data = self.extract_ephys_data(group_name, recording_name, cellid_name)
            
            try:
                # First grab the pre data
                pre_data = cell_data['Pre']
                pre_post_data['Pre'] = pre_data
            except KeyError:
                print(f"Key 'Pre' not found. Available keys: {list(cell_data.keys())}")
                return
            
            try:
                # Then grab the post data
                post_data = cell_data['Post']
                pre_post_data['Post'] = post_data
            except KeyError:
                print(f"Key 'Post' not found. Available keys: {list(cell_data.keys())}")
                return
        
        # If not the above, then return the pre and post data for all cellids in the group by iterating through the group, recording, and cellid names
        else: 
            for group_name in self.group_names:
                for recording_name in self.get_recording_names(group_name):
                    for cellid_name in self.get_cellid_names(group_name, recording_name):
                        # Use the extract_ephys_data method to get the data
                        cell_data = self.extract_ephys_data(group_name, recording_name, cellid_name)
                        
                        try:
                            # First grab the pre data
                            pre_data = cell_data['Pre']
                            pre_post_data['Pre'] = pre_data
                        except KeyError:
                            print(f"Key 'Pre' not found in group {group_name}, recording {recording_name}, cellid {cellid_name}. Available keys: {list(cell_data.keys())}")
                            continue
                        
                        try:
                            # Then grab the post data
                            post_data = cell_data['Post']
                            pre_post_data['Post'] = post_data
                        except KeyError:
                            print(f"Key 'Post' not found in group {group_name}, recording {recording_name}, cellid {cellid_name}. Available keys: {list(cell_data.keys())}")
                            continue
            
        # Return the pre and post data
        return pre_post_data
    
    


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



