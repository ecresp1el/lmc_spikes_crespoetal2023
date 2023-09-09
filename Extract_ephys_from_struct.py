#import mat73 to load mat files
import mat73
import matplotlib.pyplot as plt
class ExtractEphysData: 
    """_summary_
    Create a class that will allow me to extract the ephys data from the matfiles 
    Args:
        matfile_directory (str): the directory of the matfile
        
    Attributes:
        mat (matfile): the matfile
        group_names (list): the names of the groups in the matfile
        lmc_opsin_recording_names (list): the names of the recordings in the lmc_opsin group
        lmc_noopsin_recording_names (list): the names of the recordings in the lmc_noopsin group
        
    Returns:
        _type_: _description_
        #TODO: add the return types and descriptions for each function
        
        
    """
    
    #initialize the class
    def __init__(self, matfile_directory):

            #use mat73.loadmat to load mat files
            mat = mat73.loadmat(matfile_directory, use_attrdict=True)
            
            #store the matfile
            self.mat = mat 
            
            #store the group names
            self.group_names = mat['all_data'].keys() 
            
            #store the recording names for the lmc_opsin group
            self.lmc_opsin_recording_names = mat['all_data']['Lmc_opsin'].keys() 
            
            #store the recording names for the lmc_noopsin group
            self.lmc_noopsin_recording_names = mat['all_data']['Lmc_noopsin'].keys()
            
    
    def extract_ephys_data(self, group_name, recording_name, cellid_name):
            
            #extract the data from the matfile
            data = self.mat['all_data'][group_name][recording_name][cellid_name]
            
            #return the data
            return data
        
    #create a function that will take in a .mat file directory and print the keys of each group, and recording, and tell you how many mice are in each group
    #this will become a module that can be imported into other notebooks
    
    def load_matfiles_printdata(self):
        #kind the level of all_data that contains the group level data
        print(self.mat['all_data'].keys())
        
        #for each group, for each recording names, print the total number of cells in each recording
        for group_name in self.group_names:
                
                print('For the group', group_name, 'there are', len(self.mat['all_data'][group_name]), 'mice')
                
                for recording in self.mat['all_data'][group_name]:
                    
                    print('For the recording', recording, 'there are', len(self.mat['all_data'][group_name][recording]), 'cells')
                    
    #create a method that will access the name of each recording in a group using the attributes of the class
    def get_recording_names(self, group_name):
        """
        Returns a list of recording names for a given group name.
        
        Args:
            group_name (str): The name of the group to retrieve recording names from.
            
        Returns:
            list: A list of recording names for the given group name.
        """
        # implementation code here
            
        #create an empty list to store the recording names
        recording_names = []
            
        #for each recording in the group, append the recording names to the recording_names list
        for recording in self.mat['all_data'][group_name]:
                
            recording_names.append(recording)
                
        #return the recording names
        return recording_names 
    
    def get_cellid_names(self, group_name, recording_name):
        """
        Returns a list of cell ID names for a specific group and recording.

        Args:
            group_name (str): The name of the group to retrieve cell ID names from.
            recording_name (str): The name of the recording to retrieve cell ID names from.

        Returns:
            list: A list of cell ID names for the given group and recording.
        """
        # Create an empty list to store the cell ID names
        cellid_names = []

        # Get the cell ID names for the specified recording
        for cellid_name in self.mat['all_data'][group_name][recording_name].keys():
            cellid_names.append(cellid_name)

        # Return the list of cell ID names
        return cellid_names

    

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

    def plot_distribution(self, group_name, epoch=None, stim_level='Zero', bins=30, overlay=False):
        """
        Plot the distribution of mean responses for a specific group, epoch, and stimulation level.

        Parameters:
        group_name (str): The name of the group to plot.
        epoch (str, optional): The epoch to plot ('Pre', 'Post', or None). If None, both 'Pre' and 'Post' are plotted together. Defaults to None.
        stim_level (str): The stimulation level to plot ('Zero', 'Low', 'Mid', 'Max', or 'Pooled'). Defaults to 'Zero'.
        bins (int): The number of bins to use in the histogram. Defaults to 30.
        overlay (bool): Whether to overlay the 'Pre' and 'Post' histograms on a single plot. Defaults to False.

        Returns:
        None: The function plots the distribution and does not return any value.
        """
        epochs = [epoch] if epoch else ['Pre', 'Post']
        
        for epoch in epochs:
            # Get the mean responses for the specified group, epoch, and stimulation level
            mean_responses = [unit[f'{epoch}_{stim_level}'] for unit in self.data[group_name][epoch]]
            
            # Define color based on the epoch
            color = 'grey' if epoch == 'Pre' else 'blue'
            
            # Plot the distribution of mean responses
            plt.hist(mean_responses, bins=bins, color=color, edgecolor='black', alpha=0.5, facecolor='none', linewidth=1.2, label=epoch)
        
        # Set plot title and labels
        plt.title(f'{group_name} - {stim_level} Stimulation')
        plt.xlabel('Mean Response')
        plt.ylabel('Frequency')
        
        if overlay:
            plt.legend()
        
        # Display the plot
        plt.show()
