#import mat73 to load mat files
import mat73
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
    
    #create a method that will acces the cellid names for each recording using the attributes of the class
    def get_cellid_names(self, group_name):
        
        #create an empty list to store the cellid names
        cellid_names = []
        
        #for each recording in the group, append the cellid names to the cellid_names list
        for recording in self.mat['all_data'][group_name]:
            
            cellid_names.append(self.mat['all_data'][group_name][recording].keys())
            
        #return the cellid names
        return cellid_names 
    

        #create a method that will access the 'Pre' and 'Post' dictionaries for each cellid 
    def get_pre_post_data(self, group_name=None, recording_name=None, cellid_name=None):
            
            #create an empty list to store the pre and post data
            pre_post_data = [] 
            
            #if the user specifies a group name, recording name, and cellid name, then return the pre and post data for that cellid 
            if group_name is not None and recording_name is not None and cellid_name is not None: 
                
                #first grab the pre data, by finding the 'Pre' key in the cellid dictionary 
                pre_post_data.append(self.mat['all_data'][group_name][recording_name][cellid_name]['Pre'])
                
                #then grab the post data
                pre_post_data.append(self.mat['all_data'][group_name][recording_name][cellid_name]['Post']) 
            
            # if not the above, then return the pre and post data for all cellids in the group by iterating through the group, recording, and cellid names
            else: 
                for group_name in self.group_names:
                    for recording_name in self.get_recording_names(group_name):
                        for cellid_name in self.get_cellid_names(group_name, recording_name):
                            #first grab the pre data
                            pre_post_data.append(self.mat['all_data'][group_name][recording_name][cellid_name]['Pre'])
                            
                            #then grab the post data
                            pre_post_data.append(self.mat['all_data'][group_name][recording_name][cellid_name]['Post'])
                
            #return the pre and post data