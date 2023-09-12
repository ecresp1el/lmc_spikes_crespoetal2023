import pandas as pd 

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
        
        self.unit_data['SpikeTimes_in_seconds'] = self.unit_data['SpikeTimes_all'] / self.unit_data['Sampling_Frequency']
        self.unit_data['ChemStimTime_in_seconds'] = self.unit_data['ChemStimTime_samples'] / self.unit_data['Sampling_Frequency']
        