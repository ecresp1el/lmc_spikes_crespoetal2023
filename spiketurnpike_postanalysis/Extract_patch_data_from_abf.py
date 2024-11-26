import pyabf  # Make sure you have the pyabf library installed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PatchDataExtractor:
    def __init__(self, abf_file_path):
        """
        Initialize the extractor with the .abf file path.

        Args:
            abf_file_path (str): Path to the .abf file.
        """
        if not os.path.exists(abf_file_path):
            raise FileNotFoundError(f"{abf_file_path} does not exist.")
        self.abf_file_path = abf_file_path
        self.abf = pyabf.ABF(abf_file_path)
    
    def plot_channel(self, channel=0, sweep=0):
        """
        Plot the data for a specific channel and sweep.
        
        Args:
            channel (int): The channel index.
            sweep (int): The sweep number.
        """
        self.abf.setSweep(sweepNumber=sweep, channel=channel)
        plt.figure(figsize=(8, 5))
        plt.plot(self.abf.sweepX, self.abf.sweepY, label=f"Channel {channel}, Sweep {sweep}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()