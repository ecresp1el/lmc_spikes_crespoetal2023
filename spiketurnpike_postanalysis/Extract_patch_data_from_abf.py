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