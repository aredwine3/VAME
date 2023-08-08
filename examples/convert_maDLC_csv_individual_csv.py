# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:20:44 2022

@author: Charitha Omprakash, LIN Magdeburg, charitha.omprakash@lin-magdeburg.de 

This file converts a multi-animal DLC CSV to several single animal DLC files.
Those can be used as input to run VAME.
"""

import sys
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def convert_multi_csv_to_individual_csv(csv_files_path):
    # Get a sorted list of all csv files in the provided path
    csvs = sorted(glob.glob(os.path.join(csv_files_path, '*.csv*')))
    
    # Loop through each csv file
    for csv in csvs:
        # Read the csv file into a pandas DataFrame
        fname = pd.read_csv(csv, header=[0,1,2], index_col=0, skiprows=1)
        # Get a list of unique individuals from the DataFrame columns
        individuals = fname.columns.get_level_values('individuals').unique()
        # Loop through each individual
        for ind in individuals:
            # Create a temporary DataFrame for the current individual
            fname_temp = fname[ind]
            # Define the path for the new csv file
            fname_temp_path = os.path.splitext(csv)[0] + '_' + ind + '.csv'
            # Write the temporary DataFrame to a new csv file
            fname_temp.to_csv(fname_temp_path, index=True, header=True)

if __name__ == "__main__":
    # Check if the script was called with the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python convert_maDLC_csv_individual_csv.py <csv_files_path>")
        sys.exit(1)

    # Get the path to the csv files from the command line arguments
    csv_files_path = sys.argv[1]
    # Call the function to convert the csv files
    convert_multi_csv_to_individual_csv(csv_files_path)
