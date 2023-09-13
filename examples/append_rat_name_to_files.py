import os
import glob
import re

def append_rat_name_to_files(directory):
    # Get all .csv files in the directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    for file_path in csv_files:
        # Extract the base name of the file (without the directory path)
        base_name = os.path.basename(file_path)
        # Extract the parts of the filename
        match = re.match(r'(.*)_T(.)(.).*_Rat(\d)\.csv', base_name)
        if match:
            prefix, top_letter, bottom_letter, rat_number = match.groups()
            # Determine the rat name based on the rat number
            if rat_number in ['1', '2']:
                rat_name = top_letter + rat_number
            else:
                rat_name = bottom_letter + str(int(rat_number) - 2)
            # Construct the new filename
            new_name = f'{prefix}_Rat{rat_number}_{rat_name}.csv'
            # Rename the file
            os.rename(file_path, os.path.join(directory, new_name))