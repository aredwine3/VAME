import concurrent.futures
import csv
import glob
import os
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from icecream import ic
from ruamel.yaml import YAML
from scipy.stats import ttest_ind, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM
from statsmodels.regression.mixed_linear_model import MixedLM
from tqdm import tqdm
import vame.custom.ALR_kinematics as kin
import polars as pl
from vame.util.auxiliary import read_config
import vame.custom.ALR_analysis as ana
from concurrent.futures import ProcessPoolExecutor

# Set the Matplotlib backend based on the environment.
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')  # Use this backend for headless environments (e.g., Google Colab, some remote servers)
else:
    matplotlib.use('Qt5Agg')  # Use this backend for environments with a display server


def process_file(file_data):
    file, i, path_to_file, dlc_data_type, fps, labels_list = file_data
    
    filename, time_point, group, full_group_notation = parse_filename(file)
    labels = labels_list[i]
    data = kin.get_dlc_file(path_to_file, dlc_data_type, file)
    rat = kin.get_dat_rat(data)    
    centroid_x, centroid_y = kin.calculate_centroid(data)
    in_center = kin.is_dat_rat_in_center(data)
    distance_cm = kin.distance_traveled(data)
    rat_speed = kin.calculate_speed_with_spline(data, fps, window_size=5, pixel_to_cm=0.215)
    
    trim_length = len(centroid_x) - len(labels)
    
    centroid_x_trimmed = centroid_x[trim_length:]
    centroid_y_trimmed = centroid_y[trim_length:]
    in_center_trimmed = in_center[trim_length:]
    distance_trimmed = distance_cm[trim_length:]
    rat_speed_trimmed = rat_speed[trim_length:]
    
    
    motif_series = pl.Series(labels).cast(pl.Int64)
    centroid_x_series = pl.Series(centroid_x_trimmed).cast(pl.Float32)
    centroid_y_series = pl.Series(centroid_y_trimmed).cast(pl.Float32)
    in_center_series = pl.Series(in_center_trimmed).cast(pl.Float32)
    distance_series = pl.Series(distance_trimmed).cast(pl.Float32)
    speed_series = pl.Series(rat_speed_trimmed).cast(pl.Float32)
    
    temp_df = pl.DataFrame({
        "file_name": [file] * len(labels),
        "frame": list(range(len(labels))),
        "motif": motif_series,
        "centroid_x": centroid_x_series,
        "centroid_y": centroid_y_series,
        "in_center": in_center_series,
        "distance": distance_series,
        "speed": speed_series,
        "rat": [rat] * len(labels),
        "rat_id": [full_group_notation] * len(labels),
        "group": [group] * len(labels),
        "time_point": [time_point] * len(labels)   
    })
    return temp_df


def create_andOR_get_master_df(config, fps=30, create_new_df=False):
    files = get_files(config)
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    path_to_file = cfg['project_path']
    dlc_data_type = input("Were your DLC .csv files originally multi-animal? (y/n): ")
    labels_list = ana.get_labels(cfg, files, model_name, n_cluster)
    
    df = pl.DataFrame({
        "file_name": pl.Series([], dtype=pl.Utf8),
        "frame": pl.Series([], dtype=pl.Int64),
        "motif": pl.Series([], dtype=pl.Int64),
        "centroid_x": pl.Series([], dtype=pl.Float32),
        "centroid_y": pl.Series([], dtype=pl.Float32),
        "in_center": pl.Series([],  dtype=pl.Float32),
        "distance": pl.Series([], dtype=pl.Float32),
        "speed": pl.Series([], dtype=pl.Float32),
        "rat": pl.Series([], dtype=pl.Utf8),
        "rat_id": pl.Series([], dtype=pl.Utf8),
        "group": pl.Series([], dtype=pl.Utf8),
        "time_point": pl.Series([], dtype=pl.Utf8)
    })

    if parameterization == 'hmm':
        df_path = (os.path.join(path_to_file, 'results', f"all_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
    else:
        df_path = (os.path.join(path_to_file, 'results', f"all_sequences_{parameterization}-{n_cluster}.csv"))
    

    if not create_new_df and os.path.exists(df_path):
        df = pl.read_csv(df_path)
    else:
        print("Creating new master data frame...")
        cpu_count = os.cpu_count()
        if cpu_count is not None and cpu_count > 1:
            # Prepare the data for processing
            file_data_list = [(file, i, path_to_file, dlc_data_type, fps, labels_list) for i, file in enumerate(files)]

            # Process the files in parallel
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(process_file, file_data_list))

            # Concatenate all DataFrames into one
            df = pl.concat(results)
        else:
            # Create a store of when each motif is happening for each file
            for i, file in enumerate(files):
                filename, time_point, group, full_group_notation = parse_filename(file)
                labels = labels_list[i]
                data = kin.get_dlc_file(path_to_file, dlc_data_type, file)
                rat = kin.get_dat_rat(data)    
                centroid_x, centroid_y = kin.calculate_centroid(data)
                in_center = kin.is_dat_rat_in_center(data)
                distance_cm = kin.distance_traveled(data)
                rat_speed = kin.calculate_speed_with_spline(data, fps, window_size=5, pixel_to_cm=0.215)
                
                # Calculate the number of elements to trim
                trim_length = len(centroid_x) - len(labels)

                # Trim the centroid lists
                centroid_x_trimmed = centroid_x[trim_length:]
                centroid_y_trimmed = centroid_y[trim_length:]
                in_center_trimmed = in_center[trim_length:]
                distance_trimmed = distance_cm[trim_length:]
                rat_speed_trimmed = rat_speed[trim_length:]


                motif_series = pl.Series(labels).cast(pl.Int64)
                centroid_x_series = pl.Series(centroid_x_trimmed).cast(pl.Float32)
                centroid_y_series = pl.Series(centroid_y_trimmed).cast(pl.Float32)
                in_center_series = pl.Series(in_center_trimmed).cast(pl.Float32)
                distance_series = pl.Series(distance_trimmed).cast(pl.Float32)
                speed_series = pl.Series(rat_speed_trimmed).cast(pl.Float32)

                # Create a new DataFrame with the labels and the rat value for this file
                temp_df = pl.DataFrame({
                    "file_name": [file] * len(labels),
                    "frame": list(range(len(labels))),
                    "motif": motif_series,
                    "centroid_x": centroid_x_series,
                    "centroid_y": centroid_y_series,
                    "in_center": in_center_series,
                    "distance": distance_series,
                    "speed": speed_series,
                    "rat": [rat] * len(labels),
                    "rat_id": [full_group_notation] * len(labels),
                    "group": [group] * len(labels),
                    "time_point": [time_point] * len(labels)   
                })
                
                temp_df = temp_df.with_columns([
                    pl.col("frame").cast(pl.Int64),
                ])

                # Concatenate the new DataFrame with the existing one
                df = pl.concat([df, temp_df])
   

        if parameterization == 'hmm':
            df.write_csv(os.path.join(path_to_file, 'results', f"all_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
        else:
            df.write_csv(os.path.join(path_to_file, 'results', f"all_sequences_{parameterization}-{n_cluster}.csv"))

    return df

def run_kruskal_wallis_test(df, groups, values):
    """
    Run a Kruskal-Wallis H-test for independent samples.

    Parameters:
    - df: pandas DataFrame containing the data
    - groups: the column name in df that represents the groups
    - values: the column name in df that represents the values to compare

    Returns:
    - Kruskal-Wallis H-test result
    """
    data = [group[values].values for name, group in df.groupby(groups)]
    return kruskal(*data)


def run_repeated_measures_anova(df, subject, within, between, dependent_var):
    """
    Run a repeated measures ANOVA.

    Parameters:
    - df: pandas DataFrame containing the data
    - subject: the column name in df that represents the subject ID
    - within: the column name in df that represents the within-subject factor (e.g., time)
    - between: the column name in df that represents the between-subject factor (e.g., group)
    - dependent_var: the column name in df that represents the dependent variable (e.g., value)

    Returns:
    - AnovaRM object containing the ANOVA results
    """
    aovrm = AnovaRM(df, depvar=dependent_var, subject=subject, within=[within, between], aggregate_func='mean')
    res = aovrm.fit()
    return res



def run_mixed_effects_model(df, dependent_var, groups, re_formula, fe_formula='1', maxiter=100):
    """
    Run a mixed-effects model.

    Parameters:
    - df: pandas DataFrame containing the data
    - dependent_var: the column name in df that represents the dependent variable
    - groups: the column name in df that represents the random effects grouping
    - re_formula: the formula representing the random effects
    - fe_formula: the formula representing the fixed effects (default is intercept only)
    - maxiter: the maximum number of iterations for the optimizer (default is 100)

    Returns:
    - MixedLMResults object containing the model results
    """
    model = MixedLM.from_formula(f"{dependent_var} ~ {fe_formula}", groups=df[groups], re_formula=re_formula, data=df)
    optimizers = ['lbfgs', 'bfgs', 'nm', 'powell', 'cg', 'newton']
    
    for optimizer in optimizers:
        try:
            result = model.fit(method=optimizer, maxiter=maxiter)
            if result.converged:
                print(f"Converged with {optimizer}")
                return result
        except np.linalg.LinAlgError:
            print(f"Failed with {optimizer}")
    
    raise ValueError("None of the optimizers converged")
    


def group_motifs_by_cluster(clustering):
    cluster_dict = {}
    for motif, cluster in enumerate(clustering):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(motif)
    return cluster_dict


def assign_clusters(df, clustered_motifs):
    # Invert the clustered_motifs dictionary to map motifs to their cluster
    motif_to_cluster = {motif: cluster for cluster, motifs in clustered_motifs.items() for motif in motifs}
    
    # Map the 'Motif' column to the 'Cluster' column using the motif_to_cluster mapping
    df['Cluster'] = df['Motif'].map(motif_to_cluster)
    
    return df


def get_files(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    files = []
    if cfg['all_data'] == 'No' or cfg['all_data']=='no':
        all_flag = input("Do you want to get files for your entire dataset? \n"
                            "If you only want to use a specific dataset type filename: \n"
                            "yes/no/filename ")
    else:
        all_flag = 'yes'

    if all_flag == 'yes' or all_flag == 'Yes':
        for file in cfg['video_sets']:
            files.append(file)

    elif all_flag == 'no' or all_flag == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to quantify " + file + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue
    else:
        files.append(all_flag)
    return files


def copy_file(file, src_dir, dest_dir):
    shutil.copy(os.path.join(src_dir, file), dest_dir)



def copy_motif_usage_files_multithreaded(pattern, destination_folder):
    # Ensure the destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Find all files matching the pattern
    files = glob.glob(pattern, recursive=True)
    
    # Use ThreadPoolExecutor to copy files in parallel
    with ThreadPoolExecutor() as executor:
        # Create a list of futures
        futures = [executor.submit(copy_file, os.path.basename(file), os.path.dirname(file), destination_folder) for file in files]  # noqa: E501
        
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Get the result of the future, which will raise an exception if the copying failed
            except Exception as e:
                print(f"An error occurred: {e}")
    
    return files  # Return the list of files that were copied



def combineBehavior(config, files, save=True, legacy=False):
    """
    Docstring:
        Combines motif usage for all samples into one CSV file.
        
    Parameters
    ----------
    config : string 
        path to config file
    save : bool
    n_cluster : int 
        number of clusters to analyze (behavioral segmentation data must exist)
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    project_path = cfg['project_path']
    n_cluster=cfg['n_cluster']
    model_name = cfg['model_name']
    parameterization = cfg['parameterization']
    hmm_iters = cfg.get('hmm_iters', 0)
    load_data = cfg['load_data']
    
    if not files:
       files = []
       files = get_files(config)
       
    
    cat = pd.DataFrame()
    for file in files:
        if legacy:
            arr = np.load(os.path.join(project_path, 'results/' + file + '/VAME_NPW/kmeans-' + str(n_cluster) + '/behavior_quantification/motif_usage.npy'))
        elif not legacy:
            if parameterization == 'hmm':
                arr = np.load(os.path.join(project_path, 'results',file,model_name, load_data, parameterization+'-'+str(n_cluster)+'-'+str(hmm_iters),'motif_usage_'+file+'.npy'))
            else:
                arr = np.load(os.path.join(project_path, 'results',file,model_name, load_data, parameterization+'-'+str(n_cluster),'motif_usage_'+file+'.npy'))
        df = pd.DataFrame(arr, columns=[file])
        cat = pd.concat([cat, df], axis=1)
    if save:
        if parameterization == 'hmm':
            cat.to_csv(os.path.join(project_path, f'CombinedMotifUsage_{parameterization}-{n_cluster}-{hmm_iters}.csv'))
        else:
            cat.to_csv(os.path.join(project_path, f'CombinedMotifUsage_{parameterization}-{n_cluster}.csv'))
    return cat


def copy_in_results_folders(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    load_data = cfg['load_data']
    # Retrieve the list of file names using the get_files function
    file_names = get_files(config)
    
    # Define the base paths for the source and destination
    base_src_path = Path("/work/wachslab/aredwine3/VAME_working/results")
    base_dest_path = Path("/work/wachslab/aredwine3/VAME_working/results")
    
    # Define the folder to be copied
    folder_to_copy = "VAME_15prcnt_sweep_drawn-sweep-88/kmeans-40"
    folder_to_copy_to = f"VAME_15prcnt_sweep_drawn-sweep-88/{load_data}/kmeans-40/"
    
    # Iterate over the file names and copy the folders
    for file_name in file_names:
        src_folder = base_src_path / file_name / folder_to_copy
        dest_folder = base_dest_path / file_name / folder_to_copy_to
        # Check if the source folder exists
        if src_folder.exists():
            shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
            print(f"Copied folder from {src_folder} to {dest_folder}")
        else:
            print(f"Source folder {src_folder} does not exist.")


def delete_folders_in_results(config, deleting = 'Folder'):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    load_data = cfg['load_data']
    model_name = cfg['model_name']
    
    # Retrieve the list of file names using the get_files function
    file_names = get_files(config)

    folder_to_delete = "community"

    base_path = Path("/work/wachslab/aredwine3/VAME_working/results")
    """
    file = file_names[0]

    folder_path = base_path / file / model_name / folder_to_delete
    
    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)
        print(f"Deleted folder {folder_path}")
    else:
        print(f"Folder {folder_path} does not exist or is not a directory.")
    """
   

    # Iterate over the file names and delete the specified folder
    for file in file_names:
        if deleting == 'Folder':

            folder_path = base_path / file / model_name / load_data / load_data
            # Check if the folder exists before attempting to delete
            if folder_path.exists() and folder_path.is_dir():
                shutil.rmtree(folder_path)
                print(f"Deleted folder {folder_path}")
            else:
                print(f"Folder {folder_path} does not exist or is not a directory.")
        elif deleting == 'File':
            file_to_delete = f"{file}UMAP_LabeledMotifs.png"
            file_path = base_path / file / model_name / load_data / file_to_delete
            if file_path.exists() and file_path.is_file():
                os.remove(file_path)
                print(f"Deleted file {file_path}")
            else:
                print(f"File {file_path} does not exist or is not a file.")


def replace_date_underscores(directory):
    """
    Replaces underscores in date numbers (in the format xx_xx_xx) with hyphens in all filenames in a given directory.
    
    Parameters:
        directory (str): The directory where the files are located.
        
    Returns:
        None
    """
    
    for filename in os.listdir(directory):
        # Search for date pattern xx_xx_xx in the filename
        new_filename = re.sub(r'(\d{2})_(\d{2})_(\d{2})', r'\1-\2-\3', filename)
        
        # Rename the file only if the filename has changed
        if new_filename != filename:
            original_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            os.rename(original_file_path, new_file_path)
            print(f"Renamed {filename} to {new_filename}")


def delete_files(directory, file_extension):
    # Iterate over all subdirectories
    for subdir, dirs, files in os.walk(directory):
        # Find all files with the given extension
        for file in glob.glob(subdir + '/*' + file_extension):
            # Delete the file
            os.remove(file)
            print(f"File {file} has been deleted")


def find_video_files(directory):
    """
    Recursively finds all video files in a given directory.
    
    Parameters:
    - directory (str): The directory to search in.
    
    Returns:
    - List[str]: A list of complete paths to video files.
    """
    video_extensions = ['.mp4', '.avi', '.mkv', '.flv', '.mov', '.wmv']
    video_files = []
    
    # Walk through directory
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in video_extensions):
                full_path = os.path.join(dirpath, filename)
                video_files.append(full_path)
                
    return video_files

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

def create_symlinks(video_dir, new_dir):
    """_summary_
    Args:
    video_dir (str): The path to the directory containing the video files.
    new_dir (str): The path to the directory where the video files will be moved and the symbolic links will be created.

    1. It moves the video files from the original directory (video_dir) to a new directory (new_dir).
    2. It creates symbolic links in the original directory that point to the moved video files in the new directory.
    """
    # Create the new directory if it doesn't exist
    os.makedirs(new_dir, exist_ok=True)

    # Get a list of all video files in the directory
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))  # adjust the extension if needed

    # List of rat names
    rat_names = ['Rat1', 'Rat2', 'Rat3', 'Rat4']

    # Loop through each video file
    for video_path in video_files:
        
        # Get the video filename
        video_filename = os.path.basename(video_path)

        # Move the video file to the new directory
        new_video_path = os.path.join(new_dir, video_filename)
        shutil.move(video_path, new_video_path)

        # Create a symbolic link for each rat name
        for rat_name in rat_names:
            # Path for the symbolic link
            symlink_path = os.path.join(video_dir, os.path.splitext(video_filename)[0] + '_' + rat_name + os.path.splitext(video_filename)[1])
            # Create the symbolic link
            os.symlink(new_video_path, symlink_path)

# Usage:
# create_symlinks('/path/to/videos', '/path/to/new_directory')

def create_new_dirs_and_remove_old(parent_dir):
    # Display a warning message and ask the user if they want to proceed
    print("WARNING: This function will permanently delete the original directories and all files and subdirectories within them.")
    print("Make sure you have a backup of any important data before proceeding.")
    proceed = input("Do you want to proceed? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Operation cancelled.")
        return

    # Get a list of all directories in the parent directory
    dir_names = [name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]

    # List of rat names
    rat_names = ['Rat1', 'Rat2', 'Rat3', 'Rat4']

    # Loop through each directory
    for dir_name in dir_names:
        # Full path to the original directory
        dir_path = os.path.join(parent_dir, dir_name)

        # Create a new directory for each rat name
        for rat_name in rat_names:
            # Path for the new directory
            new_dir_path = os.path.join(parent_dir, dir_name + '_' + rat_name)
            # Create the new directory
            os.makedirs(new_dir_path, exist_ok=True)

        # Delete the original directory
        shutil.rmtree(dir_path)


def update_video_sets_in_config(config_path, video_dir):
    # Get a list of all video files in the directory
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))  # adjust the extension if needed

    # Get the video names (without extension)
    video_names = [os.path.splitext(os.path.basename(video_file))[0] for video_file in video_files]

    # Create a YAML object
    yaml = YAML()

    # Load the existing config data
    with open(config_path, 'r') as file:
        config_data = yaml.load(file)

    # Update the 'video_sets' field
    config_data['video_sets'] = video_names

    # Write the updated config data back to the file
    with open(config_path, 'w') as file:
        yaml.dump(config_data, file)


def print_body_parts(directory):
    # Find the first CSV file in the directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    csv_file = csv_files[0]

    # Read the CSV file
    df = pd.read_csv(csv_file, index_col=0, header=[0,1], nrows=1)

    # Print the body part each index corresponds to
    for i, body_part in enumerate(df.iloc[0]):
        print(f"Index {i} corresponds to {body_part}")



def rearrange_all_csv_columns(directory, body_parts_order):
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    for csv_file in csv_files:
        # Read the CSV file with two header rows and an index column
        df = pd.read_csv(csv_file, header=[0, 1], index_col=0, low_memory=False)

        # Create a new order of columns
        new_columns = []
        for body_part in body_parts_order:
            # Find the columns for this body part
            body_part_columns = [col for col in df.columns if body_part in col[0]]
            new_columns.extend(body_part_columns)

        # Reorder the columns
        df = df[new_columns]

        # Save the DataFrame back to the CSV file with two header rows and an index column
        df.to_csv(csv_file, index=True, header=True)

    return "All files rearranged successfully!"

# Usage:
# rearrange_all_csv_columns('/path/to/directory', ['snout', 'forehand_left', 'forehand_right', 'hindleft', 'hindright', 'tail'])



def multithreaded_copy(src_dir, dest_dir):
    files = os.listdir(src_dir)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Wrap the files list with tqdm for a progress bar
        for file in tqdm(files, desc="Copying files"):
            executor.submit(copy_file, file, src_dir, dest_dir)

def parse_filename(filename):
    # Define the mapping from letter groups to groups
    group_mapping = {
        'A': 'Sham', 'I': 'Sham', 'L': 'Sham', 'T': 'Sham', 'W': 'Sham', 'X': 'Sham',
        'D': 'Injured', 'E': 'Injured', 'F': 'Injured', 'H': 'Injured', 'K': 'Injured', 'O': 'Injured',
        'J': 'Treated', 'M': 'Treated', 'N': 'Treated', 'P': 'Treated', 'S': 'Treated', 'Y': 'Treated',
        'B': 'ABX', 'C': 'ABX', 'G': 'ABX', 'Q': 'ABX', 'R': 'ABX', 'U': 'ABX'
    }

    # Split the filename into parts
    parts = filename.split('_')

    # Print each part and its index
    #for i, part in enumerate(parts):
    #    print(f"{i}: {part}")


    # Extract the study point, letter groups, and rat number
    study_point = parts[1] + '_' + parts[2]
    letter_groups = parts[4]
    #rat_number = parts[-1][-1]

    # Get the number directly following the text "Rat"
    rat_number = re.search(r'Rat(\d)', filename).group(1)

    # Check that letter_groups has the expected length
    if len(letter_groups) != 4:
        print(f"Letter groups: {letter_groups}")
        raise ValueError(f"Unexpected letter group length in filename {filename}")

    if rat_number in ['1', '2']:
        full_group_notation = letter_groups[1]
    elif rat_number in ['3', '4']:
        full_group_notation = letter_groups[3]
    else:
        print(f"Rat number: {rat_number}")
        raise ValueError(f"Unexpected rat number in filename {filename}")
    
    if rat_number in ['1']:
        full_group_notation = full_group_notation + '1'
    elif rat_number in ['2']:
        full_group_notation = full_group_notation + '2'
    elif rat_number in ['3']:
        full_group_notation = full_group_notation + '1'
    elif rat_number in ['4']:
        full_group_notation = full_group_notation + '2'
    else:
        print(f"Rat number: {rat_number}")
        raise ValueError(f"Unexpected rat number in filename {filename}")

    # Determine the group of the animal
    group = group_mapping[full_group_notation[0]]

    return filename, study_point, group, full_group_notation


def categorize_columns(df):
    # Drop the first column of the df
    df = df.drop(df.columns[0], axis=1)
    
    # Initialize lists for each group
    sham_columns = []
    injured_columns = []
    treated_columns = []
    abx_columns = []

    # Iterate over the column names
    for column in df.columns:
        _, _, group, _ = parse_filename(column)
        
        # Append the column name to the appropriate list
        if group == 'Sham':
            sham_columns.append(column)
        elif group == 'Injured':
            injured_columns.append(column)
        elif group == 'Treated':
            treated_columns.append(column)
        elif group == 'ABX':
            abx_columns.append(column)

    return sham_columns, injured_columns, treated_columns, abx_columns

def categorize_fileNames(config):
    files = get_files(config)

    # Initialize lists for each group
    sham_files = []
    injured_files = []
    treated_files = []
    abx_files = []

    # Iterate over the column names
    for file in files:
        _, _, group, _ = parse_filename(file)
        
        # Append the column name to the appropriate list
        if group == 'Sham':
            sham_files.append(file)
        elif group == 'Injured':
            injured_files.append(file)
        elif group == 'Treated':
            treated_files.append(file)
        elif group == 'ABX':
            abx_files.append(file)

    return sham_files, injured_files, treated_files, abx_files



def get_time_point_columns_for_group(group_columns, time_point = "Baseline_1"):
    filtered_columns = []
    
    for column in group_columns:
        parts = column.split('_')
        # Ensure that the column name has enough parts to prevent index errors
        if len(parts) >= 3:
            study_point = parts[1] + '_' + parts[2]
            if study_point == time_point:
                filtered_columns.append(column)
                    
    return filtered_columns


def create_meta_data_df(df):
    
    # Initialize an empty DataFrame for the meta data
    meta_data = pd.DataFrame()
    
    # Initialize lists for each group
    sham = []
    injured = []
    treated = []
    abx = []
    
    sham, injured, treated, abx = categorize_columns(df)

    # Create columns in the meta data dataframe called "Name", "Time_Point", "Group", and "Treatment_State", fill them with NA for now
    meta_data['Name'] = df.columns[1: ]
    meta_data['Animal_ID'] = 'NA'
    meta_data['Time_Point'] = 'NA'
    meta_data['Group'] = 'NA'
    meta_data['Treatment_State'] = 'NA'
    
    # Get the text between the first and third underscore in the column Name, write it in the "Time_Point" column
    meta_data['Time_Point'] = meta_data['Name'].apply(lambda x: x.split('_')[1] + '_' + x.split('_')[2])
    
    time_points = meta_data['Time_Point'].unique()
    
    
    meta_data['Animal_ID'] = meta_data['Name'].apply(lambda x: parse_filename(x)[3])

    
    # Create a dictionary to map column names to their group and time point
    column_group_time_map = {}
    
    # Populate the dictionary
    for group, group_columns in {'Sham': sham, 'Injured': injured, 'Treated': treated, 'ABX': abx}.items():
        for time_point in time_points:
            time_point_columns = get_time_point_columns_for_group(group_columns, time_point)
            for col in time_point_columns:
                column_group_time_map[col] = {'Group': group, 'Time_Point': time_point}

    # Now iterate over the meta_data DataFrame to fill in the "Group" and "Time_Point" columns
    for index, row in meta_data.iterrows():
        col_name = row['Name']  # Replace with the actual column name holding the file names
        if col_name in column_group_time_map:
            meta_data.at[index, 'Group'] = column_group_time_map[col_name]['Group']
            meta_data.at[index, 'Time_Point'] = column_group_time_map[col_name]['Time_Point']
    
    pre_treatment = ['Baseline_1', 'Baseline_2', 'Week_02', 'Week_04', 'Week_06', 'Week_08']
    
    # Time points not in pre_treatment are post_treament
    post_treatment = [time_point for time_point in time_points if time_point not in pre_treatment] 
    
    # If group is Injured treatment state is Injured
    meta_data.loc[(meta_data['Group'] == 'Injured'), 'Treatment_State'] = 'Injured'
    
    # If group is Treated and time point is in pre_treatment, treatment state is Injured
    meta_data.loc[(meta_data['Group'] == 'Treated') & (meta_data['Time_Point'].isin(pre_treatment)), 'Treatment_State'] = 'Injured'
    
    # If group is Treated and time point is in post_treatment, treatment state is Treated
    meta_data.loc[(meta_data['Group'] == 'Treated') & (meta_data['Time_Point'].isin(post_treatment)), 'Treatment_State'] = 'Treated'
    
    # IF group is Sham, treatment state is Sham
    meta_data.loc[(meta_data['Group'] == 'Sham'), 'Treatment_State'] = 'Sham'
    
    # If group is ABX, treatment state is ABX
    meta_data.loc[(meta_data['Group'] == 'ABX'), 'Treatment_State'] = 'ABX'
    
    return meta_data


# Assuming distance_traveled_normality_results is your dictionary
def shapiro_normality_results_to_polars_df(normality_results):
    # Prepare a list to collect rows for the DataFrame
    rows_list = []
    
    for group, time_points in normality_results.items():
        for time_point, (stat, p_value) in time_points.items():
            # Append a dictionary for each row
            rows_list.append({"group": group, "time_point": time_point, "shapiro_stat": stat, "shapiro_p_value": p_value})
    

    # Create a DataFrame from the list of dictionaries
    df = pl.DataFrame(rows_list)
    return df


def shapiro_test(series):
    stat, p_value = scipy.stats.shapiro(series)
    return (stat, p_value)


def check_normality_total_distance_polars(master_df, col_to_test="distance"):
    """
    Calculates the total distance for each individual within each group and time point from a given master dataframe.
    Performs the Shapiro-Wilk test for normality on the total distance values and returns the results.

    Args:
        master_df (DataFrame): The master dataframe containing the data.
        col_to_test (str, optional): The column name in the dataframe to test for normality. 
                                    Defaults to "distance".

    Returns:
        dict: A dictionary containing the Shapiro-Wilk test results for each group and time point. 
              The keys of the dictionary are the group names, and the values are nested dictionaries 
              where the keys are the time points and the values are tuples of the test statistic and p-value.
    """
    # Apply the function to each group and time_point
    normality_results = {}

    # Sum the distances for each individual within each group and time point
    total_distance_by_individual = master_df.groupby(["group", "time_point", "rat_id"]).agg(
        pl.col(col_to_test).sum().alias("total_distance")
    )

    # Iterate over each group
    for group in total_distance_by_individual.get_column("group").unique().to_list():
        normality_results[group] = {}
        # Filter data for the group
        group_data = total_distance_by_individual.filter(pl.col("group") == group)
        
        # Iterate over each time_point
        for time_point in group_data.get_column("time_point").unique().to_list():
            # Filter data for the time_point
            time_point_data = group_data.filter(pl.col("time_point") == time_point)
            
            # Get the list of total distances for the Shapiro-Wilk test
            total_distance_list = time_point_data.get_column("total_distance").to_list()
            
            # Perform Shapiro-Wilk test and store the results if there are enough data points
            if len(total_distance_list) > 3:  # Shapiro-Wilk requires more than 3 values
                normality_results[group][time_point] = shapiro_test(total_distance_list)
            else:
                normality_results[group][time_point] = (None, None)  # Not enough data for the test
    
    return normality_results



def check_normality_group_timepoint_motif(df, meta_data):
    
    normality_results = []
    # number of motifs
    num_motifs = df.shape[0]
    
    # Normality in Each Group and 
    # Iterate over each unique group and time point and motif combination 
    for group in meta_data['Group'].unique():
        for time_point in meta_data['Time_Point'].unique():
            # Filter meta_data for the current group and time point
            filtered_meta = meta_data[(meta_data['Group'] == group) & (meta_data['Time_Point'] == time_point)]

            # Iterate over each motif
            for motif in range(df.shape[0]):
                # Collect data for the current motif from all relevant columns
                motif_data = df.loc[motif, filtered_meta['Name']]
                
                # Fill any NaN values with 0
                motif_data = np.nan_to_num(motif_data)

                # Check if there are enough data points for the test
                if len(motif_data) >= 3:
                    # Perform the Shapiro-Wilk test
                    shapiro_results = scipy.stats.shapiro(motif_data)
                    # Add the results to the DataFrame
                    normality_results.append({'Motif': motif, 'Group': group, 'Time_Point': time_point, 'W': shapiro_results[0], 'p': shapiro_results[1]})
                else:
                    # Handle cases with insufficient data
                    normality_results.append({'Motif': motif, 'Group': group, 'Time_Point': time_point, 'W': None, 'p': None})
    
    return pd.DataFrame(normality_results)


def check_normality_by_group(df, meta_data):
    normality_results = []

    for group in meta_data['Group'].unique():
        filtered_meta = meta_data[meta_data['Group'] == group]
        group_data = df.loc[:, filtered_meta['Name']].values.flatten()
        
        # Fill any NaN values with 0
        group_data = np.nan_to_num(group_data)

        # Check for constant data or NaN values
        if np.all(group_data == group_data[0]) or np.isnan(group_data).any():
            normality_results.append({'Group': group, 'W': None, 'p': None, 'Note': 'Constant data or NaN'})
            continue

        # Check if there are enough data points for the test
        if len(group_data) >= 3:
            # Import the stats module from scipy
            from scipy import stats
            shapiro_results = stats.shapiro(group_data)
            normality_results.append({'Group': group, 'W': shapiro_results[0], 'p': shapiro_results[1]})
        else:
            normality_results.append({'Group': group, 'W': None, 'p': None, 'Note': 'Insufficient data'})

    return pd.DataFrame(normality_results)


def check_normality_by_group_and_time(df, meta_data):
    normality_results = []
    
    for group in meta_data['Group'].unique():
        for time_point in meta_data['Time_Point'].unique():
            # Filter meta_data for the current group and time point
            filtered_meta = meta_data[(meta_data['Group'] == group) & (meta_data['Time_Point'] == time_point)]
            
            # Check if the filtered meta_data is not empty
            if not filtered_meta.empty:
                # Get the corresponding names for filtering the df
                names = filtered_meta['Name'].tolist()
                # Filter the df for the current group and time point
                group_time_data = df[names].values.flatten()
                
                # Handle case where all data may be NaN after filtering
                if group_time_data.size == 0 or np.isnan(group_time_data).all():
                    normality_results.append({
                        'Group': group,
                        'Time_Point': time_point,
                        'W': None,
                        'p': None,
                        'Note': 'No data or all data is NaN'
                    })
                    continue
                
                # Fill any NaN values with 0
                group_time_data = np.nan_to_num(group_time_data)

                # Check for constant data
                if np.all(group_time_data == group_time_data[0]):
                    normality_results.append({
                        'Group': group,
                        'Time_Point': time_point,
                        'W': None,
                        'p': None,
                        'Note': 'Constant data'
                    })
                    continue

                # Check if there are enough data points for the test
                if len(group_time_data) >= 3:
                    shapiro_results = scipy.stats.shapiro(group_time_data)
                    normality_results.append({
                        'Group': group,
                        'Time_Point': time_point,
                        'W': shapiro_results[0],
                        'p': shapiro_results[1]
                    })
                else:
                    normality_results.append({
                        'Group': group,
                        'Time_Point': time_point,
                        'W': None,
                        'p': None,
                        'Note': 'Insufficient data'
                    })
            else:
                # Handle case where the filtered meta_data is empty
                normality_results.append({
                    'Group': group,
                    'Time_Point': time_point,
                    'W': None,
                    'p': None,
                    'Note': 'No matching data for group/time point'
                })

    return pd.DataFrame(normality_results)

def perform_independent_t_test(sample1, sample2):
    # Perform the t-test
    t_stat, p_value = scipy.stats.ttest_ind(sample1, sample2, equal_var=False)
    
    # Calculate Cohen's d for effect size
    n1, n2 = len(sample1), len(sample2)
    s1, s2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    d = (np.mean(sample1) - np.mean(sample2)) / s
    
    return p_value, d



def compared_group_to_sham_ind_t_test(df, col_to_test="distance"):
    # Perform between-group comparisons to Sham
    independent_comparison_results = []
    
    # Calculate the total distance for the Sham group at each time point
    sham_totals = df.filter(pl.col("group") == "Sham").groupby(["time_point", "rat_id"]).agg(
        pl.col(col_to_test).sum().alias("total_distance")
    )
    
    # Iterate over each time point
    for time_point in sham_totals.get_column("time_point").unique().to_list():
        # Get the total distance for the Sham group at this time point
        sham_total_distances = sham_totals.filter(pl.col("time_point") == time_point)["total_distance"].to_numpy()
        
        # Iterate over each group except Sham
        for group in df.get_column("group").unique().to_list():
            if group != "Sham":
                # Calculate the total distance for the current group at this time point
                group_totals = df.filter((pl.col("group") == group) & (pl.col("time_point") == time_point)).groupby(["rat_id"]).agg(
                    pl.col(col_to_test).sum().alias("total_distance")
                )
                
                # Get the total distance for the current group at this time point
                group_total_distances = group_totals["total_distance"].to_numpy()
                
                # Perform the independent t-test between the Sham group and the current group
                p_value, d = perform_independent_t_test(sham_total_distances, group_total_distances)
                independent_comparison_results.append((group, time_point, p_value, d))
    
    return independent_comparison_results


def perform_paired_t_test(before, after):
    # Perform the paired t-test
    t_stat, p_value = scipy.stats.ttest_rel(before, after)
    # Calculate Cohen's d for effect size
    d = (np.mean(after) - np.mean(before)) / np.std(after - before, ddof=1)
    return p_value, d


def within_group_across_total_distance_time_paired_t_test(df, baseline_data, col_to_test="distance"):
    
    # Perform within-group comparisons
    paired_comparison_results = []
    for group in df.get_column("group").unique().to_list():
        
        # Get baseline total distances for the group
        baseline_totals = baseline_data.filter(pl.col("group") == group).groupby("rat_id").agg(
            pl.col(col_to_test).sum().alias("total_baseline_distance")
        )
        
        # Perform comparisons for each time point
        for time_point in df.get_column("time_point").unique().to_list():
            if time_point not in ["Baseline_1", "Baseline_2"]:
                # Get total distances for the current time point
                time_point_totals = df.filter((pl.col("group") == group) & (pl.col("time_point") == time_point)).groupby("rat_id").agg(
                    pl.col(col_to_test).sum().alias("total_time_point_distance")
                )
                
                # Ensure we have matching rats before performing paired t-test
                if baseline_totals.shape[0] == time_point_totals.shape[0]:
                    # Perform the paired t-test
                    p_value, d = perform_paired_t_test(
                        baseline_totals["total_baseline_distance"].to_numpy(),
                        time_point_totals["total_time_point_distance"].to_numpy()
                    )
                    paired_comparison_results.append((group, time_point, p_value, d))
    
    return paired_comparison_results


def paired_t_test_to_polars_df(t_test_results):
    # Prepare a list to collect dictionaries for the DataFrame
    rows_list = []
    
    # Iterate over the list of tuples in the t_test_results
    for result in t_test_results:
        group, time_point, p_value, d = result
        # Append a dictionary for each row
        rows_list.append({"group": group, "time_point": time_point, "p_value": p_value, "effect_size": d})
    
    # Create a DataFrame from the list of dictionaries
    df = pl.DataFrame(rows_list)
    return df



def melt_df(df, meta_data):
    
    # Add the name Motif to the unnamed column
    df = df.rename(columns={df.columns[0]: 'Motif'})
    
    df_long = pd.melt(df, id_vars=['Motif'], value_vars=meta_data['Name']) 
    
    df_long['Animal_ID'] = df_long['variable'].apply(lambda x: parse_filename(x)[3])
    df_long['Time_Point'] = df_long['variable'].apply(lambda x: parse_filename(x)[1])
    
    df_long['Group'] = df_long['variable'].map(meta_data.set_index('Name')['Group'])
    
    df_long['Treatment_State'] = df_long['variable'].map(meta_data.set_index('Name')['Treatment_State'])

    df_long = df_long.drop('variable', axis=1)
    
    df_long = df_long[['Animal_ID', 'Time_Point', 'Group', 'Treatment_State', 'Motif', 'value']]
    
    return df_long
    

""" NORMALIZATION FUNCTIONS """

def log_normalize_values(df_long):
    if (df_long['value'] <= 0).any():
        # Handle non-positive values. Here we add a small constant (e.g., 1) to all values.
        df_long['log_value'] = np.log(df_long['value'] + 1)
    else:
        # If all values are positive, we can directly apply the log transformation
        df_long['log_value'] = np.log(df_long['value'])

    return df_long


def normalize_to_baseline_log(df_long):
    baseline_time_points = ['Baseline_1', 'Baseline_2']
    
    # Calculate the baseline mean for log-transformed values
    if (df_long['value'] <= 0).any():
        df_long['log_value'] = np.log(df_long['value'] + 1)  # Adding 1 to avoid log(0)
    else:
        df_long['log_value'] = np.log(df_long['value'])
    
      # Calculate the baseline mean for log-transformed values only for baseline time points
    baseline_means = df_long[df_long['Time_Point'].isin(baseline_time_points)].groupby(['Motif', 'Animal_ID'])['log_value'].mean().reset_index(name='Baseline_Log_Mean')
    
    # Merge the baseline means back onto the original dataframe
    df_long = df_long.merge(baseline_means, on=['Motif', 'Animal_ID'], how='left')
    
    # Normalize the log-transformed values to the log-transformed baseline mean
    df_long['Log_Normalized_Value'] = df_long['log_value'] - df_long['Baseline_Log_Mean']
    
    # When the Time_Point is Baseline_1 or Baseline_2, the Log_Normalized_Value should be 0
    df_long.loc[df_long['Time_Point'].isin(baseline_time_points), 'Log_Normalized_Value'] = 0
    
    return df_long

            
def normalize_to_baseline(df_long):
    
    baseline_time_points = ['Baseline_1', 'Baseline_2']    
    
    df_long['Baseline_Mean'] = df_long.loc[df_long['Time_Point'].isin(baseline_time_points)].groupby(['Motif', 'Animal_ID'])['value'].transform('mean')
    
    # For each animal, motif, and timepoint, ensure the baseline mean is filled in
    df_long['Baseline_Mean'] = df_long.groupby(['Motif', 'Animal_ID'])['Baseline_Mean'].transform(lambda x: x.fillna(x.mean()))
    
    # Normalize the values to the baseline mean
    df_long['Normalized_Value'] = df_long['value'] / df_long['Baseline_Mean']
    
    # When the Time_Point is Baseline_1 or Baseline_2, the Normalized_Value should be 1
    df_long.loc[df_long['Time_Point'].isin(baseline_time_points), 'Normalized_Value'] = 1
    
    return df_long

def normalize_to_baseline_sham_log(df_long):

    # Calculate the baseline mean for log-transformed values
    if (df_long['value'] <= 0).any():
        df_long['log_value'] = np.log(df_long['value'] + 1)  # Adding 1 to avoid log(0)
    else:
        df_long['log_value'] = np.log(df_long['value'])

    # Step 2: Filter Sham Group Data and compute log-transformed means
    sham_df = df_long[df_long['Group'] == 'Sham']
    sham_means = sham_df.groupby(['Time_Point', 'Motif'])['log_value'].mean().reset_index(name='Sham_Log_Mean')

    # Step 3: Merge Sham Log Means with Original Data
    df_long = df_long.merge(sham_means, on=['Time_Point', 'Motif'], how='left')

    # Step 4: Normalize Log-Transformed Values
    df_long['Log_Normalized_Value'] = df_long['log_value'] - df_long['Sham_Log_Mean']

    # Set Log_Normalized_Value to 0 for the sham group to represent no change
    df_long.loc[df_long['Group'] == 'Sham', 'Log_Normalized_Value'] = 0
    
    return df_long


def normalize_to_baseline_sham(df_long):

    # Step 1: Filter Sham Group Data
    sham_df = df_long[df_long['Group'] == 'Sham']

    # Step 2: Compute Sham Group Means
    sham_means = sham_df.groupby(['Time_Point', 'Motif'])['value'].mean().reset_index(name='Sham_Mean')

    # Step 3: Merge Sham Means with Original Data
    df_long = df_long.merge(sham_means, on=['Time_Point', 'Motif'], how='left')

    # Step 4: Normalize Values
    df_long['Normalized_Value'] = df_long['value'] / df_long['Sham_Mean']

    # Set Sham_Normalized_Value to 1 for the sham group
    df_long.loc[df_long['Group'] == 'Sham', 'Normalized_Value'] = 1
    
    return df_long


    
    
def calculate_mean_and_sd(df_long, normalization=True, type = "Group"):
        if type == "Group":
            if normalization:
                # Calculate the mean and standard deviation for each motif, time point, and group
                stats = df_long.groupby(['Motif', 'Time_Point', 'Group'])['Normalized_Value'].agg(['mean', 'std']).reset_index()
            else:
                stats = df_long.groupby(['Motif', 'Time_Point', 'Group'])['value'].agg(['mean', 'std']).reset_index()
            
        elif type == "State":
            if normalization:
                # Calculate the mean and standard deviation for each motif, time point, and group
                stats = df_long.groupby(['Motif', 'Time_Point', 'Treatment_State'])['Normalized_Value'].agg(['mean', 'std']).reset_index()
            else:
                stats = df_long.groupby(['Motif', 'Time_Point', 'Treatment_State'])['value'].agg(['mean', 'std']).reset_index()

            # Sort the Time_Point values
            sorted_time_points = sorted(stats['Time_Point'].unique())
            stats['Time_Point'] = pd.Categorical(stats['Time_Point'], categories=sorted_time_points, ordered=True)
        
        return stats


def write_video_info_to_csv(video_dir, output_csv):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Study Point", "Group", "Full Group Notation"])  # Write header
        for filename in os.listdir(video_dir):
            if filename.endswith('.mp4'):  # or whatever video format you're using
                filename, study_point, group, full_group_notation = parse_filename(filename)
                writer.writerow([filename, study_point, group, full_group_notation])  # Write data
                

def group_counts(videos):
    counts = defaultdict(lambda: defaultdict(int))
    for video in videos:
        _, study_point, group, _ = parse_filename(video)
        counts[study_point][group] += 1
    return counts


def rename_files_and_dirs(directory):
    for root, dirs, files in os.walk(directory):
        for name in files + dirs:
            new_name = re.sub(r'(\d{2})_(\d{2})_(\d{2})', r'\1-\2-\3', name)
            if new_name != name:
                os.rename(os.path.join(root, name), os.path.join(root, new_name))


def select_videos(csv_file, percentage):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Group the data by 'Group' and 'Study Point'
    grouped = df.groupby(['Group', 'Study Point'])

    # Initialize an empty list to store the selected videos
    selected_videos = []

    # Loop over the groups
    for name, group in grouped:
        # Calculate the number of videos to select from this group
        n = round(len(group) * percentage / 100)

        print(name, n)

        # Randomly select 'n' videos from this group
        selected = group.sample(n)


        # Append the selected videos to 'selected_videos'
        selected_videos.append(selected)

    # Concatenate all the selected videos into a single DataFrame
    selected_videos = pd.concat(selected_videos)

    # Return the names of the selected videos
    return selected_videos['Filename']
