import concurrent.futures
import csv
import glob
import os
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy
import seaborn as sns
from icecream import ic
from matplotlib.pylab import f
from ruamel.yaml import YAML
from scipy.stats import kruskal, ttest_ind
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

import vame.custom.ALR_analysis as ana
import vame.custom.ALR_kinematics as kin
from vame.util.auxiliary import read_config

# Set the Matplotlib backend based on the environment.
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use(
        "Agg"
    )  # Use this backend for headless environments (e.g., Google Colab, some remote servers)
else:
    matplotlib.use("Qt5Agg")  # Use this backend for environments with a display server
        
def process_file(file_data):
    file, i, path_to_file, dlc_data_type, fps, labels_list = file_data

    filename, time_point, group, full_group_notation = parse_filename(file)
    labels = labels_list[i]
    data = kin.get_dlc_file(path_to_file, dlc_data_type, file)
    rat = kin.get_dat_rat(data)
    centroid_x, centroid_y = kin.calculate_centroid(data)
    in_center = kin.is_dat_rat_in_center(data)
    distance_cm = kin.distance_traveled(data)
    rat_speed = kin.calculate_speed_with_spline(
        data, fps, window_size=5, pixel_to_cm=0.215
    )

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

    temp_df = pl.DataFrame(
        {
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
            "time_point": [time_point] * len(labels),
        }
    )
    return temp_df


def create_andOR_get_master_df(config, fps=30, create_new_df=False, df_kind="polars"):
    files = get_files(config)
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg["model_name"]
    n_cluster = cfg["n_cluster"]
    parameterization = cfg["parameterization"]
    path_to_file = cfg["project_path"]
    dlc_data_type = input("Were your DLC .csv files originally multi-animal? (y/n): ")
    labels_list = ana.get_labels(cfg, files, model_name, n_cluster)

    df = pl.DataFrame(
        {
            "file_name": pl.Series([], dtype=pl.Utf8),
            "frame": pl.Series([], dtype=pl.Int64),
            "motif": pl.Series([], dtype=pl.Int64),
            "centroid_x": pl.Series([], dtype=pl.Float32),
            "centroid_y": pl.Series([], dtype=pl.Float32),
            "torso_angle": pl.Series([], dtype=pl.Float32),
            "torso_length": pl.Series([], dtype=pl.Float32),
            "in_center": pl.Series([], dtype=pl.Float32),
            "distance": pl.Series([], dtype=pl.Float32),
            "speed": pl.Series([], dtype=pl.Float32),
            "rat": pl.Series([], dtype=pl.Utf8),
            "rat_id": pl.Series([], dtype=pl.Utf8),
            "group": pl.Series([], dtype=pl.Utf8),
            "time_point": pl.Series([], dtype=pl.Utf8),
        }
    )

    if parameterization == "hmm":
        df_path = os.path.join(
            path_to_file,
            "results",
            f"all_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv",
        )
    else:
        df_path = os.path.join(
            path_to_file, "results", f"all_sequences_{parameterization}-{n_cluster}.csv"
        )

    if not create_new_df and os.path.exists(df_path):
        if df_kind == "polars":
            df = pl.read_csv(df_path)
        elif df_kind == "pandas":
            df = pd.read_csv(df_path)
    else:
        print("Creating new master data frame...")
        cpu_count = os.cpu_count()
        if cpu_count is not None and cpu_count > 1:
            # Prepare the data for processing
            file_data_list = [
                (file, i, path_to_file, dlc_data_type, fps, labels_list)
                for i, file in enumerate(files)
            ]

            # Process the files in parallel
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(process_file, file_data_list))

            # Concatenate all DataFrames into one
            df = pl.concat(results)
        else:
            for i, file in enumerate(files):
                filename, time_point, group, full_group_notation = parse_filename(file)
                labels = labels_list[i]
                data = kin.get_dlc_file(path_to_file, dlc_data_type, file)
                rat = kin.get_dat_rat(data)
                
                #! Need to only keep data points on rats torso...
                columns_to_drop = [('Nose', 'x'), ('Nose', 'y'), ('Nose', 'likelihood'),
                   ('Caudal_Skull_Point', 'x'), ('Caudal_Skull_Point', 'y'), ('Caudal_Skull_Point', 'likelihood'),
                   ('LeftEar', 'x'), ('LeftEar', 'y'), ('LeftEar', 'likelihood'),
                   ('RightEar', 'x'), ('RightEar', 'y'), ('RightEar', 'likelihood')]

                data_Torso = data.drop(columns=columns_to_drop)
                
                centroid_x, centroid_y = kin.calculate_centroid(data_Torso)
                torso_angle = kin.calculate_torso_angle(data_Torso)
                torso_length = kin.calculate_torso_length(data_Torso)
                in_center = kin.is_dat_rat_in_center(data_Torso)
                distance_cm = kin.distance_traveled(data_Torso)
                rat_speed = kin.calculate_speed_with_spline(
                    data_Torso, fps, window_size=5, pixel_to_cm=0.215
                )

                # Calculate the number of elements to trim
                trim_length = len(centroid_x) - len(labels)

                # Trim the centroid lists
                centroid_x_trimmed = centroid_x[trim_length:]
                centroid_y_trimmed = centroid_y[trim_length:]
                torso_angle_trimmed = torso_angle[trim_length:]
                torso_length_trimmed = torso_length[trim_length:]
                in_center_trimmed = in_center[trim_length:]
                distance_trimmed = distance_cm[trim_length:]
                rat_speed_trimmed = rat_speed[trim_length:]

                motif_series = pl.Series(labels).cast(pl.Int64)
                centroid_x_series = pl.Series(centroid_x_trimmed).cast(pl.Float32)
                centroid_y_series = pl.Series(centroid_y_trimmed).cast(pl.Float32)
                torso_angle_series = pl.Series(torso_angle_trimmed).cast(pl.Float32)
                torso_length_series = pl.Series(torso_length_trimmed).cast(pl.Float32)
                in_center_series = pl.Series(in_center_trimmed).cast(pl.Float32)
                distance_series = pl.Series(distance_trimmed).cast(pl.Float32)
                speed_series = pl.Series(rat_speed_trimmed).cast(pl.Float32)

                # Create a new DataFrame with the labels and the rat value for this file
                temp_df = pl.DataFrame(
                    {
                        "file_name": [file] * len(labels),
                        "frame": list(range(len(labels))),
                        "motif": motif_series,
                        "centroid_x": centroid_x_series,
                        "centroid_y": centroid_y_series,
                        "torso_angle": torso_angle_series,
                        "torso_length": torso_length_series,
                        "in_center": in_center_series,
                        "distance": distance_series,
                        "speed": speed_series,
                        "rat": [rat] * len(labels),
                        "rat_id": [full_group_notation] * len(labels),
                        "group": [group] * len(labels),
                        "time_point": [time_point] * len(labels),
                    }
                )

                temp_df = temp_df.with_columns(
                    [
                        pl.col("frame").cast(pl.Int64),
                    ]
                )

                # Concatenate the new DataFrame with the existing one
                df = pl.concat([df, temp_df])

        if parameterization == "hmm":
            df.write_csv(
                os.path.join(
                    path_to_file,
                    "results",
                    f"all_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv",
                )
            )
        else:
            df.write_csv(
                os.path.join(
                    path_to_file,
                    "results",
                    f"all_sequences_{parameterization}-{n_cluster}.csv",
                )
            )

    return df


def group_motifs_by_cluster(clustering):
    cluster_dict = {}
    for motif, cluster in enumerate(clustering):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(motif)
    return cluster_dict


def assign_clusters(df, clustered_motifs):
    # Invert the clustered_motifs dictionary to map motifs to their cluster
    motif_to_cluster = {
        motif: cluster
        for cluster, motifs in clustered_motifs.items()
        for motif in motifs
    }

    # Map the 'Motif' column to the 'Cluster' column using the motif_to_cluster mapping
    df["Cluster"] = df["Motif"].map(motif_to_cluster)

    return df


def get_files(config: str) -> list:
    """
    Retrieve a list of files based on the configuration provided.

    Args:
        config (str): The path to the configuration file.

    Returns:
        list: A list of file names based on the configuration.
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)

    files = []
    if cfg["all_data"] == "No" or cfg["all_data"] == "no":
        all_flag = input(
            "Do you want to get files for your entire dataset? \n"
            "If you only want to use a specific dataset type filename: \n"
            "yes/no/filename "
        )
    else:
        all_flag = "yes"

    if all_flag == "yes" or all_flag == "Yes":
        for file in cfg["video_sets"]:
            files.append(file)

    elif all_flag == "no" or all_flag == "No":
        for file in cfg["video_sets"]:
            use_file = input("Do you want to quantify " + file + "? yes/no: ")
            if use_file == "yes":
                files.append(file)
            if use_file == "no":
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
        futures = [
            executor.submit(
                copy_file,
                os.path.basename(file),
                os.path.dirname(file),
                destination_folder,
            )
            for file in files
        ]  # noqa: E501

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
    project_path = cfg["project_path"]
    n_cluster = cfg["n_cluster"]
    model_name = cfg["model_name"]
    parameterization = cfg["parameterization"]
    hmm_iters = cfg.get("hmm_iters", 0)
    load_data = cfg["load_data"]

    if not files:
        files = []
        files = get_files(config)

    cat = pd.DataFrame()
    for file in files:
        if legacy:
            arr = np.load(
                os.path.join(
                    project_path,
                    "results/"
                    + file
                    + "/VAME_NPW/kmeans-"
                    + str(n_cluster)
                    + "/behavior_quantification/motif_usage.npy",
                )
            )
        elif not legacy:
            if parameterization == "hmm":
                arr = np.load(
                    os.path.join(
                        project_path,
                        "results",
                        file,
                        model_name,
                        load_data,
                        parameterization + "-" + str(n_cluster) + "-" + str(hmm_iters),
                        "motif_usage_" + file + ".npy",
                    )
                )
            else:
                arr = np.load(
                    os.path.join(
                        project_path,
                        "results",
                        file,
                        model_name,
                        load_data,
                        parameterization + "-" + str(n_cluster),
                        "motif_usage_" + file + ".npy",
                    )
                )
        df = pd.DataFrame(arr, columns=[file])
        cat = pd.concat([cat, df], axis=1)
    if save:
        if parameterization == "hmm":
            cat.to_csv(
                os.path.join(
                    project_path,
                    f"CombinedMotifUsage_{parameterization}-{n_cluster}-{hmm_iters}.csv",
                )
            )
        else:
            cat.to_csv(
                os.path.join(
                    project_path,
                    f"CombinedMotifUsage_{parameterization}-{n_cluster}.csv",
                )
            )
    return cat


def copy_in_results_folders(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    load_data = cfg["load_data"]
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


def delete_folders_in_results(config, deleting="Folder"):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    load_data = cfg["load_data"]
    model_name = cfg["model_name"]

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
        if deleting == "Folder":
            folder_path = base_path / file / model_name / load_data / load_data
            # Check if the folder exists before attempting to delete
            if folder_path.exists() and folder_path.is_dir():
                shutil.rmtree(folder_path)
                print(f"Deleted folder {folder_path}")
            else:
                print(f"Folder {folder_path} does not exist or is not a directory.")
        elif deleting == "File":
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
        new_filename = re.sub(r"(\d{2})_(\d{2})_(\d{2})", r"\1-\2-\3", filename)

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
        for file in glob.glob(subdir + "/*" + file_extension):
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
    video_extensions = [".mp4", ".avi", ".mkv", ".flv", ".mov", ".wmv"]
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
    csvs = sorted(glob.glob(os.path.join(csv_files_path, "*.csv*")))

    # Loop through each csv file
    for csv in csvs:
        # Read the csv file into a pandas DataFrame
        fname = pd.read_csv(csv, header=[0, 1, 2], index_col=0, skiprows=1)
        # Get a list of unique individuals from the DataFrame columns
        individuals = fname.columns.get_level_values("individuals").unique()
        # Loop through each individual
        for ind in individuals:
            # Create a temporary DataFrame for the current individual
            fname_temp = fname[ind]
            # Define the path for the new csv file
            fname_temp_path = os.path.splitext(csv)[0] + "_" + ind + ".csv"
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
    video_files = glob.glob(
        os.path.join(video_dir, "*.mp4")
    )  # adjust the extension if needed

    # List of rat names
    rat_names = ["Rat1", "Rat2", "Rat3", "Rat4"]

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
            symlink_path = os.path.join(
                video_dir,
                os.path.splitext(video_filename)[0]
                + "_"
                + rat_name
                + os.path.splitext(video_filename)[1],
            )
            # Create the symbolic link
            os.symlink(new_video_path, symlink_path)


# Usage:
# create_symlinks('/path/to/videos', '/path/to/new_directory')


def create_new_dirs_and_remove_old(parent_dir):
    # Display a warning message and ask the user if they want to proceed
    print(
        "WARNING: This function will permanently delete the original directories and all files and subdirectories within them."
    )
    print("Make sure you have a backup of any important data before proceeding.")
    proceed = input("Do you want to proceed? (yes/no): ")
    if proceed.lower() != "yes":
        print("Operation cancelled.")
        return

    # Get a list of all directories in the parent directory
    dir_names = [
        name
        for name in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, name))
    ]

    # List of rat names
    rat_names = ["Rat1", "Rat2", "Rat3", "Rat4"]

    # Loop through each directory
    for dir_name in dir_names:
        # Full path to the original directory
        dir_path = os.path.join(parent_dir, dir_name)

        # Create a new directory for each rat name
        for rat_name in rat_names:
            # Path for the new directory
            new_dir_path = os.path.join(parent_dir, dir_name + "_" + rat_name)
            # Create the new directory
            os.makedirs(new_dir_path, exist_ok=True)

        # Delete the original directory
        shutil.rmtree(dir_path)


def update_video_sets_in_config(config_path, video_dir):
    # Get a list of all video files in the directory
    video_files = glob.glob(
        os.path.join(video_dir, "*.mp4")
    )  # adjust the extension if needed

    # Get the video names (without extension)
    video_names = [
        os.path.splitext(os.path.basename(video_file))[0] for video_file in video_files
    ]

    # Create a YAML object
    yaml = YAML()

    # Load the existing config data
    with open(config_path, "r") as file:
        config_data = yaml.load(file)

    # Update the 'video_sets' field
    config_data["video_sets"] = video_names

    # Write the updated config data back to the file
    with open(config_path, "w") as file:
        yaml.dump(config_data, file)


def print_body_parts(directory):
    # Find the first CSV file in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    csv_file = csv_files[0]

    # Read the CSV file
    df = pd.read_csv(csv_file, index_col=0, header=[0, 1], nrows=1)

    # Print the body part each index corresponds to
    for i, body_part in enumerate(df.iloc[0]):
        print(f"Index {i} corresponds to {body_part}")


def rearrange_all_csv_columns(directory, body_parts_order):
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

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
        "A": "Sham",
        "I": "Sham",
        "L": "Sham",
        "T": "Sham",
        "W": "Sham",
        "X": "Sham",
        "D": "Injured",
        "E": "Injured",
        "F": "Injured",
        "H": "Injured",
        "K": "Injured",
        "O": "Injured",
        "J": "Treated",
        "M": "Treated",
        "N": "Treated",
        "P": "Treated",
        "S": "Treated",
        "Y": "Treated",
        "B": "ABX",
        "C": "ABX",
        "G": "ABX",
        "Q": "ABX",
        "R": "ABX",
        "U": "ABX",
    }

    # Split the filename into parts
    parts = filename.split("_")

    # Print each part and its index
    # for i, part in enumerate(parts):
    #    print(f"{i}: {part}")

    # Extract the study point, letter groups, and rat number
    study_point = parts[1] + "_" + parts[2]
    letter_groups = parts[4]
    # rat_number = parts[-1][-1]

    # Get the number directly following the text "Rat"
    rat_number = re.search(r"Rat(\d)", filename).group(1)

    # Check that letter_groups has the expected length
    if len(letter_groups) != 4:
        print(f"Letter groups: {letter_groups}")
        raise ValueError(f"Unexpected letter group length in filename {filename}")

    if rat_number in ["1", "2"]:
        full_group_notation = letter_groups[1]
    elif rat_number in ["3", "4"]:
        full_group_notation = letter_groups[3]
    else:
        print(f"Rat number: {rat_number}")
        raise ValueError(f"Unexpected rat number in filename {filename}")

    if rat_number in ["1"]:
        full_group_notation = full_group_notation + "1"
    elif rat_number in ["2"]:
        full_group_notation = full_group_notation + "2"
    elif rat_number in ["3"]:
        full_group_notation = full_group_notation + "1"
    elif rat_number in ["4"]:
        full_group_notation = full_group_notation + "2"
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
        if group == "Sham":
            sham_columns.append(column)
        elif group == "Injured":
            injured_columns.append(column)
        elif group == "Treated":
            treated_columns.append(column)
        elif group == "ABX":
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
        if group == "Sham":
            sham_files.append(file)
        elif group == "Injured":
            injured_files.append(file)
        elif group == "Treated":
            treated_files.append(file)
        elif group == "ABX":
            abx_files.append(file)

    return sham_files, injured_files, treated_files, abx_files


def get_time_point_columns_for_group(group_columns, time_point="Baseline_1"):
    filtered_columns = []

    for column in group_columns:
        parts = column.split("_")
        # Ensure that the column name has enough parts to prevent index errors
        if len(parts) >= 3:
            study_point = parts[1] + "_" + parts[2]
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
    meta_data["File_Name"] = df.columns[1:]
    meta_data["Animal_ID"] = "NA"
    meta_data["Time_Point"] = "NA"
    meta_data["Group"] = "NA"
    meta_data["Treatment_State"] = "NA"

    # Get the text between the first and third underscore in the column Name, write it in the "Time_Point" column
    meta_data["Time_Point"] = meta_data["File_Name"].apply(
        lambda x: x.split("_")[1] + "_" + x.split("_")[2]
    )

    time_points = meta_data["Time_Point"].unique()

    meta_data["Animal_ID"] = meta_data["File_Name"].apply(
        lambda x: parse_filename(x)[3]
    )

    # Create a dictionary to map column names to their group and time point
    column_group_time_map = {}

    # Populate the dictionary
    for group, group_columns in {
        "Sham": sham,
        "Injured": injured,
        "Treated": treated,
        "ABX": abx,
    }.items():
        for time_point in time_points:
            time_point_columns = get_time_point_columns_for_group(
                group_columns, time_point
            )
            for col in time_point_columns:
                column_group_time_map[col] = {"Group": group, "Time_Point": time_point}

    # Now iterate over the meta_data DataFrame to fill in the "Group" and "Time_Point" columns
    for index, row in meta_data.iterrows():
        col_name = row[
            "File_Name"
        ]  # Replace with the actual column name holding the file names
        if col_name in column_group_time_map:
            meta_data.at[index, "Group"] = column_group_time_map[col_name]["Group"]
            meta_data.at[index, "Time_Point"] = column_group_time_map[col_name][
                "Time_Point"
            ]

    pre_treatment = [
        "Baseline_1",
        "Baseline_2",
        "Week_02",
        "Week_04",
        "Week_06",
        "Week_08",
    ]

    # Time points not in pre_treatment are post_treament
    post_treatment = [
        time_point for time_point in time_points if time_point not in pre_treatment
    ]

    # If group is Injured treatment state is Injured
    meta_data.loc[(meta_data["Group"] == "Injured"), "Treatment_State"] = "Injured"

    # If group is Treated and time point is in pre_treatment, treatment state is Injured
    meta_data.loc[
        (meta_data["Group"] == "Treated")
        & (meta_data["Time_Point"].isin(pre_treatment)),
        "Treatment_State",
    ] = "Injured"

    # If group is Treated and time point is in post_treatment, treatment state is Treated
    meta_data.loc[
        (meta_data["Group"] == "Treated")
        & (meta_data["Time_Point"].isin(post_treatment)),
        "Treatment_State",
    ] = "Treated"

    # IF group is Sham, treatment state is Sham
    meta_data.loc[(meta_data["Group"] == "Sham"), "Treatment_State"] = "Sham"

    # If group is ABX, treatment state is ABX
    meta_data.loc[(meta_data["Group"] == "ABX"), "Treatment_State"] = "ABX"

    return meta_data


def melt_df(df, meta_data):
    # Add the name Motif to the unnamed column
    df = df.rename(columns={df.columns[0]: "Motif"})

    df_long = pd.melt(df, id_vars=["Motif"], value_vars=meta_data["File_Name"])

    df_long["Animal_ID"] = df_long["variable"].apply(lambda x: parse_filename(x)[3])
    df_long["Time_Point"] = df_long["variable"].apply(lambda x: parse_filename(x)[1])

    df_long["Group"] = df_long["variable"].map(
        meta_data.set_index("File_Name")["Group"]
    )

    df_long["Treatment_State"] = df_long["variable"].map(
        meta_data.set_index("File_Name")["Treatment_State"]
    )

    df_long = df_long.drop("variable", axis=1)

    df_long = df_long[
        ["Animal_ID", "Time_Point", "Group", "Treatment_State", "Motif", "value"]
    ]

    return df_long



def write_video_info_to_csv(video_dir, output_csv):
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Filename", "Study Point", "Group", "Full Group Notation"]
        )  # Write header
        for filename in os.listdir(video_dir):
            if filename.endswith(".mp4"):  # or whatever video format you're using
                filename, study_point, group, full_group_notation = parse_filename(
                    filename
                )
                writer.writerow(
                    [filename, study_point, group, full_group_notation]
                )  # Write data


def group_counts(videos):
    counts = defaultdict(lambda: defaultdict(int))
    for video in videos:
        _, study_point, group, _ = parse_filename(video)
        counts[study_point][group] += 1
    return counts


def rename_files_and_dirs(directory):
    for root, dirs, files in os.walk(directory):
        for name in files + dirs:
            new_name = re.sub(r"(\d{2})_(\d{2})_(\d{2})", r"\1-\2-\3", name)
            if new_name != name:
                os.rename(os.path.join(root, name), os.path.join(root, new_name))


def select_videos(csv_file, percentage):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Group the data by 'Group' and 'Study Point'
    grouped = df.groupby(["Group", "Study Point"])

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
    return selected_videos["Filename"]
