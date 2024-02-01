import sys
import os

# Get the parent directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)

# Get the grandparent directory, which should contain the 'vame' directory
grandparent_dir = os.path.dirname(os.path.dirname(script_dir))

# Add the grandparent directory to sys.path
sys.path.append(grandparent_dir)

# Add the parent directory to sys.path
#sys.path.append(parent_dir)

from typing import List, Tuple, Dict
from vame.util.auxiliary import read_config

from typing import Union
from polars import DataFrame
import glob
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter
from tqdm import trange
from vame.util import auxiliary as aux
import re
import csv
import pandas as pd
import polars as pl
from pathlib import Path
import cv2 as cv
from cv2 import VideoCapture
import os
import glob
import numpy as np
import vame.analysis.community_analysis as ana
import vame.custom.ALR_helperFunctions as AlHf
import vame.custom.ALR_kinematics as kin
import csv
from concurrent.futures import ProcessPoolExecutor
from icecream import ic
import tqdm
from tqdm import tqdm

def consecutive(data, stepsize=1):
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def count_headers(file_path):
    num_headers = 0
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # Assuming headers contain only non-numeric values
            if all(not x.replace('.', '', 1).isdigit() for x in row if x):
                num_headers += 1
            else:
                break
    return num_headers

def extract_pose_data(path_to_file, projectPath, file, extractData):
    """
    Extracts pose data from a CSV file if extractData is True.

    Args:
        path_to_file (str): Path to the directory where the 'dlcPoseData' folder will be created.
        projectPath (str): Path to the project directory.
        file (str): The base name of the file to process.
        extractData (bool): Flag to determine whether to extract data.

    Returns:
        pd.DataFrame or None: Returns a DataFrame with the pose data if extractData is True and the file exists, otherwise None.
    """
    if extractData:
        # Ensure the 'dlcPoseData' directory exists
        dlc_pose_data_path = os.path.join(path_to_file, 'dlcPoseData')
        if not os.path.exists(dlc_pose_data_path):
            os.mkdir(dlc_pose_data_path)

        # Find the data file
        dataFile_pattern = os.path.join(projectPath, 'videos', 'pose_estimation', file + '*.csv')
        dataFile_list = glob.glob(dataFile_pattern)
        dataFile = dataFile_list[0] if dataFile_list else None

        # If the data file exists, read it with the appropriate headers
        data = None  # Initialize the "data" variable with a default value of None
        if dataFile:
            header_check = count_headers(dataFile)

            if header_check == 2:
                data = pd.read_csv(dataFile, index_col=0, header=[0, 1])
            elif header_check == 3:
                data = pd.read_csv(dataFile, index_col=0, header=[0, 1, 2])
        return data
    return None


def get_cluster_vid(cfg, path_to_file, file, n_cluster, videoType, flag, fps=30, bins=6, cluster_method='kmeans', extractData=False, symlinks=False):
    """This creates motif videos based on the longest sequences of each motif, rather than the first frames in that motif (the default in VAME).
    You can limit the length of each sequence used with 'bins', this parameter sets the minimum number of distinct examples that will be sampled
    in the video (if that many examples exist).

    Parameters
    ----------
    cfg : dict
        VAME config dictionary.
    path_to_file : str
        Path to video file to process (base directory without filename).
    file : str
        Name of file to process within path_to_file (without extension).
    n_cluster : int
        Number of clusters (from config file).
    flag : str
        'motif' or 'community', depending on type of video being created. Only tested with 'motif' so far.
    videoType: str
        Extension of video file.
    fps : int (optional, default 30)
        Framerate for output video.
    bins : int (optional, default 6)
        Number of distinct example to sample. For example if length_of_motif_video in config.yaml is set to 1000, and bins=6, a maximum ~166 (1000/6)
        frames of each sequence will be used in creating the motif video. Prevents motif videos from being one long behavioral sequence.
    cluster_method : str (optional, default 'kmeans')
        'kmeans' or 'GMM'

    Returns
    -------
    None. Saves motif videos.

    """

    import tqdm
    import cv2 as cv
    import os
    import glob
    import numpy as np
    import pandas as pd
    from typing import Optional


    labels: Optional[np.ndarray] = None
    output: Optional[str] = None

    cluster_method = cfg['parameterization']

    projectPath=cfg['project_path']
    print("Videos being created for "+file+" ...")
    if cluster_method == 'kmeans':
        labels = np.load(glob.glob(path_to_file+'/'+str(n_cluster)+'_km_label_'+'*.npy')[0])
    elif cluster_method == 'GMM':
        labels = np.load(glob.glob(path_to_file+'/'+str(n_cluster)+'_gmm_label_'+'*.npy')[0])
    elif cluster_method == 'hmm':
        labels = np.load(glob.glob(path_to_file+'/'+str(n_cluster)+'_km_label_'+'*.npy')[0])

    capture, width, height =  capture_video(cfg, file, videoType, symlinks)

    data = extract_pose_data(path_to_file, projectPath, file, extractData)

    if labels is not None:
        for cluster in range(n_cluster):
            print('Cluster: %d' %(cluster))
            vid_length = cfg['length_of_motif_video']
            cluster_lbl = np.where(labels == cluster)
            cluster_lbl = cluster_lbl[0]
            if cluster_lbl.shape[0] < vid_length:
                vid_length=cluster_lbl.shape[0]-1
            cons_lbl = consecutive(cluster_lbl, 1)
            cons_df = pd.DataFrame(cons_lbl)
            try:
                startFrames = cons_df[0].tolist()
            except KeyError:
                startFrames=0
            cons_df = cons_df.T
            cons_counts = pd.DataFrame(cons_df.count())
            cons_counts.columns=['length']
            cons_counts['startFrame']=startFrames
            cons_counts.sort_values('length', inplace=True, ascending=False)
            cons_counts.reset_index(inplace=True)
            frames = cons_counts['startFrame'].tolist()
            lengths = cons_counts['length'].tolist()

            used_seqs=[]
            while len(used_seqs)<vid_length:
                if lengths[0]==0:
                    break
                for i in frames:
                    if len(used_seqs)>=vid_length:
                        break
                    idx = frames.index(i)
                    length = lengths[idx]
                    endFrame = i+length
                    seq = list(range(i, endFrame))
                    if len(seq)<vid_length//bins:
                        used_seqs.extend(seq)
                    else:
                        seq = seq[:vid_length//bins]
                        used_seqs.extend(sorted(seq))
            if extractData and data is not None:
                clusterData = data.iloc[used_seqs,:]
                clusterData.to_csv(os.path.join(path_to_file,'dlcPoseData', file+'_DLC_Results_Cluster'+str(cluster)+'.csv'))

            if len(used_seqs) > vid_length:
                used_seqs = used_seqs[:vid_length]

            if flag == "motif":
                output = os.path.join(path_to_file,"cluster_videos",file+'-motif_%d_longestSequences_binned%d.avi' %(cluster,bins))
            elif flag == "community":
                output = os.path.join(path_to_file,"community_videos",file+'-community_%d_longestSequences_binned%d.avi' %(cluster,bins))

            if output is not None and not os.path.exists(output):
                video = cv.VideoWriter(output, cv.VideoWriter_fourcc('M','J','P','G'), fps, (int(width), int(height)))

                if len(used_seqs) < cfg['length_of_motif_video']:
                    vid_length = len(used_seqs)
                else:
                    vid_length = cfg['length_of_motif_video']

                for num in tqdm.tqdm(range(vid_length)):
                    idx = used_seqs[num]
                    capture.set(1,idx)
                    _, frame = capture.read()
                    video.write(frame)

                video.release()
            else:
                print("Video for cluster %d already found, skipping..." %cluster)
        capture.release()





def motif_videos(config, files = None, videoType='.mp4', fps=30, bins=6, cluster_method="kmeans", extractData=False, symlinks=False):
    """Create custom motif videos. This differs from the function in the main vame repository in that
    rather than the first frames 1000 frames (or whatever number assigned in config.yaml) of each cluster,
    motif videos will sample the longest sequential number of frames in the same behavioral cluster.
    The 'bins' parameter allows manual diversifying of the motif videos, as it will represent
    the minimum number of examples sampled from. For example if creating a 1000 frame motif video with
    6 bins, each bin will be ~167 frames long, to prevent motif videos from having only one long continuous
    behavior video. Milage may very, but this helps with my data, and feedback is welcome.

    Parameters
    ----------
    config : string
        Path to config.yaml file for project.
    model_name : string
        Name of model (subdirectory of 'results/')
    videoType : string, optional (default '.mp4')
        Video format analyzed.
    fps : int (optional, default 30)
        Frames per second for output result video.
    bins : int (optional, default 6)
        Minimum number of bins different behavioral epochs to include in motif videos.
    cluster_method : string (optional, default 'kmeans')
        Method used for clustering. Options are 'kmeans' or 'hmm'
    rename : tuple (optional, default None)
        In very early trial testing, but allows analysis of videos that have been renamed,
        or that you want to rename during analysis.

    Returns
    -------
    Creates files with examples of each segmented behavior,
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    symlinks = symlinks
    flag = 'motif'
    parameterization = cfg['parameterization']
    ic(parameterization)

    if files is None:
        files = AlHf.get_files(config)

    for file in files:
        if parameterization == 'hmm':
            path_to_file=os.path.join(cfg['project_path'], 'results',file,model_name,load_data,cluster_method+'-'+str(n_cluster)+'-'+str(cfg['hmm_iters']),'')
            ic(path_to_file)
        else:
            path_to_file=os.path.join(cfg['project_path'], 'results',file,model_name,load_data,cluster_method+'-'+str(n_cluster),'')

        if not os.path.exists(os.path.join(path_to_file, 'cluster_videos')):
            os.makedirs(os.path.join(path_to_file, 'cluster_videos'))

        get_cluster_vid(cfg, path_to_file, file, n_cluster, videoType, flag, fps=fps, bins=bins, cluster_method=cluster_method, extractData=extractData, symlinks=symlinks)

    print("All videos have been created!")



def find_longest_sequences(labels, n_cluster, vid_length, bins, limit_by_vid_length=True):
    """
    Finds the longest sequences for each cluster and returns a list of frame indices.

    Args:
        labels (np.array): Array of cluster labels for each frame.
        n_cluster (int): Number of clusters.
        vid_length (int): Desired length of the video.
        bins (int): Number of distinct examples to sample.

    Returns:
        dict: Dictionary with cluster numbers as keys and lists of frame indices as values.
    """
    # Check that the input labels are a NumPy array
    if not isinstance(labels, np.ndarray):
        raise ValueError("Input labels must be a NumPy array")

    longest_sequences = {cluster: [] for cluster in range(n_cluster)}  # Initialize with empty lists for each cluster

    for cluster in range(n_cluster):
        cluster_lbl = np.where(labels == cluster)[0]

        if cluster_lbl.size == 0:  # No frames for this cluster
            continue  # Skip to the next cluster

        # Create a new variable to store the modified vid_length
        cluster_vid_length = min(vid_length, cluster_lbl.shape[0] - 1)

        cons_lbl = consecutive(cluster_lbl, stepsize=1)
        cons_df = pd.DataFrame(cons_lbl)

        try:
            startFrames = cons_df[0].tolist()
        except KeyError:
            startFrames = []
        if startFrames:
            cons_df = cons_df.T
            cons_counts = pd.DataFrame(cons_df.count())
            cons_counts.columns = ['length']
            cons_counts['startFrame'] = startFrames
            cons_counts.sort_values('length', inplace=True, ascending=False)
            cons_counts.reset_index(inplace=True)

            used_seqs = []
            for _, row in cons_counts.iterrows():
                startFrame = row['startFrame']
                length = row['length']
                endFrame = startFrame + length
                seq = list(range(startFrame, endFrame))

                if limit_by_vid_length:
                    if len(seq) < cluster_vid_length // bins:
                        used_seqs.extend(seq)
                    else:
                        seq = seq[:cluster_vid_length // bins]
                        used_seqs.extend(sorted(seq))
                    if len(used_seqs) >= cluster_vid_length:
                        break
                else:
                    used_seqs.extend(seq)

            if limit_by_vid_length and len(used_seqs) > cluster_vid_length:
                used_seqs = used_seqs[:cluster_vid_length]

            longest_sequences[cluster] = used_seqs

    return longest_sequences

def find_sequences(group_df, min_length=30):
    """_summary_

    Identify and filter sequences of consecutive frames in a DataFrame.

    Parameters:
    group_df (DataFrame): The input DataFrame.
    min_length (int, optional): The minimum length of sequences to keep. Defaults to 30.

    Returns:
    DataFrame: The input DataFrame with sequences that meet the minimum length requirement.
    """
    # Check that the required columns are present
    required_columns = ['frame']
    for col in required_columns:
        if col not in group_df.columns:
            raise ValueError(f"Input DataFrame must contain '{col}' column")

    # Calculate the difference between consecutive frame numbers
    new_sequence = group_df['frame'].diff().fill_null(1) != 1
    group_df = group_df.with_columns([
        new_sequence.alias('new_sequence')
    ])
    # Cumulatively sum the new_sequence column to identify unique sequences
    sequence_id = group_df['new_sequence'].cumsum()
    group_df = group_df.with_columns([
        sequence_id.alias('sequence_id')
    ])
    # Count the length of each sequence
    sequence_lengths = group_df.groupby('sequence_id').agg([
        pl.count('frame').alias('sequence_length')
    ])
    # Join the sequence lengths back to the original group DataFrame
    group_df = group_df.join(sequence_lengths, on='sequence_id')
    # Filter sequences by the minimum length
    group_df = group_df.filter(pl.col('sequence_length') >= min_length)

    # Drop the columns used for intermediate calculations
    return group_df.drop(['new_sequence', 'sequence_id', 'sequence_length'])

def find_consecutive_sequences(df, min_length=30):
    """_summary_

    Apply find_sequences to each group in the DataFrame grouped by 'file_name' and 'motif'.

    Parameters:
    df (DataFrame): The input DataFrame.
    min_length (int, optional): The minimum length of sequences to keep. Defaults to 30.

    Returns:
    DataFrame: The input DataFrame with sequences that meet the minimum length requirement.
    """
    # Check that the required columns are present
    required_columns = ['file_name', 'motif']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Input DataFrame must contain '{col}' column")

    return df.groupby(['file_name', 'motif']).apply(find_sequences, min_length=min_length)



def find_longest_sequence(df):
    # Step 1: Sort the DataFrame by 'group', 'time_point', and 'frame' to ensure the frames are in order
    df = df.sort(['time_point', 'group', 'rat', 'frame'])

    # Step 2: Identify when a new sequence starts (when the difference between consecutive frames is not 1)
    df = df.with_columns(
        (pl.col('frame').diff().fill_null(1) != 1).alias('new_sequence')
    )

    # Step 3: Create a unique identifier for each sequence by cumulatively summing the 'new_sequence' column
    df = df.with_columns(
        pl.col('new_sequence').cumsum().alias('sequence_id')
    )

    # Step 4: Calculate the length of each sequence
    sequence_lengths = df.groupby('sequence_id').agg(
        pl.count('frame').alias('sequence_length')
    )

    # Step 5: Join the sequence lengths back to the original DataFrame
    df = df.join(sequence_lengths, on='sequence_id')

    # Step 6: Find the longest sequence for each combination of 'motif', 'group', and 'time_point'
    longest_sequences = df.groupby(['motif', 'group', 'time_point']).agg(
        pl.max('sequence_length').alias('max_sequence_length')
    ).join(df, on=['motif', 'group', 'time_point'])

    # Step 7: Identify the sequence_id(s) of the longest sequences
    longest_sequence_ids = longest_sequences.filter(
        pl.col('sequence_length') == pl.col('max_sequence_length')
    ).select('sequence_id').unique()

    # Step 8: Filter the original DataFrame to only include the rows that are part of the longest sequences
    df = df.join(longest_sequence_ids, on='sequence_id')

    return df




def capture_video(cfg, file, videoType, symlinks):
    """
    Summary:
        This function captures frames from a video file and returns the capture object along with the width and height of the frames.

    Args:
        cfg (dict): VAME config dictionary.
        file (str): Name of the video file to capture frames from.
        videoType (str): Extension of the video file.
        symlinks (bool): Flag indicating whether to use symlinks or not.

    Raises:
        Exception: If the video capture fails.

    Returns:
        tuple: A tuple containing the capture object, width, and height of the frames.
    """

    width = height = None

    if not symlinks:
        capture = cv.VideoCapture(os.path.join(cfg['project_path'], "videos", file + videoType))
    else:
        original_file_name = file + videoType
        trimmed_file_name = re.sub(r'_Rat\d+', '', original_file_name)
        trimmed_full_video_path = os.path.join(cfg['project_path'], "real_videos", trimmed_file_name)
        expanded_path = os.path.expandvars(trimmed_full_video_path)
        resolved_path = os.path.realpath(expanded_path)
        capture = cv.VideoCapture(resolved_path)

    if capture.isOpened():
        width  = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    else:
        raise Exception("Video capture failed")

    return capture, width, height


def crop_following_rat(capture: cv.VideoCapture, centroid_x: List[int], centroid_y: List[int], crop_size: int, num_frames_to_process: int, time_point: str, group: str, motif: str, speed: List[float]) -> List[cv.VideoCapture]:
    """
    Crop frames from a video capture around the given centroid coordinates, add text to the cropped frames, and return a list of the cropped frames.

    Args:
        capture (cv.VideoCapture): The video capture object.
        centroid_x (List[int]): List of x-coordinates of the centroids.
        centroid_y (List[int]): List of y-coordinates of the centroids.
        crop_size (int): The size of the crop window (e.g., 100x100 pixels).
        num_frames_to_process (int): The number of frames to process.
        time_point (str): The time point of the frames.
        group (str): The group of the frames.
        motif (str): The motif of the frames.
        speed (List[float]): List of speeds corresponding to each centroid.

    Returns:
        List[cv.VideoCapture]: List of cropped frames from the video.
    """
    # TODO: Crop video frames to the arena the animal is in. Cropping to centroid is not ideal.
    crop_around_rat = False
    # Define half the window size for cropping
    half_crop = crop_size // 2

    # List to hold the cropped frames
    cropped_frames = []

    # Text settings
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White color
    line_type = 2
    circle_color = (0, 0, 255)  # Red color for circle
    circle_radius = 2
    bottom_left_corner = (10, crop_size - 10)  # Adjust as needed
    #bottom_left_corner_speed = (10, crop_size - 25)  # Adjust as needed for speed text
    rat_boundaries = {
        "Rat1": {"x": (0, 332.5), "y": (0, 328.5)},
        "Rat2": {"x": (332.5, 680), "y": (0, 328.5)},
        "Rat3": {"x": (0, 332.5), "y": (328.5, 680)},
        "Rat4": {"x": (332.5, 680), "y": (328.5, 680)},
    }
    # Iterate over each frame and the corresponding centroid coordinates
    for x, y in zip(centroid_x, centroid_y):
        # Read the next frame from the video capture
        ret, frame = capture.read()
        if not ret:
            break  # Break if the video has ended or if there's an error
        
        if crop_around_rat:
            # Calculate the top-left corner of the crop
            start_x = int(max(0, x - half_crop))
            start_y = int(max(0, y - half_crop))

            crop = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]

            # Ensure the crop window doesn't go outside the frame
            end_x = int(min(frame.shape[1], x + half_crop))
            end_y = int(min(frame.shape[0], y + half_crop))

            # Calculate padding to ensure the centroid is in the center of the crop
            pad_x = max(0, half_crop - start_x, end_x - frame.shape[1] + half_crop)
            pad_y = max(0, half_crop - start_y, end_y - frame.shape[0] + half_crop)

            crop_padded = cv.copyMakeBorder(crop, pad_y, pad_y, pad_x, pad_x, cv.BORDER_CONSTANT, value=[0, 0, 0])

            # Calculate the centroid position relative to the crop
            centroid_x_relative = x - start_x
            centroid_y_relative = y - start_y

            # Calculate the centroid position relative to the padded crop
            centroid_x_padded = int(centroid_x_relative + pad_x)
            centroid_y_padded = int(centroid_y_relative + pad_y)

            cv.circle(crop_padded, (centroid_x_padded, centroid_y_padded), circle_radius, circle_color, line_type)

            # Write the text (time_point, group, motif) on the cropped frame
            cv.putText(crop_padded, f"{time_point} {group} {motif}", bottom_left_corner, font, font_scale, font_color, line_type)

            # Add the crop to the list of cropped frames
            cropped_frames.append(crop_padded)
        
        else:
            for rat, boundary in rat_boundaries.items():
                if boundary["x"][0] <= x < boundary["x"][1] and boundary["y"][0] <= y < boundary["y"][1]:
                    # Calculate the top-left corner of the crop based on the rat boundary
                    start_x = int(boundary["x"][0])
                    start_y = int(boundary["y"][0])

                    # Calculate the bottom-right corner of the crop based on the rat boundary
                    end_x = int(boundary["x"][1])
                    end_y = int(boundary["y"][1])

                    # Crop the frame
                    crop = frame[start_y:end_y, start_x:end_x]

                    # Write the text (time_point, group, motif) on the cropped frame
                    cv.putText(crop, f"{time_point} {group} {motif}", bottom_left_corner, font, font_scale, font_color, line_type)

                    # Add the crop to the list of cropped frames
                    cropped_frames.append(crop)

                    # Break the loop once the correct rat boundary is found
                    break

    # Release the video capture
    capture.release()

    # Return the list of cropped frames
    return cropped_frames


def process_video_group(cfg: dict, group_df: pd.DataFrame, videoType: str, symlinks: bool, start_frame: int, end_frame: int) -> List[cv.VideoCapture]:
    """
    This function processes a group of frames from a video by cropping them around given centroid coordinates, adding text to the cropped frames, and returning a list of the cropped frames.

    Args:
        cfg (dict): VAME config dictionary.
        group_df (pd.DataFrame): DataFrame containing the group of frames to process.
        videoType (str): Extension of the video file.
        symlinks (bool): Flag indicating whether to use symlinks or not.
        start_frame (int): Start frame index.
        end_frame (int): End frame index.

    Returns:
        List[cv.VideoCapture]: List of cropped frames from the video.
    """

    # Extract necessary information from group_df
    file_name = group_df['file_name'].iloc[0]
    time_point = group_df['time_point'].iloc[0]
    group = group_df['group'].iloc[0]
    motif = group_df['motif'].iloc[0]
    centroid_x = group_df['centroid_x'].tolist()
    centroid_y = group_df['centroid_y'].tolist()
    speed = group_df['speed'].tolist()

    # Open the video file
    capture, _, _ = capture_video(cfg, file_name, videoType, symlinks)
    capture.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    num_frames_to_process = end_frame - start_frame + 1

    # Define the crop size
    crop_size = 200
    new_cropped_frames = crop_following_rat(capture, centroid_x, centroid_y, crop_size, num_frames_to_process, time_point, group, motif, speed)

    capture.release()

    return new_cropped_frames


def write_frames_to_video(frames_list: List, output_path: str, fps: int = 30):
    """
    Writes a list of frames to a video file.

    Args:
        frames_list (list): A list of frames to be written to the video file.
        output_path (str): The path where the video file will be saved.
        fps (int): The frames per second for the video. Default is 30.

    Returns:
        None
    """
    if not frames_list:
        print("No frames to write to video.")
        return

    height, width, layers = frames_list[0].shape
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    # Check if the output file already exists
    if os.path.exists(output_path):
        os.remove(output_path)  # Remove the existing file

    out = cv.VideoWriter(output_path, fourcc, float(fps), (width, height))

    for frame in frames_list:
        out.write(frame)
        
    out.release()





def create_videos_for_motifs(video_df: pl.DataFrame | pd.DataFrame, cfg: dict, videoType: str, symlinks: bool, fps: int, n_cluster: int, vid_length: int, bins: int, min_consecutive_frames: int, videos_path: str) -> None:
    """
    Create videos for each motif in the DataFrame.

    Parameters:
    video_df (DataFrame): The input DataFrame.
    cfg (dict): Configuration settings.
    videoType (str): The type of video to create.
    symlinks (bool): Whether to create symbolic links.
    fps (int): The frames per second for the video.
    n_cluster (int): The number of clusters.
    vid_length (int): The length of the video.
    bins (int): The number of bins.
    min_consecutive_frames (int): The minimum number of consecutive frames.
    videos_path (str): The path to save the videos.

    Returns:
    None
    """
    # Check that the required columns are present
    required_columns = ['rat_id', 'motif', 'frame']
    for col in required_columns:
        if col not in video_df.columns:
            raise ValueError(f"Input DataFrame must contain '{col}' column")

    motif_videos: Dict[int, List[str]] = {}  # Dictionary to store the frames for each motif

    # Reset the index of video_df
    video_df = video_df.reset_index()

    for motif in range(n_cluster):
        # Filter the motif in video_df by the cluster
        motif_df = video_df[video_df['motif'] == motif].sort_values(['file_name', 'rat_id', 'frame'])

        if motif_df.empty:
            print(f"No frames found for motif {motif} in video_df")
            continue
        
        cropped_frames_list: list[VideoCapture] = []

        for sequence_id in motif_df['sequence_id'].unique():
            new_cropped_frames: list[VideoCapture] = []
            sequence_df = motif_df[motif_df['sequence_id'] == sequence_id].sort_values('frame')

            start_frame = min(sequence_df['frame'])
            end_frame = max(sequence_df['frame'])

            new_cropped_frames: list[VideoCapture] = process_video_group(cfg, sequence_df, videoType, symlinks, start_frame, end_frame)

            for frame in new_cropped_frames:
                if frame is not None and np.any(frame):
                    cropped_frames_list.append(frame)
                else: 
                    print("Frame is None or empty")
                    
            # Store the frames in the dictionary
        #cropped_frames_list.extend(new_cropped_frames)
        
        max_frame_height = 0
        max_frame_width = 0
        
        max_frame_width = max([frame.shape[1] for frame in cropped_frames_list])
        max_frame_height = max([frame.shape[0] for frame in cropped_frames_list])
        
        
        # Resize frames
        for i, frame in enumerate(cropped_frames_list):
            current_frame_width = frame.shape[1]
            current_frame_height = frame.shape[0]
            
            left_frame_side = right_frame_side = top_frame_side = bottom_frame_side = 0
                
            if current_frame_width < max_frame_width:
                left_frame_side = (max_frame_width - current_frame_width) // 2
                right_frame_side = max_frame_width - current_frame_width - left_frame_side

            if current_frame_height < max_frame_height:
                top_frame_side = (max_frame_height - current_frame_height) // 2
                bottom_frame_side = max_frame_height - current_frame_height - top_frame_side

            cropped_frames_list[i] = cv.copyMakeBorder(frame, top=top_frame_side, bottom=bottom_frame_side, left=left_frame_side, right=right_frame_side, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                
                #cropped_frames_list[i] = cv.resize(frame, mean_shape)
            
    
        # Check to make sure all frames are the same shape:
        for frame in cropped_frames_list:
            if frame.shape != cropped_frames_list[0].shape:
                print("Frame shape is not equal to the first frame shape")
                print(frame.shape)
                print(cropped_frames_list[0].shape)
                raise ValueError("Frame shape is not equal to the first frame shape")
        
        if cropped_frames_list:
            motif_videos[motif] = cropped_frames_list # ? Haven't found a use (need?) for this yet
            output_video_path = os.path.join(videos_path, f"motif_{motif}_clips.mp4")  # Change the file name as needed
            write_frames_to_video(cropped_frames_list, output_video_path, fps)
            print(f"Video for cluster {motif} created at {output_video_path}")


def find_conserved_motif_sequences(df_pandas: pd.DataFrame, sequence_length: int = 30) -> pd.DataFrame:
    """
    Adds several columns to the input DataFrame to calculate sequence conservation and length.

    Args:
        df_pandas (pd.DataFrame): Input DataFrame with columns 'file_name', 'frame', 'rat_id', 'motif', and 'time_point'.
        sequence_length (int, optional): Minimum length of a sequence. Defaults to 30.

    Returns:
        pd.DataFrame: Modified DataFrame with additional columns 'is_sequence_conserved', 'sequence_id', 'sequence_length', and 'is_long_sequence'.
    """
    # Calculate whether a sequence is conserved
    df_pandas['is_sequence_conserved'] = df_pandas.sort_values('frame').groupby(['file_name', 'rat_id','motif', 'time_point'])['frame'].transform(lambda x: x.diff().eq(1))

    df_pandas.sort_values(['file_name', 'frame', 'rat_id'], inplace=True)

    # Every time is_sequence_conserved changes from True to False, generate a new sequence ID
    df_pandas['sequence_id'] = (df_pandas['is_sequence_conserved'] != df_pandas['is_sequence_conserved'].shift()).cumsum()

    # Calculate the length of each sequence
    df_pandas['sequence_length'] = df_pandas.groupby('sequence_id')['frame'].transform('count')

    # Determine if a sequence is longer than the specified length
    df_pandas['is_long_sequence'] = df_pandas['sequence_length'] > sequence_length

    return df_pandas


def select_long_sequences(df_pandas: pd.DataFrame, fps: int = 30, maximum_video_length: int = 5) -> tuple:
    """
    Selects long sequences from a pandas DataFrame based on certain criteria and calculates the video length for each motif.
    It also creates bins for sequence lengths and counts the number of sequences in each bin for each motif.

    Args:
    - df_pandas: A pandas DataFrame containing information about sequences, including columns 'sequence_id', 'is_long_sequence', 'motif', 'frame', and 'sequence_length'.
    - fps: An integer representing the frames per second of the video (default is 30).
    - maximum_video_length: An integer representing the maximum video length in minutes (default is 5).

    Returns:
    - long_sequences: A pandas DataFrame containing the selected long sequences, sorted by 'sequence_length' in descending order.
    - max_video_length_frames: An integer representing the maximum video length in frames.
    """

    # Select the rows from the DataFrame where 'is_long_sequence' is True
    long_sequences = df_pandas[df_pandas['is_long_sequence']]

    # Calculate the total number of unique long sequences
    total_long_sequences = long_sequences['sequence_id'].nunique()

    # Calculate the total number of long sequences per motif
    total_long_sequences_per_motif = long_sequences.groupby('motif')['sequence_id'].nunique()

    # Calculate the total number of frames per motif for long sequences
    total_long_sequence_frames_per_motif = long_sequences.groupby('motif')['frame'].count()

    # Calculate the video length per motif in seconds
    video_length_per_motif = total_long_sequence_frames_per_motif / fps

    # Create a DataFrame for video length per motif
    video_length_per_motif = video_length_per_motif.to_frame('seconds')

    # Add a 'minutes' column to the video length per motif DataFrame
    video_length_per_motif['minutes'] = video_length_per_motif['seconds'] / 60

    # Determine the longest sequence length in the DataFrame
    longest_sequence = df_pandas['sequence_length'].max()

    # Define the bin edges for sequence length bins
    bin_edges = list(range(30, longest_sequence + 10, 10))

    # Create the sequence length bins using the 'pd.cut' function
    df_pandas['sequence_length_bin'] = pd.cut(df_pandas['sequence_length'], bins=bin_edges)

    # Calculate the number of sequences per bin per motif
    sequences_per_bin_per_motif = df_pandas.groupby(['motif', 'sequence_length_bin'], observed=False)['sequence_id'].nunique()

    # Convert the resulting series to a DataFrame and reset the index
    sequences_per_bin_per_motif_df = sequences_per_bin_per_motif.reset_index(name='count')

    # Calculate the maximum video length in frames
    max_video_length_frames = maximum_video_length * 60 * fps

    # Sort the long sequences DataFrame by 'sequence_length' in descending order
    long_sequences = long_sequences.sort_values('sequence_length', ascending=False)

    num_false = long_sequences[long_sequences['is_long_sequence'] == False].shape[0]

    if num_false == 0:
        return long_sequences, max_video_length_frames
    else:
        print(f"There are {num_false} sequences that are not long enough to be included in the video in 'long_sequences.")


def select_sequences(group: pd.DataFrame, max_length: int, extra_rows: int, random_sample: bool = False) -> pd.Series:
    """
    Select sequences from a group based on their length, up to a maximum length.
    It also allows for selecting additional sequences beyond the maximum length if specified.
    If random_sample is True, sequences are randomly selected from each bin.

    Args:
        group (pd.DataFrame): A pandas DataFrame containing the sequences to select from for a given motif.
                              It should have columns 'sequence_id' and 'sequence_length'.
        max_length (int): The maximum total length of the selected sequences.
        extra_rows (int): The number of additional sequences to select beyond the maximum length.
        random_sample (bool): If True, sequences are randomly selected from each bin.

    Returns:
        pd.Series: A pandas Series containing the IDs of the selected sequences.
    """
    if random_sample:
        group = group.sample(frac=1)
    else:
        group = group.sort_values('sequence_length', ascending=False)
    
    group.reset_index(drop=True, inplace=True)

    selected_sequences = []
    total_length = 0
    over_max = False

    for _, row in group.iterrows():
        if total_length + row['sequence_length'] > max_length:
            if over_max:
                break
            over_max = True
            continue

        selected_sequences.append(row['sequence_id'])
        total_length += row['sequence_length']

        if over_max:
            extra_rows -= 1
            if extra_rows == 0:
                break

    return pd.Series(selected_sequences)


def motif_videos_conserved(config, symlinks = False, videoType = '.mp4', fps = 30, bins = 6, maximum_video_length=5, min_consecutive_frames = 30):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    path_to_file = cfg['project_path']
    vid_length = maximum_video_length

    symlinks = symlinks

    rat_boundaries = {
        "Rat1": {"x": (0, 332.5), "y": (0, 328.5)},
        "Rat2": {"x": (332.5, 680), "y": (0, 328.5)},
        "Rat3": {"x": (0, 332.5), "y": (328.5, 680)},
        "Rat4": {"x": (332.5, 680), "y": (328.5, 680)},
    }

    df: Union[pl.DataFrame, pd.DataFrame] = AlHf.create_andOR_get_master_df(config, fps=fps, create_new_df=False, df_kind = 'pandas')

    df = find_conserved_motif_sequences(df_pandas = df, sequence_length = min_consecutive_frames)

    long_sequences, max_video_length_frames = select_long_sequences(df, maximum_video_length)

    selected_sequences_per_motif = long_sequences.groupby('motif').apply(select_sequences, max_length=max_video_length_frames, extra_rows = 10, random_sample = True)

    # Convert to DataFrame and reset index
    selected_sequences_df = selected_sequences_per_motif.reset_index()

    # Rename the columns for clarity
    selected_sequences_df.columns = ['motif', 'sequence_rank', 'sequence_id']

    if parameterization == 'hmm':
        long_sequences.to_csv(os.path.join(path_to_file, 'results', f"all_consecutive_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
        selected_sequences_df.to_csv(os.path.join(path_to_file, 'results', f"selected_consecutive_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
    else:
        long_sequences.to_csv(os.path.join(path_to_file, 'results', f"all_consecutive_sequences_{parameterization}-{n_cluster}.csv"))
        selected_sequences_df.to_csv(os.path.join(path_to_file, 'results', f"selected_consecutive_sequences_{parameterization}-{n_cluster}.csv"))

    #video_df = longest_sequences_df

    # Filter the dataframe by the selected sequences, ensure is_sequence_conserved is True
    video_df_1 = df[df['sequence_id'].isin(selected_sequences_df['sequence_id'])] # * Temporary change to see how it affects the videos
    #video_df_1 = df[df['sequence_id'].isin(long_sequences['sequence_id'])] !=# ! Using this line instead of the one above will potentially create much longer videos
    video_df_1 = video_df_1[video_df_1['is_sequence_conserved'] == True]

    video_df_2 = long_sequences[long_sequences['sequence_id'].isin(selected_sequences_df['sequence_id'])]
    video_df_2 = video_df_2[video_df_2['is_sequence_conserved'] == True]

    # Sort both data frames identically
    video_df_1 = video_df_1.sort_values(['file_name', 'rat_id', 'frame'])
    video_df_2 = video_df_2.sort_values(['file_name', 'rat_id', 'frame'])

    """
    # Check if video_df_1 and video_df_2 are identical before moving on
    if video_df_1.equals(video_df_2):
        video_df = video_df_1
    else:
        import sys
        print('video_df_1 and video_df_2 are not identical.')
        print('video_df_1:')
        print(video_df_1)
        print('video_df_2:')
        print(video_df_2)
        print('Exiting...')
        sys.exit()
        
    video_df_1 and video_df_2 are difference in the presence of one column and they are otherwise identical. 
    Feel free to check if needed.
    If they are not identical for you please submit an issue on GitHub.
    """

    video_df = video_df_1

    # sort by file_name, rat_id and frame
    video_df = video_df.sort_values(['file_name', 'rat_id', 'frame'])

    if parameterization == 'hmm':
        videos_path = os.path.join(path_to_file, 'results', 'videos', parameterization+'-'+str(n_cluster)+'-'+str(cfg['hmm_iters']))
    else:
        videos_path = os.path.join(path_to_file, 'results', 'videos', parameterization+'-'+str(n_cluster))

    os.makedirs(videos_path, exist_ok=True)

    create_videos_for_motifs(video_df, cfg, videoType, symlinks, fps, n_cluster, vid_length, bins, min_consecutive_frames, videos_path)
    
"""
if __name__ == "__main__":
    config = 'D:\\Users\\tywin\\VAME\\config.yaml'
    motif_videos_conserved(config, symlinks=True, videoType='.mp4', fps=30, bins=6, maximum_video_length=10, min_consecutive_frames=30)

"""
