from typing import List, Tuple
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
    
    # Define half the window size for cropping
    half_crop = crop_size // 2
    
    # List to hold the cropped frames
    cropped_frames = []
    
    # Text settings
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White color
    line_type = 2
    #circle_color = (0, 0, 255)  # Red color for circle
    #circle_radius = 2
    bottom_left_corner = (10, crop_size - 10)  # Adjust as needed
    #bottom_left_corner_speed = (10, crop_size - 25)  # Adjust as needed for speed text
    
    # Iterate over each frame and the corresponding centroid coordinates
    for x, y, spd in zip(centroid_x, centroid_y, speed):
        # Read the next frame from the video capture
        ret, frame = capture.read()
        if not ret:
            break  # Break if the video has ended or if there's an error
        
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
        #centroid_x_relative = x - start_x
        #centroid_y_relative = y - start_y

        # Calculate the centroid position relative to the padded crop
        #centroid_x_padded = int(centroid_x_relative + pad_x)
        #centroid_y_padded = int(centroid_y_relative + pad_y)

        #cv.circle(crop_padded, (centroid_x_padded, centroid_y_padded), circle_radius, circle_color, line_type)

        # Write the text (time_point, group, motif) on the cropped frame
        cv.putText(crop_padded, f"{time_point} {group} {motif}", bottom_left_corner, font, font_scale, font_color, line_type)
        
        # Write the speed on the cropped frame
        #cv.putText(crop_padded, f"Speed: {spd:.2f}", bottom_left_corner_speed, font, font_scale, font_color, line_type)
        
        # Add the crop to the list of cropped frames
        cropped_frames.append(crop_padded)
    
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
    file_name = group_df.loc[0, 'file_name']
    time_point = group_df.loc[0, 'time_point']
    group = group_df.loc[0, 'group']
    motif = group_df.loc[0, 'motif']
    centroid_x = group_df['centroid_x'].tolist()
    centroid_y = group_df['centroid_y'].tolist()
    speed = group_df['speed'].tolist()

    # Open the video file
    capture, _, _ = capture_video(cfg, file_name, videoType, symlinks)
    capture.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    num_frames_to_process = end_frame - start_frame + 1
    
    # Define the crop size
    crop_size = 200
    cropped_frames_list = crop_following_rat(capture, centroid_x, centroid_y, crop_size, num_frames_to_process, time_point, group, motif, speed)

    capture.release()
        
    return cropped_frames_list

def write_frames_to_video(frames_list, output_path, fps=30):
    # Check if there are any frames to write
    if not frames_list:
        print("No frames to write to video.")
        return
    
    # Get the height and width of the first frame
    height, width, layers = frames_list[0].shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # You can change 'mp4v' to 'XVID' if you prefer
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write each frame to the video
    for frame in frames_list:
        out.write(frame)
    
    # Release the VideoWriter
    out.release()
    print(f"Video written to {output_path}")


def process_motif(cfg, group_df, videoType, symlinks, motif):
    try:
        motif_df = group_df.filter(pl.col('motif') == motif)
        if motif_df.height == 0:
            ic(f"No data found for motif {motif}")
            return []
        start_frame = motif_df['frame'].min()
        end_frame = motif_df['frame'].max()
        ic("Start frame:", start_frame, "End frame:", end_frame)
    
        # Process the video for the given motif
        cropped_frames_list = process_video_group(cfg, motif_df, videoType, symlinks, start_frame, end_frame)
        ic("Processed video group, number of frames:", len(cropped_frames_list))
    
        return cropped_frames_list
    
    except Exception as e:
        ic(f"An error occurred while processing motif {motif}: {e}")
        return []


def motif_videos_conserved_OG(config, symlinks = False, videoType = '.mp4', fps = 30, bins = 6, maximum_video_length=1000, min_consecutive_frames = 60):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    vid_length = cfg['length_of_motif_video']
    
    symlinks = symlinks

    rat_boundaries = {
        "Rat1": {"x": (0, 332.5), "y": (0, 328.5)},
        "Rat2": {"x": (332.5, 680), "y": (0, 328.5)},
        "Rat3": {"x": (0, 332.5), "y": (328.5, 680)},
        "Rat4": {"x": (332.5, 680), "y": (328.5, 680)},
    }
    
    #df = pl.DataFrame({"file_name": [], "frame": [], "motif": [], "centroid_x": [], "centroid_y": [], "speed": [], "rat": [], "group": [], "time_point": []})

    # Get all the files
    files = AlHf.get_files(config)
    path_to_file = cfg['project_path']
    dlc_data_type = input("Were your DLC .csv files originally multi-animal? (y/n): ")
    labels_list = ana.get_labels(cfg, files, model_name, n_cluster)
    
    df = pl.DataFrame({
        "file_name": pl.Series([], dtype=pl.Utf8),
        "frame": pl.Series([], dtype=pl.Int64),
        "motif": pl.Series([], dtype=pl.Int64),
        "centroid_x": pl.Series([], dtype=pl.Float32),
        "centroid_y": pl.Series([], dtype=pl.Float32),
        "speed": pl.Series([], dtype=pl.Float32),
        "rat": pl.Series([], dtype=pl.Utf8),
        "group": pl.Series([], dtype=pl.Utf8),
        "time_point": pl.Series([], dtype=pl.Utf8)
    })

    if parameterization == 'hmm':
        df_path = (os.path.join(path_to_file, 'results', f"all_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
    else:
        df_path = (os.path.join(path_to_file, 'results', f"all_sequences_{parameterization}-{n_cluster}.csv"))
    
    if os.path.exists(df_path):
        df = pl.read_csv(df_path)
    else:
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
                filename, time_point, group, full_group_notation = AlHf.parse_filename(file)
                labels = labels_list[i]
                data = kin.get_dlc_file(path_to_file, dlc_data_type, file)
                rat = kin.get_dat_rat(data)    
                centroid_x, centroid_y = kin.calculate_centroid(data)
                rat_speed = kin.calculate_speed_with_spline(data, fps, window_size=5, pixel_to_cm=0.215)
                # Calculate the number of elements to trim
                trim_length = len(centroid_x) - len(labels)

                # Trim the centroid lists
                centroid_x_trimmed = centroid_x[trim_length:]
                centroid_y_trimmed = centroid_y[trim_length:]
                rat_speed_trimmed = rat_speed[trim_length:]

                motif_series = pl.Series(labels).cast(pl.Int64)
                centroid_x_series = pl.Series(centroid_x_trimmed).cast(pl.Float32)
                centroid_y_series = pl.Series(centroid_y_trimmed).cast(pl.Float32)
                speed_series = pl.Series(rat_speed_trimmed).cast(pl.Float32)

                # Create a new DataFrame with the labels and the rat value for this file
                temp_df = pl.DataFrame({
                    "file_name": [file] * len(labels),
                    "frame": list(range(len(labels))),
                    "motif": motif_series,
                    "centroid_x": centroid_x_series,
                    "centroid_y": centroid_y_series,
                    "speed": speed_series,
                    "rat": [rat] * len(labels),
                    "group": [group] * len(labels),
                    "time_point": [time_point] * len(labels)   
                })
                
                temp_df = temp_df.with_columns([
                    pl.col("frame").cast(pl.Int64),
                    #pl.col("motif").cast(pl.Int64),
                ])

                # Concatenate the new DataFrame with the existing one
                df = pl.concat([df, temp_df])
   

        if parameterization == 'hmm':
            df.write_csv(os.path.join(path_to_file, 'results', f"all_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
        else:
            df.write_csv(os.path.join(path_to_file, 'results', f"all_sequences_{parameterization}-{n_cluster}.csv"))
    

    
    """
    df = (df.lazy()
      .groupby(['motif', 'group', 'time_point'])
      .agg(pl.col('*').apply(find_consecutive_sequences, min_length=min_consecutive_frames))
      .explode(['file_name', 'frame', 'centroid_x', 'centroid_y', 'speed', 'rat', 'group', 'time_point'])
      .collect())
    """
        
    df_min_consecutive = find_consecutive_sequences(df, min_length=min_consecutive_frames)
    
    longest_sequences_df = find_longest_sequence(df_min_consecutive)
    
    if parameterization == 'hmm':
        df_min_consecutive.write_csv(os.path.join(path_to_file, 'results', f"all_minimum_consecutive_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
        longest_sequences_df.write_csv(os.path.join(path_to_file, 'results', f"all_longest_consecutive_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
    else:
        df_min_consecutive.write_csv(os.path.join(path_to_file, 'results', f"all_minimum_consecutive_sequences_{parameterization}-{n_cluster}.csv"))
        longest_sequences_df.write_csv(os.path.join(path_to_file, 'results', f"all_longest_consecutive_sequences_{parameterization}-{n_cluster}.csv"))
    

    video_df = longest_sequences_df

    if parameterization == 'hmm':
        videos_path = os.path.join(path_to_file, 'results', 'videos', parameterization+'-'+str(n_cluster)+'-'+str(cfg['hmm_iters']))
    else:
        videos_path = os.path.join(path_to_file, 'results', 'videos', parameterization+'-'+str(n_cluster))
    os.makedirs(videos_path, exist_ok=True)
    
    # Iterate over each group in the DataFrame
    for time_point in video_df['time_point'].unique():
        
        week_path =  os.path.join(videos_path, time_point)
        os.makedirs(week_path, exist_ok=True)
        
        for group in video_df.filter(pl.col('time_point') == time_point)['group'].unique():
            group_week_path = os.path.join(week_path, group)
            os.makedirs(group_week_path, exist_ok=True)
        
            group_motif_videos = []

            for motif in video_df.filter((pl.col('time_point') == time_point) & (pl.col('group') == group))['motif'].unique():
                for file_name in video_df.filter((pl.col('time_point') == time_point) & (pl.col('group') == group) & (pl.col('motif') == motif))['file_name'].unique():
                    
                    # Filter the DataFrame for the specific time_point, group, motif, and file_name, then sort by frame
                    group_df = video_df.filter((pl.col('time_point') == time_point) & 
                                               (pl.col('group') == group) & 
                                               (pl.col('motif') == motif) & 
                                               (pl.col('file_name') == file_name)).sort(['frame'])
                    
                    # Find all consecutive frame sequences
                    consecutive_sequences = group_df.filter(pl.col('frame').diff().fill_null(1) == 1)

                    # If there are no consecutive sequences, skip to the next file_name
                    if consecutive_sequences.height == 0:
                        continue
                    
                    # Create a 'group_id' column that identifies each consecutive sequence
                    # This should be done in the original group_df to ensure the column exists for the join operation
                    group_df = group_df.with_columns(
                        (pl.col('frame').diff().fill_null(1) != 1).alias('new_sequence')
                    )

                    group_df = group_df.with_columns(
                        pl.col('new_sequence').cumsum().alias('group_id')
                    )
                    
                    # Now use the group_df with the 'group_id' column to find the sequence lengths
                    sequence_lengths = group_df.groupby('group_id').agg(
                        pl.count().alias('length')
                    )

                    # Get the top three longest sequences
                    top_sequences = sequence_lengths.sort(by='length', descending=True).head(3)

                    # If there are no top sequences, skip to the next file_name
                    if top_sequences.height == 0:
                        continue
                    
                    ic(top_sequences)
                    
                    # Filter the original DataFrame to only include the top three longest consecutive frame sequences
                    group_df = group_df.join(top_sequences, on='group_id', how='semi')
                
                    for group_id in top_sequences['group_id']:
                        # Filter the group_df for the current group_id
                        sequence_df = group_df.filter(pl.col('group_id') == group_id)
                        
                        start_frame = sequence_df['frame'].min()
                        end_frame = sequence_df['frame'].max()
                        
                        if (end_frame-start_frame < 60):
                            continue
                        
                        ic(file_name,group, group_id, motif, start_frame, end_frame)

                        cropped_frames_list = process_video_group(cfg, sequence_df, videoType, symlinks, start_frame, end_frame)
                        group_motif_videos.append(cropped_frames_list)

                    
            # Concatenate all frame lists for the group into a single list
            all_frames_for_group = [frame for motif_frames in group_motif_videos for frame in motif_frames]

            # Write the concatenated list of frames to a video file
            output_video_path = os.path.join(group_week_path, f"all_motif_usage_{time_point}_{group}_{motif}.mp4")  # Change the file name as needed
            write_frames_to_video(all_frames_for_group, output_video_path)


def motif_videos_conserved(config, symlinks = False, videoType = '.mp4', fps = 30, bins = 6, maximum_video_length=1000, min_consecutive_frames = 60):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    vid_length = cfg['length_of_motif_video']
    
    symlinks = symlinks

    rat_boundaries = {
        "Rat1": {"x": (0, 332.5), "y": (0, 328.5)},
        "Rat2": {"x": (332.5, 680), "y": (0, 328.5)},
        "Rat3": {"x": (0, 332.5), "y": (328.5, 680)},
        "Rat4": {"x": (332.5, 680), "y": (328.5, 680)},
    }
    
    #df = pl.DataFrame({"file_name": [], "frame": [], "motif": [], "centroid_x": [], "centroid_y": [], "speed": [], "rat": [], "group": [], "time_point": []})

    path_to_file = cfg['project_path']
    
    df = AlHf.create_andOR_get_master_df(config, fps, create_new_df=False)
        
    df_min_consecutive = find_consecutive_sequences(df, min_length=min_consecutive_frames)
    
    longest_sequences_df = find_longest_sequence(df_min_consecutive)
    
    if parameterization == 'hmm':
        df_min_consecutive.write_csv(os.path.join(path_to_file, 'results', f"all_minimum_consecutive_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
        longest_sequences_df.write_csv(os.path.join(path_to_file, 'results', f"all_longest_consecutive_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
    else:
        df_min_consecutive.write_csv(os.path.join(path_to_file, 'results', f"all_minimum_consecutive_sequences_{parameterization}-{n_cluster}.csv"))
        longest_sequences_df.write_csv(os.path.join(path_to_file, 'results', f"all_longest_consecutive_sequences_{parameterization}-{n_cluster}.csv"))
    
    video_df = df

    if parameterization == 'hmm':
        videos_path = os.path.join(path_to_file, 'results', 'videos', parameterization+'-'+str(n_cluster)+'-'+str(cfg['hmm_iters']))
    else:
        videos_path = os.path.join(path_to_file, 'results', 'videos', parameterization+'-'+str(n_cluster))
    os.makedirs(videos_path, exist_ok=True)
    
    unique_motifs = video_df['motif'].unique()
    pbar = tqdm(total=len(unique_motifs), desc="Processing Motifs")
        
    for motif in video_df['motif'].unique():
        motif_videos = []      
        for rat_id in video_df.filter((pl.col('motif') == motif))['rat_id'].unique():
            
            # Filter the DataFrame for the specific motif and rat_id, then sort by frame
            group_df = video_df.filter((pl.col('motif') == motif) & 
                                       (pl.col('rat_id') == rat_id)).sort(['frame'])
            
            # Check if the motif is still in the dataframe after filtering
            if motif not in group_df['motif'].unique():
                print(f"Motif {motif} not found in the dataframe after filtering for rat_id {rat_id}.")
                continue
            
            # Find all consecutive frame sequences
            consecutive_sequences = group_df.filter(pl.col('frame').diff().fill_null(1) == 1)

            # If there are no consecutive sequences, skip to the next rat_id
            if consecutive_sequences.height == 0:
                continue
            
            # Create a 'group_id' column that identifies each consecutive sequence
            # This should be done in the original group_df to ensure the column exists for the join operation
            group_df = group_df.with_columns(
                (pl.col('frame').diff().fill_null(1) != 1).alias('new_sequence')
            )


            group_df = group_df.with_columns(
                pl.col('new_sequence').cumsum().alias('group_id')
            )
            
            # Now use the group_df with the 'group_id' column to find the sequence lengths
            sequence_lengths = group_df.groupby('group_id').agg(
                pl.count().alias('length')
            )

            # Get the top three longest sequences
            top_sequences = sequence_lengths.sort(by='length', descending=True).head(5)

            # If there are no top sequences, skip to the next file_name
            if top_sequences.height == 0:
                continue
            
            ic(top_sequences)
            
            # Filter the original DataFrame to only include the top longest consecutive frame sequences
            group_df = group_df.join(top_sequences, on='group_id', how='semi')
        

            for group_id in top_sequences['group_id']:
                # Filter the group_df for the current group_id
                sequence_df = group_df.filter(pl.col('group_id') == group_id)
                
                start_frame = sequence_df['frame'].min()
                end_frame = sequence_df['frame'].max()
                
                if (end_frame-start_frame < 5):
                    continue
                
                ic(group_id, motif, start_frame, end_frame)

                cropped_frames_list = process_video_group(cfg, sequence_df, videoType, symlinks, start_frame, end_frame)
                
                motif_videos.append(cropped_frames_list)
            

        # Concatenate all frame lists for the group into a single list
        all_frames_for_group = [frame for motif_frames in motif_videos for frame in motif_frames]

        # Write the concatenated list of frames to a video file
        output_video_path = os.path.join(videos_path, f"motif_{motif}_clips.mp4")  # Change the file name as needed
        write_frames_to_video(all_frames_for_group, output_video_path)
        pbar.update(1)


def create_videos_for_motifs(video_df: pl.DataFrame, cfg: dict, videoType: str, symlinks: bool, fps: int, n_cluster: int, vid_length: int, bins: int, min_consecutive_frames: int, videos_path: str) -> None:
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

    motif_videos = {}  # Dictionary to store the frames for each motif

    for cluster in range(n_cluster):
        cropped_frames_list = []
        for rat_id in video_df['rat_id'].unique():
            # Filter the DataFrame for the specific cluster and rat_id
            cluster_df = video_df.filter((pl.col('motif') == cluster) & (pl.col('rat_id') == rat_id)).sort('frame')

            # Convert the 'frame' column to a NumPy array
            frames = cluster_df.get_column('frame').to_numpy()

            # Find the longest sequences for the cluster
            longest_sequences = find_longest_sequences(frames, n_cluster, vid_length, bins, limit_by_vid_length=False)

            longest_seq_cluster = longest_sequences.get(cluster, [])

            if not longest_seq_cluster:  # Check if the list is empty
                print(f"No sequences found for cluster {cluster}")
                continue

            # Check if the sequence is long enough
            if len(longest_seq_cluster) < min_consecutive_frames:
                continue

            # Process the video for the given sequence
            cropped_frames_list = process_video_group(cfg, cluster_df, videoType, symlinks, longest_seq_cluster[0], longest_seq_cluster[-1])

            # Store the frames in the dictionary
            motif_videos[cluster] = cropped_frames_list

        if cropped_frames_list:
            output_video_path = os.path.join(videos_path, f"motif_{cluster}_clips.mp4")  # Change the file name as needed
            # Create a video from the frames
            write_frames_to_video(cropped_frames_list, output_video_path, fps)

            print(f"Video for cluster {cluster} created at {output_video_path}")


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


def select_long_sequences(df_pandas: pd.DataFrame, fps: int = 30, max_video_length: int = 5) -> tuple:
    """
    Selects long sequences from a pandas DataFrame based on certain criteria and calculates the video length for each motif.
    It also creates bins for sequence lengths and counts the number of sequences in each bin for each motif.

    Args:
    - df_pandas: A pandas DataFrame containing information about sequences, including columns 'sequence_id', 'is_long_sequence', 'motif', 'frame', and 'sequence_length'.
    - fps: An integer representing the frames per second of the video (default is 30).
    - max_video_length: An integer representing the maximum video length in minutes (default is 5).

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
    longest_sequence = max(df_pandas['sequence_length'])

    # Define the bin edges for sequence length bins
    bin_edges = list(range(30, longest_sequence, 10))

    # Create the sequence length bins using the 'pd.cut' function
    sequence_length_bin = pd.cut(df_pandas['sequence_length'], bins=bin_edges)

    # Count the number of sequences in each bin for each motif
    sequences_per_bin_per_motif = df_pandas.groupby(['motif', sequence_length_bin], observed=False).size()

    # Convert the resulting series to a DataFrame and reset the index
    sequences_per_bin_per_motif_df = sequences_per_bin_per_motif.reset_index(name='count')

    # Calculate the maximum video length in frames
    max_video_length_frames = max_video_length * 60 * fps

    # Sort the long sequences DataFrame by 'sequence_length' in descending order
    long_sequences = long_sequences.sort_values('sequence_length', ascending=False)

    return long_sequences, max_video_length_frames






def select_sequences(group: pd.DataFrame, max_length: int, extra_rows: int) -> pd.Series:
    """
    Select sequences from a group based on their length, up to a maximum length.
    It also allows for selecting additional sequences beyond the maximum length if specified.

    Args:
        group (pd.DataFrame): A pandas DataFrame containing the sequences to select from.
                              It should have columns 'sequence_id' and 'sequence_length'.
        max_length (int): The maximum total length of the selected sequences.
        extra_rows (int): The number of additional sequences to select beyond the maximum length.

    Returns:
        pd.Series: A pandas Series containing the IDs of the selected sequences.
    """
    group = group.sort_values('sequence_length', ascending=False)
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


# Group the DataFrame by motif and apply the selection function to each group



def motif_videos_conserved_newest(config, symlinks = False, videoType = '.mp4', fps = 30, bins = 6, maximum_video_length=1000, min_consecutive_frames = 60):
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

    df: Union[pl.DataFrame, pd.DataFrame] = AlHf.create_andOR_get_master_df(config, fps=30, create_new_df=False, df_kind = 'pandas')

    df = find_conserved_motif_sequences(df_pandas = df, sequence_length = 30)  
    
    long_sequences, max_video_length_frames = select_long_sequences(df)
    selected_sequences_per_motif = long_sequences.groupby('motif').apply(select_sequences, max_length=max_video_length_frames, extra_rows = 10)
    
    # TODO: Continue from here!
        
    #df_min_consecutive = find_consecutive_sequences(df, min_length=min_consecutive_frames)
    
    #longest_sequences_df = find_longest_sequence(df_min_consecutive)
    
    if parameterization == 'hmm':
        df_min_consecutive.write_csv(os.path.join(path_to_file, 'results', f"all_minimum_consecutive_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
        longest_sequences_df.write_csv(os.path.join(path_to_file, 'results', f"all_longest_consecutive_sequences_{parameterization}-{n_cluster}-{cfg['hmm_iters']}.csv"))
    else:
        df_min_consecutive.write_csv(os.path.join(path_to_file, 'results', f"all_minimum_consecutive_sequences_{parameterization}-{n_cluster}.csv"))
        longest_sequences_df.write_csv(os.path.join(path_to_file, 'results', f"all_longest_consecutive_sequences_{parameterization}-{n_cluster}.csv"))
    
    #video_df = longest_sequences_df
    video_df = df

    if parameterization == 'hmm':
        videos_path = os.path.join(path_to_file, 'results', 'videos', parameterization+'-'+str(n_cluster)+'-'+str(cfg['hmm_iters']))
    else:
        videos_path = os.path.join(path_to_file, 'results', 'videos', parameterization+'-'+str(n_cluster))
    
    os.makedirs(videos_path, exist_ok=True)
    
    create_videos_for_motifs(video_df, cfg, videoType, symlinks, fps, n_cluster, vid_length, bins, min_consecutive_frames, videos_path)
