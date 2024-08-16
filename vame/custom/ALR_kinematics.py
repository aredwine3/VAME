import concurrent
import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from icecream import ic
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

import vame.custom.ALR_analysis as AlAn
import vame.custom.ALR_helperFunctions as AlHf
from vame.util.auxiliary import read_config


def get_files(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)

    files = []
    if cfg['all_data'] == 'No' or cfg['all_data']=='no':
        all_flag = input("Do you want to write motif videos for your entire dataset? \n"
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
    
    return files, cfg


def get_dlc_file(path_to_file, dlc_data_type, file):
    dataFile = glob.glob(os.path.join(path_to_file,'videos','pose_estimation',file+'*.csv'))
    
    if len(dataFile)>1:
        raise KeyError("Multiple csv files match video filename")
    else:
        dataFile=dataFile[0]
        
    if dlc_data_type.lower() == 'n':
        data = pd.read_csv(dataFile, skiprows=0, index_col=0, header=[1,2])
    if dlc_data_type.lower() == 'y':
        data = pd.read_csv(dataFile, index_col=0, header=[0, 1])
        
    return data
    

def calculate_centroid(data):
    """
    Calculate the centroid coordinates (x, y) for each frame in a given dataset.

    Args:
        data (DataFrame): A DataFrame containing x and y coordinates for each frame. The DataFrame should have a multi-level column index with 'coords' as one level and 'x' or 'y' as the other level.

    Returns:
        centroid_x (Series): A Series containing the centroid x-coordinates for each frame.
        centroid_y (Series): A Series containing the centroid y-coordinates for each frame.
    """

    # Calculate the centroid coordinates (x, y) for each frame
    # Extract all 'x' and 'y' coordinates
    x_coords = data.xs('x', level='coords', axis=1)
    y_coords = data.xs('y', level='coords', axis=1)

    # Calculate the centroid for each frame (mean across all body parts)
    centroid_x = x_coords.mean(axis=1)
    centroid_y = y_coords.mean(axis=1)

    centroid_x = centroid_x.interpolate().ffill().bfill()
    centroid_y = centroid_y.interpolate().ffill().bfill()

    return centroid_x, centroid_y


def calculate_speed_with_spline(data, fps, window_size=5, pixel_to_cm=0.215):
    """
    Calculate the speed of an object in centimeters per second using spline interpolation.

    Args:
        data (DataFrame): A DataFrame containing x and y coordinates for each frame.
                          The DataFrame should have a multi-level column index with 'coords' as one level
                          and 'x' or 'y' as the other level.
        fps (float): The frames per second of the video.
        window_size (int, optional): The size of the rolling window used for smoothing the speed. Default is 5.
        pixel_to_cm (float, optional): The conversion factor from pixels to centimeters. Default is 0.215.

    Returns:
        ndarray: A 1-dimensional array containing the smoothed speed values in centimeters per second.
    """

    # Calculate the centroid for each frame
    centroid_x, centroid_y = calculate_centroid(data)
    
    # Calculate the displacement between consecutive frames
    delta_x = np.diff(centroid_x, prepend=centroid_x.iloc[0])
    delta_y = np.diff(centroid_y, prepend=centroid_y.iloc[0])
    
    # Calculate the distance moved between consecutive frames in pixels
    distance_pixels = np.sqrt(delta_x**2 + delta_y**2)
    
    # Convert distance from pixels to centimeters
    distance_cm = distance_pixels * pixel_to_cm
    
    # Calculate the speed in cm/s
    speed_cm_per_s = distance_cm * fps
    
    # Fit a spline to the speed data
    frames_to_fit = min(len(speed_cm_per_s), 10)  # Ensure we don't exceed the length of the speed array
    
    spline = UnivariateSpline(np.arange(frames_to_fit), speed_cm_per_s[:frames_to_fit], s=0, k=3)
    
    # Use the spline to estimate the speed at the first frame
    estimated_first_speed = spline(0)
    
    # Smooth the speed using a rolling window
    speed_smoothed = pd.Series(speed_cm_per_s).rolling(window=window_size, min_periods=1).mean().to_numpy()
    
    # Replace the first speed value with the estimated value from the spline
    speed_smoothed[0] = estimated_first_speed
    
    # Replace NaN values with 0
    if np.isnan(speed_smoothed).any():
        print("Speed data contains NaN values. Handling...")
        #speed_smoothed = np.nan_to_num(speed_smoothed)
        speed_smoothed = pd.Series(speed_smoothed).interpolate().bfill().ffill().to_numpy()

    return speed_smoothed


def plot_speed(speed_data, fps):
    """
    Creates a plot of the speed of the animal over time.

    Parameters
    ----------
    speed_data : array_like
        The speed data to plot.
    fps : int
        The frames per second of the video from which the speed was calculated.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """

    # Calculate the time for each frame
    time = np.arange(len(speed_data)) / fps

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 5))  # Adjust the figsize to make the x-axis wider
    ax.plot(time, speed_data, label='Speed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (cm/s)')
    ax.set_title('Animal Speed Over Time')
    ax.legend()

    return fig

def plot_location_heatmap(data):
    centroid_x, centroid_y = calculate_centroid(data)

    # Assuming centroid_x and centroid_y are the arrays of x and y coordinates of the animal's centroid
    heatmap, xedges, yedges = np.histogram2d(centroid_x, centroid_y, bins=(50, 50))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Heatmap of Animal Position')

    return plt

    
def plot_speed_spectrogram(speed, fps):
    from scipy.signal import spectrogram

    # Calculate the spectrogram
    frequencies, times, Sxx = spectrogram(speed, fs=fps)

    # Plot the spectrogram
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto')
    plt.colorbar()
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram of Speed')
    
    return plt


def plot_speed_histogram(speed_data, bins=30):
    """
    Plots a histogram of the speed data.

    Parameters
    ----------
    speed_data : array_like
        The speed data to plot.
    bins : int, optional
        The number of bins for the histogram. Default is 30.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the histogram.
    """
    fig, ax = plt.subplots()
    ax.hist(speed_data, bins=bins, color='blue', edgecolor='black')
    ax.set_title('Speed Histogram')
    ax.set_xlabel('Speed (cm/s)')
    ax.set_ylabel('Frequency')

    return fig


def calculate_kinematics(config, fps=30, window_size=5, plot=False):
    files, cfg = get_files(config)
    path_to_file = cfg['project_path']
    dlc_data_type = input("Were your DLC .csv files originally multi-animal? (y/n): ")

    # Use ProcessPoolExecutor to parallelize file processing
    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor for each file
        futures = [executor.submit(process_file, file, cfg, path_to_file, dlc_data_type, fps, window_size, plot) for file in files]

        # Optionally, collect results or handle exceptions
        for future in as_completed(futures):
            try:
                # Result from process_file (if any)
                result = future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')


def process_file(file, cfg, path_to_file, dlc_data_type, fps, window_size, plot):
    if cfg['parameterization'] == 'hmm':
        save_path = os.path.join(cfg['project_path'], 'results', file, cfg['model_name'], cfg['load_data'], cfg['parameterization'] + '-' + str(cfg['n_cluster']) + '-' + str(cfg['hmm_iters']))
    else:
        save_path = os.path.join(cfg['project_path'], 'results', file, cfg['model_name'], cfg['load_data'], cfg['parameterization'] + '-' + str(cfg['n_cluster']))
    
    if not os.path.exists(os.path.join(save_path, "kinematics")):
        os.mkdir(os.path.join(save_path, "kinematics"))

    data = get_dlc_file(path_to_file, dlc_data_type, file)
    speed = calculate_speed_with_spline(data, fps, window_size)
    rat_in_center = is_dat_rat_in_center(data)
    distance_FrameByFrame = distance_traveled(data, pixel_to_cm=0.215)

    np.save(os.path.join(save_path, "kinematics", f"{file}-window-{window_size}-speed.npy"), speed)
    np.save(os.path.join(save_path, "kinematics", f"{file}-isDatRat_in_center.npy"), rat_in_center)
    np.save(os.path.join(save_path, "kinematics", f"{file}-distance_FrameByFrame.npy"), distance_FrameByFrame)

    if plot:
        speed_plot = plot_speed(speed, fps)
        plot_save_path = os.path.join(save_path, "kinematics", f"{file}-window-{window_size}-speed.svg")
        speed_plot.savefig(plot_save_path)
        
        plt.close('all')

        location_heatmap = plot_location_heatmap(data) 
        plot_save_path = os.path.join(save_path, "kinematics", f"{file}-location_heatmap.svg")
        location_heatmap.savefig(plot_save_path)

        plt.close('all')

        spectrogram_plot = plot_speed_spectrogram(speed, fps)
        plot_save_path = os.path.join(save_path, "kinematics", f"{file}-spectrogram_plot.png")
        spectrogram_plot.savefig(plot_save_path)
        
        plt.close('all')

        speed_histogram = plot_speed_histogram(speed, bins=30)
        plot_save_path = os.path.join(save_path, "kinematics", f"{file}-speed_histogram.png")
        speed_histogram.savefig(plot_save_path)

        plt.close('all')
    


def get_dat_rat(data):
    # Calculate the mean centroid for the dataset
    centroid_x, centroid_y = calculate_centroid(data)
    mean_centroid_x = centroid_x.mean()
    mean_centroid_y = centroid_y.mean()

    rat_boundaries = {
        "Rat1": {"x": (0, 332.5), "y": (0, 328.5)},
        "Rat2": {"x": (332.5, 680), "y": (0, 328.5)},
        "Rat3": {"x": (0, 332.5), "y": (328.5, 680)},
        "Rat4": {"x": (332.5, 680), "y": (328.5, 680)},
    }

    for rat, boundaries in rat_boundaries.items():
        if (boundaries['x'][0] <= mean_centroid_x <= boundaries['x'][1]) and (boundaries['y'][0] <= mean_centroid_y <= boundaries['y'][1]):
            return rat
    raise ValueError("Mean centroid coordinates do not fall within any rat boundaries")
 


def is_dat_rat_in_center(data):
    """
    Check if the centroid coordinates of a given dataset fall within the center zone for the identified rat.

    Args:
        data (DataFrame): A DataFrame containing x and y coordinates for each frame.

    Returns:
        numpy array: An array of boolean values indicating if each frame's centroid coordinates fall within the center zone for the identified rat.
    """

    rat_boundaries = {
        "Rat1": {"x": (0, 332.5), "y": (0, 328.5)},
        "Rat2": {"x": (332.5, 680), "y": (0, 328.5)},
        "Rat3": {"x": (0, 332.5), "y": (328.5, 680)},
        "Rat4": {"x": (332.5, 680), "y": (328.5, 680)},
    }
    wall_edges = {
        "Rat1": {
            'TL': {'x': 40, 'y': 41},
            'TR': {'x': 320, 'y': 41},
            'BL': {'x': 40, 'y': 319},
            'BR': {'x': 320, 'y': 319}
        },
        "Rat2": {
            'TL': {'x': 338, 'y': 34},
            'TR': {'x': 612, 'y': 40},
            'BL': {'x': 340, 'y': 320},
            'BR': {'x': 618, 'y': 322}
        },
        "Rat3": {
            'TL': {'x': 39, 'y': 339},
            'TR': {'x': 325, 'y': 340},
            'BL': {'x': 42, 'y': 611},
            'BR': {'x': 321, 'y': 616}
        },
        "Rat4": {
            'TL': {'x': 339, 'y': 339},
            'TR': {'x': 615, 'y': 340},
            'BL': {'x': 336, 'y': 616},
            'BR': {'x': 610, 'y': 616}
        }
    }

    center_zone_percentage = 0.5  # Define the percentage of the arena that makes up the center zone
    center_zones = {}
    for rat, edges in wall_edges.items():
        x_center_Top = (edges['TR']['x'] + edges['TL']['x']) / 2
        x_center_Bottom = (edges['BR']['x'] + edges['BL']['x']) / 2
        
        y_center_L = (edges['TL']['y'] + edges['BL']['y']) / 2
        y_center_R = (edges['TR']['y'] + edges['BR']['y']) / 2
        
        x_center = (x_center_Top + x_center_Bottom) / 2
        y_center = (y_center_L + y_center_R) / 2
        
        width = ((edges['TR']['x'] - edges['TL']['x']) + (edges['BR']['x'] - edges['BL']['x'])) / 2 * center_zone_percentage
        height = ((edges['BL']['y'] - edges['TL']['y']) + (edges['BR']['y'] - edges['TR']['y'])) / 2 * center_zone_percentage

        center_zones[rat] = {
            "x": (x_center - width / 2, x_center + width / 2),
            "y": (y_center - height / 2, y_center + height / 2)
        }

    rat = get_dat_rat(data)
    centroid_x, centroid_y = calculate_centroid(data)
    
    # Check if the centroid for each frame is within the center zone for the identified rat
    in_center = []
    for frame in range(len(centroid_x)):
        x_in_center = center_zones[rat]['x'][0] <= centroid_x[frame] <= center_zones[rat]['x'][1]
        y_in_center = center_zones[rat]['y'][0] <= centroid_y[frame] <= center_zones[rat]['y'][1]
        in_center.append(x_in_center and y_in_center)

    return np.array(in_center)


def distance_traveled(data, pixel_to_cm=0.215):
    # Calculate the centroid for each frame
    centroid_x, centroid_y = calculate_centroid(data)
    
    # Calculate the displacement between consecutive frames
    delta_x = np.diff(centroid_x, prepend=centroid_x.iloc[0])
    delta_y = np.diff(centroid_y, prepend=centroid_y.iloc[0])
    
    # Calculate the distance moved between consecutive frames in pixels
    distance_pixels = np.sqrt(delta_x**2 + delta_y**2)
    
    # Convert distance from pixels to centimeters
    distance_cm = distance_pixels * pixel_to_cm
    
    # Return the distance traveled frame by frame in centimeters
    return distance_cm


def calculate_torso_angle(data):
    # Extract the needed points
    hips_center_x = data[('Center_of_Hips', 'x')]
    hips_center_y = data[('Center_of_Hips', 'y')]

    torso_center_x = data[('Center_of_Body', 'x')]
    torso_center_y = data[('Center_of_Body', 'y')]

    shoulder_center_x = data[('ShoulderCenter', 'x')]
    shoulder_center_y = data[('ShoulderCenter', 'y')]

    V1 = np.array([torso_center_x - shoulder_center_x, torso_center_y - shoulder_center_y])
    V2 = np.array([torso_center_x - hips_center_x, torso_center_y - hips_center_y])
    
    # Calculate the angle
    dot_product = np.sum(V1 * V2, axis=0)
    
    angle = np.arccos(dot_product / (np.linalg.norm(V1, axis=0) * np.linalg.norm(V2, axis=0)))

    return np.degrees(angle)     


def calculate_torso_length(data, pixel_to_cm=0.215):
    # Extract the needed points
    hips_center_x = data[('Center_of_Hips', 'x')]
    hips_center_y = data[('Center_of_Hips', 'y')]

    lumbar_center_x = data[('Lumbar_Spine_Center', 'x')]
    lumbar_center_y = data[('Lumbar_Spine_Center', 'y')]

    torso_center_x = data[('Center_of_Body', 'x')]
    torso_center_y = data[('Center_of_Body', 'y')]

    thoracic_center_x = data[('Thoracic_Spine_Center', 'x')]
    thoracic_center_y = data[('Thoracic_Spine_Center', 'y')]

    shoulder_center_x = data[('ShoulderCenter', 'x')]
    shoulder_center_y = data[('ShoulderCenter', 'y')]

    V1 = np.array([lumbar_center_x - hips_center_x, lumbar_center_y - hips_center_y])
    V2 = np.array([torso_center_x - lumbar_center_x, torso_center_y - lumbar_center_y])
    V3 = np.array([thoracic_center_x - torso_center_x, thoracic_center_y - torso_center_y])
    V4 = np.array([shoulder_center_x - thoracic_center_x, shoulder_center_y - thoracic_center_y])
    
    # Calculate the length
    length = np.linalg.norm(V1, axis=0) + np.linalg.norm(V2, axis=0) + np.linalg.norm(V3, axis=0) + np.linalg.norm(V4, axis=0)
    
    # Convert length from pixels to centimeters
    length_cm = length * pixel_to_cm
    
    return length_cm

def normalize_centroid_positions(centroid_x, centroid_y, rat_boundaries, rat):
    # Get the boundaries for the specific rat
    boundaries = rat_boundaries[rat]
    
    # Normalize the centroid coordinates
    normalized_centroid_x = centroid_x - boundaries['x'][0]
    normalized_centroid_y = centroid_y - boundaries['y'][0]
    
    return normalized_centroid_x, normalized_centroid_y


def create_group_heatmap(group_data, rat_boundaries, global_min_x, global_max_x, global_min_y, global_max_y, global_max_heatmap_value):
    # Initialize lists to hold normalized positions for the group
    group_centroid_x = []
    group_centroid_y = []
    
    # Normalize centroid positions for each rat and combine them
    for data in group_data:
        centroid_x, centroid_y = calculate_centroid(data)
        rat = get_dat_rat(data)
        normalized_x, normalized_y = normalize_centroid_positions(centroid_x, centroid_y, rat_boundaries, rat)
        group_centroid_x.extend(normalized_x)
        group_centroid_y.extend(normalized_y)
    
    # Now create the heatmap with the combined normalized positions
    heatmap, xedges, yedges = np.histogram2d(group_centroid_x, group_centroid_y, bins=(50, 50))
    extent = [global_min_x, global_max_x, global_min_y, global_max_y]

    ic(global_max_heatmap_value)

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', interpolation='nearest', vmax=global_max_heatmap_value)
    plt.colorbar()
    plt.xlabel('Normalized X Position (pixels)')
    plt.ylabel('Normalized Y Position (pixels)')
    plt.title('Heatmap of Grouped Animal Position')

    return plt



def plot_distance_traveled(travel_data, fps=30):
    """
    Plots the total distance traveled against time in minutes.
    If travel_data is a dictionary, plots the mean distance traveled with standard deviation as error bars over time for each group.

    Args:
        travel_data (array-like, list of arrays, or dict): An array containing the distance traveled frame by frame in centimeters,
                                                           a list where each element is an array for a single dataset,
                                                           or a dictionary with group names as keys and lists of arrays as values.
        fps (int): The frames per second of the video.
    """
    plt.figure(figsize=(10, 5))

    if isinstance(travel_data, dict):
        group_colors = {'sham': '#2ca02c', 'abx': '#1f77b4', 'treat': '#b662ff', 'inj': '#d42163'}
        # Handle dictionary input with multiple groups
        for group_name, group_data in travel_data.items():
            # Find the minimum length of the arrays in group_data
            min_length = min(len(data) for data in group_data)

            # Truncate or pad the arrays in group_data to have the same length
            adjusted_group_data = [data[:min_length] if len(data) >= min_length else np.pad(data, (0, min_length - len(data)), 'constant') for data in group_data]

            # Calculate the cumulative sum of distances for each dataset
            cumulative_distances = [np.cumsum(data) for data in adjusted_group_data]

            # Calculate the mean and standard deviation across all datasets
            mean_distances = np.mean(cumulative_distances, axis=0)
            std_distances = np.std(cumulative_distances, axis=0)

            # Calculate the time for each frame in minutes
            time_in_minutes = np.arange(min_length) / fps / 60

            group_color = group_colors.get(group_name, 'gray')

            # Plot the mean distance traveled with a semi-transparent band for standard deviation
            plt.fill_between(time_in_minutes, mean_distances - std_distances, mean_distances + std_distances, color=group_color, alpha=0.5)
            plt.plot(time_in_minutes, mean_distances, 'o-', label=f'{group_name} Mean Total Distance Traveled', color=group_color)
    else:
        # Handle non-dictionary input for a single dataset or group
        if isinstance(travel_data[0], list) or isinstance(travel_data[0], np.ndarray):
            # Assume grouped data (list of arrays)
            grouped = True
        else:
            # Single dataset (single array)
            grouped = False
            travel_data = [travel_data]

        if grouped:
            # Find the minimum length of the arrays in travel_data
            min_length = min(len(data) for data in travel_data)

            # Truncate or pad the arrays in travel_data to have the same length
            adjusted_travel_data = [data[:min_length] if len(data) >= min_length else np.pad(data, (0, min_length - len(data)), 'constant') for data in travel_data]

            # Calculate the cumulative sum of distances for each dataset
            cumulative_distances = [np.cumsum(data) for data in adjusted_travel_data]

            # Calculate the mean and standard deviation across all datasets
            mean_distances = np.mean(cumulative_distances, axis=0)
            std_distances = np.std(cumulative_distances, axis=0)

            # Calculate the time for each frame in minutes
            time_in_minutes = np.arange(min_length) / fps / 60
            # Plot the mean distance traveled with a semi-transparent band for standard deviation
            plt.fill_between(time_in_minutes, mean_distances - std_distances, mean_distances + std_distances, color='grey', alpha=0.5)
            plt.plot(time_in_minutes, mean_distances, 'o-', label='Mean Total Distance Traveled')
            # Plot the mean distance traveled against time in minutes
            #plt.errorbar(time_in_minutes, mean_distances, yerr=std_distances, label='Mean Total Distance Traveled', fmt='-o')
        else:
            # Single dataset
            total_distance_traveled = np.cumsum(travel_data[0])

            # Calculate the time for each frame in minutes
            time_in_minutes = np.arange(len(travel_data[0])) / fps / 60

            # Plot the total distance traveled against time in minutes
            plt.plot(time_in_minutes, total_distance_traveled, 'o-', label='Total Distance Traveled')

    # Label the axes
    plt.xlabel('Time (minutes)')
    plt.ylabel('Distance Traveled (cm)')

    # Add a title and legend
    plt.title('Distance Traveled Over Time')
    plt.legend()

    return plt


def group_week_kinematics(config, filetype ='.pdf'):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']
    parameterization = cfg['parameterization']

    rat_boundaries = {
        "Rat1": {"x": (0, 332.5), "y": (0, 328.5)},
        "Rat2": {"x": (332.5, 680), "y": (0, 328.5)},
        "Rat3": {"x": (0, 332.5), "y": (328.5, 680)},
        "Rat4": {"x": (332.5, 680), "y": (328.5, 680)},
    }
    

    organized_files = AlAn.group_week_files(config)
    

    files = AlHf.get_files(config)

    path_to_file = cfg['project_path']
    dlc_data_type = input("Were your DLC .csv files originally multi-animal? (y/n): ")
    
    path = os.path.join(cfg['project_path'], 'results')
    if parameterization == 'hmm':
        aggregated_analysis_path = os.path.join(path, "aggregated_analysis", parameterization+'-'+str(n_cluster)+'-'+str(hmm_iters))
    else:
        aggregated_analysis_path = os.path.join(path, "aggregated_analysis", parameterization+'-'+str(n_cluster))

    plot_box_and_whisker_plot_total_distance_traveled(organized_files, aggregated_analysis_path, path_to_file, dlc_data_type, fps=30, pixel_to_cm=0.215, filetype='.pdf')

    global_min_x, global_max_x = float('inf'), float('-inf')
    global_min_y, global_max_y = float('inf'), float('-inf')
    global_max_heatmap_value = 0
    
    for file in files:
        data = get_dlc_file(path_to_file, dlc_data_type, file)
        centroid_x, centroid_y = calculate_centroid(data)
        rat = get_dat_rat(data)
        normalized_x, normalized_y = normalize_centroid_positions(centroid_x, centroid_y, rat_boundaries, rat)
        # Update global min and max
        global_min_x = min(global_min_x, normalized_x.min())
        global_max_x = max(global_max_x, normalized_x.max())
        global_min_y = min(global_min_y, normalized_y.min())
        global_max_y = max(global_max_y, normalized_y.max())
        heatmap, xedges, yedges = np.histogram2d(normalized_x, normalized_y, bins=(50, 50))
        global_max_heatmap_value = max(global_max_heatmap_value, heatmap.max())        
    
    for week in organized_files.keys():
        week_path = os.path.join(aggregated_analysis_path, week)
        os.makedirs(week_path, exist_ok=True)
        
        week_data = []
        
        grouped_travel_data = {}

        # Iterate over each group within this week
        for group_name, files in organized_files[week].items():
            if week == 'Drug_Trt' and group_name == 'abx':
                continue            

            group_path = os.path.join(week_path, group_name)
            os.makedirs(group_path, exist_ok=True)

            group_week_data = []
            group_week_speed_data = []
            group_week_travel_data = []
            
            for file in files:
                data = get_dlc_file(path_to_file, dlc_data_type, file)                
                group_week_data.append(data)

                speed_data = calculate_speed_with_spline(data, fps = 30, window_size=5, pixel_to_cm=0.215)
                group_week_speed_data.append(speed_data)
                
                travel_data = distance_traveled(data, pixel_to_cm=0.215)
                group_week_travel_data.append(travel_data)                
            
            np.save(os.path.join(group_path, f"speed_data_{group_name}.npy"), group_week_speed_data)
            np.save(os.path.join(group_path, f"travel_data_{group_name}.npy"), group_week_travel_data)

            fig = create_group_heatmap(group_week_data, rat_boundaries, global_min_x, global_max_x, global_min_y, global_max_y, global_max_heatmap_value)
            fig.savefig(os.path.join(group_path, f"mean_heatmap_for_{group_name}_{week}{filetype}"))
            plt.close('fig')

            fig = plot_distance_traveled(group_week_travel_data, fps=30)
            fig.savefig(os.path.join(group_path, f"distance_traveled_over_time_for_{group_name}_{week}{filetype}"))
            plt.close('fig')            
            
            grouped_travel_data[group_name] = group_week_travel_data

            # Append the data from the current group to the week_data
            week_data.extend(group_week_data)

        fig = plot_distance_traveled(grouped_travel_data, fps=30)
        fig.savefig(os.path.join(week_path, f"distance_traveled_over_time_for_all_groups_{week}{filetype}"))
        plt.close('fig')

        
        fig = create_group_heatmap(week_data, rat_boundaries, global_min_x, global_max_x, global_min_y, global_max_y, global_max_heatmap_value)
        fig.savefig(os.path.join(week_path, f"mean_heatmap_for_{week}{filetype}"))
        plt.close('fig')




def plot_box_and_whisker_plot_total_distance_traveled(organized_files, aggregated_analysis_path, path_to_file, dlc_data_type, fps, pixel_to_cm, filetype):
    all_group_data = []
    # y-normalized to baseline 1 and baseline 2 values

    # y-normalized to sham values:


    for week in organized_files.keys():
        week_data = []

        # Iterate over each group within this week
        for group_name, files in organized_files[week].items():
            group_week_travel_data = []

            for file in files:
                data = get_dlc_file(path_to_file, dlc_data_type, file)
                travel_data = distance_traveled(data, pixel_to_cm=pixel_to_cm)
                group_week_travel_data.append(travel_data.sum())  # Sum the total distance traveled for each file

            # Create a DataFrame for the current group and week
            group_data = pd.DataFrame({
                'Week': week,
                'Group': group_name,
                'TotalDistance': group_week_travel_data
            })
            all_group_data.append(group_data)

    # Combine all group data into a single DataFrame
    all_data_df = pd.concat(all_group_data)

    # Create the box and whisker plot
    plt.figure(figsize=(10, 5))
    boxplot = all_data_df.boxplot(
        by=['Week', 'Group'],
        column=['TotalDistance'],
        grid=False,
        return_type='dict'
    )

    # Set plot title and labels
    plt.title('Total Distance Traveled by Week and Group')
    plt.suptitle('')  # Suppress the automatic 'Group by' title
    plt.xlabel('Week and Group')
    plt.ylabel('Total Distance Traveled (cm)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Save the plot
    plt.savefig(os.path.join(aggregated_analysis_path, f"boxplot_total_distance_traveled{filetype}"))
    plt.close()

