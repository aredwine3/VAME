import os
import matplotlib
# Set the Matplotlib backend based on the environment.
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')  # Use this backend for headless environments (e.g., Google Colab, some remote servers)
else:
    matplotlib.use('Qt5Agg')  # Use this backend for environments with a display server
import matplotlib.pyplot as plt

os.chdir('/Users/adanredwine/VAME')
import vame
import numpy as np
import glob
from vame.util import auxiliary as aux
from vame.analysis import behavior_structure as bs
from vame.custom import helperFunctions as hf
from vame.custom import ACWS_videowriter as avw

#%% Load old project or start new one?
new = False #set to True if you're creating a new project, leave False if resuming an old one.
    
#%% Initialize your project
# Step 1.1:
if new:
    working_directory = '/Volumes/G-DRIVE_SSD/VAME_working'
    project= 'ALR_VAME_1'
    videos = vame.hf.get_video_file_list('/Volumes/Elements/Adan/Open_Arena/Open_Arena_Videos_0-15')
    config = vame.init_new_project(project=project, videos=videos, working_directory=working_directory, videotype='.mp4')
elif not new:
    config = '/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/config.yaml'
working_directory = os.path.dirname(config)
os.chdir(working_directory)

#%% Read paramteters from config
cfg = aux.read_config(config)
n_cluster = cfg['n_cluster']
cluster_method ='kmeans'
projectPath = cfg['project_path']
modelName = cfg['model_name']
vids = cfg['video_sets']
pcutoff = cfg['pose_confidence']


# Convert to individual csv files
vame.hf.convert_multi_csv_to_individual_csv('/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/videos/pose_estimation')

# Create symlinks to the videos to make it seem like the is a video for each animal
vame.hf.create_symlinks('/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/videos', '/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/real_videos')

# Adjust folder structure to match what VAME expects so that each animal has its own folder
vame.hf.create_new_dirs_and_remove_old('/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/data')
vame.hf.create_new_dirs_and_remove_old('/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/results')

# Update config file to reflect new video paths from converting to individual csv files
vame.hf.update_video_sets_in_config('/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/config.yaml', '/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/videos')

"""
Original pose_ref_index:
0:Caudal_Skull_Point		
1:Center_of_Body
2:Center_of_Hips
3:LeftEar
4:LeftHip
5:LeftShoulder
6:Lumbar_Spine_Center
7:Nose
8:RightEar	
9:RightHip
10:RightShoulder
11:ShoulderCenter
12:Tailbase
13:Thoracic_Spine_Center
"""

# Backup the pose_estimation folder
src_dir = '/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/videos/pose_estimation'
dest_dir = '/Volumes/G-DRIVE_SSD/VAME_backups/pose_estimation'
vame.hf.multithreaded_copy(src_dir, dest_dir)

# Rearrange the columns of the csv files so that the order of the bodyparts is along the animals body (head to tail)
body_parts_order = ['Nose', 'Caudal_Skull_Point', 'LeftEar', 'RightEar', 'ShoulderCenter', 'LeftShoulder', 'RightShoulder', 'Thoracic_Spine_Center', 'Center_of_Body', 'Lumbar_Spine_Center', 'Center_of_Hips', 'LeftHip', 'RightHip', 'Tailbase']
vame.hf.rearrange_all_csv_columns('/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/videos/pose_estimation', body_parts_order)

"""
New pose_ref_index:
0:Nose
1:Caudal_Skull_Point
2:LeftEar
3:RightEar
4:ShoulderCenter
5:LeftShoulder
6:RightShoulder
7:Thoracic_Spine_Center
8:Center_of_Body
9:Lumbar_Spine_Center
10:Center_of_Hips
11:LeftHip
12:RightHip
13:Tailbase
"""

# Update backup of the pose_estimation folder
src_dir = '/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/videos/pose_estimation'
dest_dir = '/Volumes/G-DRIVE_SSD/VAME_backups/pose_estimation'
vame.hf.multithreaded_copy(src_dir, dest_dir)

# Align data egocentric
vame.egocentric_alignment(config, pose_ref_index=[0,13], crop_size=(180,180), use_video=False, save=False, video_format='.mp4', check_video=False)

# Rename the files and directories to match the naming convention
dir = '/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/'
vame.hf.rename_files_and_dirs(dir)

# Update config file to reflect new video paths from renaming files and directories
vame.hf.update_video_sets_in_config('/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/config.yaml', '/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/videos')

# Get the information of the videos to help select a diverse range of videos for the training set
video_path = '/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/videos'
vame.hf.write_video_info_to_csv(video_path, 'ALR_video_info.csv')


# Get a random group of videos for the dataset where each group and time point is represented equally
percentage = 7  # % of videos to select
csv_file = 'ALR_video_info.csv'
selected_videos = vame.hf.select_videos(csv_file, percentage)

videos = list(selected_videos)

"""
videos = ['22-01-10_Baseline_1_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat2.mp4', 
 '22-01-10_Baseline_1_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-01-12_Baseline_2_DJL_TABB_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-01-12_Baseline_2_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-01-31_Week_02_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat1.mp4', 
 '22-02-16_Week_04_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat4.mp4', 
 '22-03-02_Week_06_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-03-16_Week_08_DJL_TUBW_cropped_CRF0_0min_to_15min_Rat1.mp4', 
 '22-04-06_Week_11_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat4.mp4', 
 '22-04-19_Week_13_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat1.mp4', 
 '22-05-02_Week_15_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat4.mp4',
 '22-05-05_Drug_Trt_DJL_TABD_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-01-31_Week_02_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat1.mp4', 
 '22-02-14_Week_04_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat2.mp4', 
 '22-02-28_Week_06_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat4.mp4', 
 '22-03-14_Week_08_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat2.mp4', 
 '22-04-04_Week_11_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat4.mp4', 
 '22-04-19_Week_13_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat1.mp4', 
 '22-05-02_Week_15_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat2.mp4', 
 '22-01-10_Baseline_1_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat2.mp4', 
 '22-01-13_Baseline_2_DJL_TSBT_cropped_CRF0_0min_to_15min_Rat4.mp4', 
 '22-05-05_Drug_Trt_DJL_THBI_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-02-02_Week_02_DJL_TUBW_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-02-14_Week_04_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat4.mp4', 
 '22-02-28_Week_06_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat2.mp4', 
 '22-03-14_Week_08_KEN_TKBL_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-04-04_Week_11_DJL_TABB_cropped_CRF0_0min_to_15min_Rat1.mp4', 
 '22-04-19_Week_13_DJL_TSBT_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-05-03_Week_15_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat1.mp4', 
 '22-01-11_Baseline_1_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-01-13_Baseline_2_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-05-05_Drug_Trt_DJL_TJBK_cropped_CRF0_0min_to_15min_Rat2.mp4', 
 '22-02-02_Week_02_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat2.mp4', 
 '22-02-16_Week_04_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat3.mp4', 
 '22-03-02_Week_06_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat4.mp4', 
 '22-03-16_Week_08_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat4.mp4', 
 '22-04-04_Week_11_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat4.mp4', 
 '22-04-19_Week_13_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat4.mp4', 
 '22-05-03_Week_15_DJL_TSBT_cropped_CRF0_0min_to_15min_Rat1.mp4']
"""


# Create your training set
vame.create_trainset(config, check_parameter=False)

"""
Please enter a suffix to denote this training set: 7prcnt
Lenght of train data: 842400
Lenght of test data: 210600
"""

# Train VAME
config = '/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/config.yaml'
vame.train_model(config)

# Making dataset large in order to get a good representation of the data for hyperparameter tuning

# Get a random group of videos for the dataset where each group and time point is represented equally
percentage = 33  # % of videos to select
csv_file = 'ALR_video_info.csv'
selected_videos = vame.hf.select_videos(csv_file, percentage)

videos = list(selected_videos)

"""
['22-03-14_Week_08_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-10_Baseline_1_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat2.mp4','22-01-10_Baseline_1_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-10_Baseline_1_DJL_TABB_cropped_CRF0_0min_to_15min_Rat4.mp4','22-01-10_Baseline_1_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-13_Baseline_2_DJL_TUBW_cropped_CRF0_0min_to_15min_Rat2.mp4','22-01-12_Baseline_2_DJL_TABB_cropped_CRF0_0min_to_15min_Rat4.mp4','22-01-12_Baseline_2_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-13_Baseline_2_DJL_TUBW_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-31_Week_02_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat2.mp4','22-02-02_Week_02_DJL_TUBW_cropped_CRF0_0min_to_15min_Rat2.mp4','22-02-02_Week_02_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat3.mp4','22-01-31_Week_02_DJL_TABB_cropped_CRF0_0min_to_15min_Rat3.mp4','22-02-14_Week_04_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat1.mp4','22-02-16_Week_04_DJL_TUBW_cropped_CRF0_0min_to_15min_Rat1.mp4','22-02-14_Week_04_DJL_TABB_cropped_CRF0_0min_to_15min_Rat3.mp4','22-02-16_Week_04_DJL_TUBW_cropped_CRF0_0min_to_15min_Rat2.mp4','22-03-02_Week_06_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat4.mp4','22-02-28_Week_06_DJL_TABB_cropped_CRF0_0min_to_15min_Rat4.mp4','22-02-28_Week_06_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat1.mp4','22-02-28_Week_06_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat2.mp4','22-03-14_Week_08_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat2.mp4','22-03-14_Week_08_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat2.mp4','22-03-16_Week_08_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat4.mp4','22-03-14_Week_08_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat1.mp4','22-04-06_Week_11_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat1.mp4','22-04-06_Week_11_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat4.mp4','22-04-04_Week_11_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat2.mp4','22-04-04_Week_11_DJL_TABB_cropped_CRF0_0min_to_15min_Rat4.mp4','22-04-19_Week_13_DJL_TUBW_cropped_CRF0_0min_to_15min_Rat2.mp4','22-04-18_Week_13_DJL_TABB_cropped_CRF0_0min_to_15min_Rat4.mp4','22-04-18_Week_13_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat2.mp4','22-04-18_Week_13_DJL_TABB_cropped_CRF0_0min_to_15min_Rat3.mp4','22-05-02_Week_15_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat4.mp4','22-05-02_Week_15_DJL_TABB_cropped_CRF0_0min_to_15min_Rat4.mp4','22-05-02_Week_15_DJL_TQBR_cropped_CRF0_0min_to_15min_Rat1.mp4','22-05-02_Week_15_DJL_TABB_cropped_CRF0_0min_to_15min_Rat3.mp4','22-01-11_Baseline_1_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-11_Baseline_1_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat2.mp4','22-01-10_Baseline_1_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-10_Baseline_1_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat4.mp4','22-01-12_Baseline_2_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat4.mp4','22-01-12_Baseline_2_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat4.mp4','22-01-12_Baseline_2_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat2.mp4','22-01-12_Baseline_2_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat2.mp4','22-05-05_Drug_Trt_DJL_TABD_cropped_CRF0_0min_to_15min_Rat3.mp4','22-05-05_Drug_Trt_DJL_TJBK_cropped_CRF0_0min_to_15min_Rat3.mp4','22-05-05_Drug_Trt_DJL_THBI_cropped_CRF0_0min_to_15min_Rat1.mp4','22-05-05_Drug_Trt_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-31_Week_02_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat3.mp4','22-01-31_Week_02_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat2.mp4','22-01-31_Week_02_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-31_Week_02_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat1.mp4','22-02-14_Week_04_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat4.mp4','22-02-14_Week_04_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat3.mp4','22-02-14_Week_04_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat4.mp4','22-02-16_Week_04_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat2.mp4','22-02-28_Week_06_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat4.mp4','22-02-28_Week_06_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat2.mp4','22-03-02_Week_06_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat1.mp4','22-02-28_Week_06_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat3.mp4','22-03-14_Week_08_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat4.mp4','22-03-14_Week_08_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat3.mp4','22-03-14_Week_08_KEN_TKBL_cropped_CRF0_0min_to_15min_Rat1.mp4','22-04-04_Week_11_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat3.mp4','22-04-04_Week_11_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat1.mp4','22-04-04_Week_11_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat4.mp4','22-04-04_Week_11_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat3.mp4','22-04-18_Week_13_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat4.mp4','22-04-19_Week_13_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat1.mp4','22-04-18_Week_13_DJL_TEBF_cropped_CRF0_0min_to_15min_Rat1.mp4','22-04-18_Week_13_DJL_TGBH_cropped_CRF0_0min_to_15min_Rat3.mp4','22-05-02_Week_15_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat4.mp4','22-05-02_Week_15_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat1.mp4','22-05-02_Week_15_DJL_TCBD_cropped_CRF0_0min_to_15min_Rat3.mp4','22-05-02_Week_15_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat2.mp4','22-01-10_Baseline_1_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat4.mp4','22-01-10_Baseline_1_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat3.mp4','22-01-11_Baseline_1_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat2.mp4','22-01-11_Baseline_1_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-13_Baseline_2_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-12_Baseline_2_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat3.mp4','22-01-13_Baseline_2_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat2.mp4','22-01-12_Baseline_2_DJL_TABB_cropped_CRF0_0min_to_15min_Rat2.mp4','22-05-05_Drug_Trt_DJL_TTBW_cropped_CRF0_0min_to_15min_Rat3.mp4','22-05-05_Drug_Trt_DJL_TLBM_cropped_CRF0_0min_to_15min_Rat2.mp4','22-05-05_Drug_Trt_DJL_TTBW_cropped_CRF0_0min_to_15min_Rat1.mp4','22-05-05_Drug_Trt_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-31_Week_02_DJL_TABB_cropped_CRF0_0min_to_15min_Rat2.mp4','22-02-02_Week_02_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-31_Week_02_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat1.mp4','22-02-02_Week_02_DJL_TSBT_cropped_CRF0_0min_to_15min_Rat3.mp4','22-02-14_Week_04_DJL_TABB_cropped_CRF0_0min_to_15min_Rat2.mp4','22-02-16_Week_04_DJL_TUBW_cropped_CRF0_0min_to_15min_Rat4.mp4','22-02-16_Week_04_DJL_TUBW_cropped_CRF0_0min_to_15min_Rat3.mp4','22-02-16_Week_04_DJL_TSBT_cropped_CRF0_0min_to_15min_Rat3.mp4','22-03-02_Week_06_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat1.mp4','22-03-02_Week_06_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat2.mp4','22-02-28_Week_06_DJL_TABB_cropped_CRF0_0min_to_15min_Rat2.mp4','22-02-28_Week_06_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat4.mp4','22-03-16_Week_08_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat1.mp4','22-03-14_Week_08_DJL_TABB_cropped_CRF0_0min_to_15min_Rat1.mp4','22-03-14_Week_08_DJL_TABB_cropped_CRF0_0min_to_15min_Rat2.mp4','22-03-16_Week_08_DJL_TSBT_cropped_CRF0_0min_to_15min_Rat3.mp4','22-04-04_Week_11_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat2.mp4','22-04-06_Week_11_DJL_TSBT_cropped_CRF0_0min_to_15min_Rat3.mp4','22-04-04_Week_11_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat3.mp4','22-04-04_Week_11_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat4.mp4','22-04-19_Week_13_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat2.mp4','22-04-18_Week_13_DJL_TABB_cropped_CRF0_0min_to_15min_Rat1.mp4','22-04-18_Week_13_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat3.mp4','22-04-19_Week_13_DJL_TSBT_cropped_CRF0_0min_to_15min_Rat3.mp4','22-05-03_Week_15_DJL_TSBT_cropped_CRF0_0min_to_15min_Rat3.mp4','22-05-02_Week_15_DJL_TABB_cropped_CRF0_0min_to_15min_Rat1.mp4','22-05-03_Week_15_DJL_TUBW_cropped_CRF0_0min_to_15min_Rat3.mp4','22-05-02_Week_15_DJL_TKBL_cropped_CRF0_0min_to_15min_Rat3.mp4','22-01-10_Baseline_1_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat3.mp4','22-01-11_Baseline_1_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat3.mp4','22-01-11_Baseline_1_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat4.mp4','22-01-11_Baseline_1_DJL_TSBT_cropped_CRF0_0min_to_15min_Rat2.mp4','22-01-13_Baseline_2_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat4.mp4','22-01-13_Baseline_2_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat1.mp4','22-01-13_Baseline_2_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat4.mp4','22-01-13_Baseline_2_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat3.mp4','22-05-05_Drug_Trt_DJL_TPBS_cropped_CRF0_0min_to_15min_Rat1.mp4','22-05-05_Drug_Trt_DJL_TNBO_cropped_CRF0_0min_to_15min_Rat1.mp4','22-05-05_Drug_Trt_DJL_TJBK_cropped_CRF0_0min_to_15min_Rat2.mp4','22-05-05_Drug_Trt_DJL_TPBS_cropped_CRF0_0min_to_15min_Rat4.mp4','22-02-02_Week_02_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat4.mp4','22-02-02_Week_02_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat3.mp4','22-01-31_Week_02_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat3.mp4','22-02-02_Week_02_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat3.mp4','22-02-14_Week_04_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat3.mp4','22-02-16_Week_04_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat4.mp4','22-02-16_Week_04_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat2.mp4','22-02-16_Week_04_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat3.mp4','22-02-28_Week_06_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat4.mp4','22-03-02_Week_06_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat3.mp4','22-03-02_Week_06_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat2.mp4','22-03-02_Week_06_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat3.mp4','22-03-16_Week_08_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat3.mp4','22-03-16_Week_08_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat4.mp4','22-03-16_Week_08_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat4.mp4','22-03-16_Week_08_DJL_TOBP_cropped_CRF0_0min_to_15min_Rat3.mp4','22-04-06_Week_11_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat4.mp4','22-04-06_Week_11_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat3.mp4','22-04-06_Week_11_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat3.mp4','22-04-04_Week_11_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat3.mp4','22-04-19_Week_13_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat1.mp4','22-04-18_Week_13_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat4.mp4','22-04-19_Week_13_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat3.mp4','22-04-19_Week_13_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat4.mp4','22-05-02_Week_15_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat4.mp4','22-05-02_Week_15_DJL_TIBJ_cropped_CRF0_0min_to_15min_Rat4.mp4','22-05-03_Week_15_DJL_TXBY_cropped_CRF0_0min_to_15min_Rat4.mp4','22-05-02_Week_15_DJL_TMBN_cropped_CRF0_0min_to_15min_Rat2.mp4']
"""

config = '/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/config.yaml'
vame.create_trainset(config, check_parameter=False)

# On the HCC
config = '/work/wachslab/aredwine3/VAME_working/config_fabric_3.yaml'
vame.create_trainset(config, check_parameter=False, data_fraction=0.15)

vame.train_model(config)

vame.pose_segmentation(config)


# Need to adjust symlinks on HCC to point to the correct data
# This will make a new directory called "real_videos" and move all video files to it,
# then create symlinks in the videos directory to the real_videos directory. This prevents
# the user from needing a copy of multianimal video sfor each animal
vame.hf.create_symlinks('/work/wachslab/aredwine3/VAME_working/videos', '/work/wachslab/aredwine3/VAME_working/real_videos')

import vame
from vame.util import auxiliary as aux
from vame.analysis import behavior_structure as bs
from vame.custom import helperFunctions as hf
from vame.custom import ACWS_videowriter as avw


config = '/work/wachslab/aredwine3/VAME_working/config_fabric_3.yaml'
modelName='VAME_15prcnt'

# Hyperparameters not yet optimized, this is just to get a feel for the process and data

hf.replace_date_underscores('/work/wachslab/aredwine3/VAME_working/videos')
hf.replace_date_underscores('/work/wachslab/aredwine3/VAME_working/real_videos')

avw.motif_videos(config, extractData=True, model_name=modelName, cluster_method='hmm', videoType='.mp4', fps=30, symlinks=True)

# After completion of 15 % Hyperparameter Sweep Optimization

# Evaluating the best models

# Change all hyperparameters in the config file to match the model you are evaluating
# Save as a new config file with sweep name


## NOTE: Evaluating the models from the hyperparameter sweep will not work because I did not save the model state dicts during training.add()

# Models were retrained with hyperparameters from the sweeps and can be evaluated instead:
''' Approaches:
Option 1: The Iterative Approach
Initial Training and Validation: Start by training the HMM on the 15% data subset that your neural network was trained on. Validate the HMM on a separate, held-out set if possible.
Evaluation on Full Dataset: Apply this trained HMM to the embeddings of the full dataset and evaluate its performance.
Assessment: If the performance is good, you can consider the problem solved. If not, move to the next steps.
Option 2: Comprehensive Training
Retrain Neural Network: Retrain your neural network on the full dataset to get more representative embeddings.
Train HMM: Train the HMM on these new embeddings.
Validation: Validate this new HMM model on a separate, held-out set to ensure it generalizes well.
Option 3: Hybrid Training
Partial Retraining: Use a portion of the 85% remaining data to fine-tune the neural network.
HMM Training: Train the HMM on a combination of the original 15% and some new data.
Validation: Validate this new HMM model on a separate, held-out set to ensure it generalizes well.
General Recommendations
Validation is Key: Regardless of the path you choose, always validate the models. Given that you're working on a complex problem involving time-series data and behavioral motifs, ensuring that your models generalize well is critical.
Interpretable Models: Since you're working on identifying behavioral motifs, make sure that the HMM states are interpretable and meaningful in the context of your research.
Computational Resources: Consider the computational costs associated with each option. Retraining models on large datasets may require significant resources.
'''


config = '/work/wachslab/aredwine3/VAME_working/config_sweep_drawn-sweep-88.yaml'

modelName='VAME_15prcnt_sweep_drawn-sweep-88'

vame.evaluate_model(config, model_name=modelName, use_snapshots=True)

"""Approach 1 on drawn-sweep-88"""




config = '/work/wachslab/aredwine3/VAME_working/config_sweep_sunny-sweep-140.yaml'

modelName='VAME_15prcnt_sweep_sunny-sweep-140'



