# Making motif videos
import vame
import numpy as np
from importlib import reload
import vame.custom.ALR_video as vid
import polars as pl

config = '/work/wachslab/aredwine3/VAME_working/config_sweep_drawn-sweep-88_cpu_hmm-40-650.yaml'

vid.motif_videos_conserved_newest(config, symlinks = True, videoType = '.mp4', fps = 30, bins = 6, maximum_video_length=1000, min_consecutive_frames = 30)

vid.motif_videos_conserved(config, symlinks = True, videoType = '.mp4', fps = 30, bins = 6, maximum_video_length=1000, min_consecutive_frames = 30)

vid.motif_videos_conserved_OG(config, symlinks = False, videoType = '.mp4', fps = 30, bins = 6, maximum_video_length=1000, min_consecutive_frames = 60)

# *** NOTE: The following code was performed in the terminal to organize the creation of motif videos ***

df = AlHf.create_andOR_get_master_df(config)

# Add a column to the dataframe called 'sequence_id' and fill it with 'NaN'
df = df.with_columns(pl.Series("sequence_id", [np.nan]*len(df)))

# Group the dataframe by file_name, rat_id, and time_point and sort each group by increasing "frame" number
grouped_df = df.group_by(['file_name', 'rat_id', 'time_point']).sort('frame')

