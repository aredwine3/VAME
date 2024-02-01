# Making motif videos
import vame
import numpy as np
from importlib import reload
import vame.custom.ALR_video as vid
import polars as pl

config = '/work/wachslab/aredwine3/VAME_working/config_sweep_drawn-sweep-88_cpu_hmm-40-650.yaml'

config = "C:\Users\tywin\VAME\config.yaml"

config = "C:\\Users\\tywin\\VAME\\config.yaml"


config = 'D:\\Users\\tywin\\VAME\\config.yaml'

vid.motif_videos_conserved(config, symlinks=True, videoType='.mp4', fps=30, bins=6, maximum_video_length=10, min_consecutive_frames=30)



# Checking if lesser used motifs will be included in the motif videos when using lesser consecutive frames
vid.motif_videos_conserved(config, symlinks=True, videoType='.mp4', fps=30, bins=6, maximum_video_length=10, min_consecutive_frames=20)