import os
import sys
from importlib import reload
from typing import Union

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
from rich import pretty
from ydata_profiling import ProfileReport

import vame
import vame.custom.ALR_analysis as ana
import vame.custom.ALR_helperFunctions as AlHf
import vame.custom.ALR_statsFunctions as AlSt
from vame.custom import helperFunctions as hf
from vame.custom.ALR_latent_vector_cluster_functions import (
    calculate_mean_latent_vector_for_motifs,
    create_tsne_projection,
    create_umap_projection,
)
from vame.util.auxiliary import read_config

pretty.install()
from rich.console import Console

console = Console()


# --------------------------------------------------------------------------- #
# Section 1: Is sham data normal at each time point?
# --------------------------------------------------------------------------- #

# - Checking the normality of the sham data at each time point
# (grouping all motifs together) will help us decide whether to
# use parametric or non-parametric tests for the sham data.

# Define the path to the config file
config = "D:\\Users\\tywin\\VAME\\config.yaml"


df_hmm_650: Union[pl.DataFrame, pd.DataFrame] = AlHf.create_andOR_get_master_df(
    config, fps=30, create_new_df=False, df_kind="pandas"
)

# Read in the manual motif identifications to merge with the data
df_classifications = pd.read_csv(
    "D:\\Users\\tywin\\VAME\\results\\videos\\hmm-40-650\\ALR_manual_motif_to_behavior_classification_hmm_650.csv"
)

df = df_hmm_650.merge(
    df_classifications[
        [
            "Motif",
            "Exclude",
            "Moving Quickly",
            "Predominant Behavior",
            "Secondary Descriptor",
            "Category",
        ]
    ],
    left_on="motif",
    right_on="Motif",
    how="left",
).drop_duplicates()

df.columns = df.columns.str.replace(" ", "_")

if df.motif.equals(df.Motif):
    df.drop(columns=["motif"], inplace=True)

motif_usage_hmm_650 = hf.combineBehavior(
    config, save=True, cluster_method="hmm", legacy=False
)

meta_data = AlHf.create_meta_data_df(motif_usage_hmm_650)
# Rename the 'Name' column to 'File_Name' in the meta_data data frame
meta_data.rename(columns={"Name": "File_Name"}, inplace=True)

df.rename(
    columns={"file_name": "File_Name", "motif": "Motif", "rat_id": "Animal_ID"},
    inplace=True,
)

df.drop_duplicates()

# Capitalist all first letters of the column names
df.columns = [col.replace("_", " ").title().replace(" ", "_") for col in df.columns]

normality_df = AlSt.check_normality_by_group_and_time(
    motif_usage_hmm_650, meta_data, type_c="Treatment_State"
)


non_normal_sham_time_points = normality_df.query(
    "Treatment_State == 'Sham' and p > 0.05"
)

if non_normal_sham_time_points is not None:
    console.print(
        "Sham data is not normal at all time points, cannot use parametric tests"
    )

# --------------------------------------------------------------------------- #
# Check if Sham data significantly changes over time
# --------------------------------------------------------------------------- #
