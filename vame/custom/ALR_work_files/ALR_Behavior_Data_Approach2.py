import os
import sys
from importlib import reload
from sqlite3 import Time
from typing import Union

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
from cv2 import sumElems
from matplotlib.pyplot import cla
from rich import pretty

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

#! ANIMAL A1 NEEDS TO BE DROPPED FROM THE ANALYSIS AT THE BEGINNING OF THE PIPELINE

df_hmm_650: Union[pl.DataFrame, pd.DataFrame] = AlHf.create_andOR_get_master_df(
    config, fps=30, create_new_df=False, df_kind="pandas"
)

# Drop A1
df_hmm_650 = df_hmm_650.query("rat_id != 'A1'")

classifications = pd.read_csv("/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/baseline_summed_df_hmm.csv")

# Keep the following columns: 'Motif', 'Moving_Quickly', 'Predominant_Behavior', 'Secondary_Descriptor', 'Category', 'Exclude', 'Cluster', 'Desc_Behavior'
classifications = classifications[['Motif', 'Moving_Quickly', 'Predominant_Behavior', 'Secondary_Descriptor', 'Category', 'Exclude', 'Cluster', 'Desc_Behavior']]

# Only keep the unique rows
classifications = classifications.drop_duplicates()

# Rename the Motif column to motif
classifications.rename(columns={"Motif": "motif"}, inplace=True)

# Merge the classifications with the data
df_hmm_650 = df_hmm_650.merge(classifications, on="motif", how="left")

# Save dmm_hmm_650 to a csv file
df_hmm_650.to_csv("C:\\Users\\tywin\\VAME\\needed_files\\ALR_hmm_650.csv")

df_hmm_650.to_csv("/Users/adanredwine/Desktop/ALR_hmm_650.csv")

""" Making motif_usages (if not imported )"""

# This can easily be made with hf.combineBehavior()
motif_usage_hmm_650 = df_hmm_650.copy()

# Strip all leading or trailing white spaces from column names
motif_usage_hmm_650.columns = motif_usage_hmm_650.columns.str.strip()

df_class_temp = motif_usage_hmm_650.groupby(["file_name", "motif"]).size().reset_index()

df_class_temp.columns = ["file_name", "motif", "value"]

df_pivot = df_class_temp.pivot(index='motif', columns='file_name', values='value')

# Fill any NaN values with 0
df_pivot.fillna(0, inplace=True)

motif_usage_hmm_650 = df_pivot.copy()

""" End Making motif_usages  """

# Read in the manual motif identifications to merge with the data
df_classifications = pd.read_csv(
    "/Users/adanredwine/Desktop/needed_files/ALR_manual_motif_to_behavior_classification_hmm_650.csv"
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
    df.drop(columns=["Motif"], inplace=True)

motif_usage_hmm_650 = hf.combineBehavior(
    config, save=True, cluster_method="hmm", legacy=False
)

meta_data = AlHf.create_meta_data_df(motif_usage_hmm_650)

# Rename the 'Name' column to 'File_Name' in the meta_data data frame
meta_data.rename(columns={"Name": "File_Name"}, inplace=True)

df.drop_duplicates()

# Capitalist all first letters of the column names
df.columns = [col.replace("_", " ").title().replace(" ", "_") for col in df.columns]

df.rename(
    columns={"Animal_Id": "Animal_ID"},
    inplace=True,
)

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
# Rename Rat_Id to Animal_ID
df.rename(columns={"Rat_Id": "Animal_ID"}, inplace=True)

# Determine the total usage of each motif for each animal at each time point
temp_series = df.groupby(["Animal_ID", "Time_Point", "Motif"]).size()

# Convert the series to a data frame, giving the column a name
per_video_total_motif_usage = (temp_series.to_frame("Value")).reset_index()

meta_data.columns = meta_data.columns.str.strip()
per_video_total_motif_usage.columns = per_video_total_motif_usage.columns.str.strip()

# Add the Group and Treatment_State columns to the data frame by mathcing the "Animal_ID" column from df
per_video_total_motif_usage = per_video_total_motif_usage.merge(
    meta_data[["File_Name", "Animal_ID", "Group", "Treatment_State", "Time_Point"]],
    left_on=["Animal_ID", "Time_Point"],
    right_on=["Animal_ID", "Time_Point"],
    how="right",
)

# Extract the integer from the 'Time_Point' column and convert to numeric
per_video_total_motif_usage["Time_Point_Int"] = pd.to_numeric(
    per_video_total_motif_usage["Time_Point"].str.extract(r"(\d+)")[0], errors="coerce"
)

# Apply the conditions to correctly set the 'Treatment_State' column
per_video_total_motif_usage["Treatment_State"] = np.where(
    (per_video_total_motif_usage["Group"] == "Sham"),
    "Sham",
    np.where(
        (per_video_total_motif_usage["Group"] == "Injured"),
        "Injured",
        np.where(
            (per_video_total_motif_usage["Group"] == "ABX"),
            "ABX",
            np.where(
                (per_video_total_motif_usage["Group"] == "Treated")
                & (per_video_total_motif_usage["Time_Point_Int"] > 8),
                "Treated",
                "Injured",
            ),
        ),
    ),
)

# Drop the 'Time_Point_Int' column
per_video_total_motif_usage.drop(columns=["Time_Point_Int"], inplace=True)

# Add Group, Treatment_State, Moving_Quickly, Predominant_Behavior, Secondary_Descriptor, and Category to the data frame by merging on Animal_ID, Time_Point, and Motif
merged_df = per_video_total_motif_usage.merge(
    df[
        [
            "File_Name",
            "Animal_ID",
            "Time_Point",
            "Motif",
            "Group",
            "Moving_Quickly",
            "Predominant_Behavior",
            "Secondary_Descriptor",
            "Category",
            "Exclude",
        ]
    ],
    left_on=["Animal_ID", "Time_Point", "Motif"],
    right_on=["Animal_ID", "Time_Point", "Motif"],
    how="left",
).drop_duplicates()

# Drop any rows where "Exclude" is True
merged_df = merged_df.query("Exclude == False")
# Assign each Predominant_Behavior a positive integer, starting at 0
merged_df["Predominant_Behavior_Manual_Cluster"] = (
    merged_df["Predominant_Behavior"].astype("category").cat.codes + 1
)

# Ensure that each predominant behavior was always assigned the same integer
assert (
    merged_df.groupby("Predominant_Behavior")["Predominant_Behavior_Manual_Cluster"]
    .nunique()
    .max()
    == 1
)

# Get a list of columns that end with "_x"
columns_to_drop = [col for col in merged_df.columns if col.endswith("_x")]

# Drop these columns
merged_df = merged_df.drop(columns=columns_to_drop)

# Rename the remaining columns to remove the "_y" suffix
merged_df.columns = merged_df.columns.str.replace("_y$", "", regex=True)

# if Animal_ID "A1" is in the data frame, drop it
if "A1" in merged_df["Animal_ID"].unique():
    merged_df = merged_df.query("Animal_ID != 'A1'")


# Reset the index
summed_df = merged_df.reset_index(drop=True)

normality_vsBase_dict_State = {
    (row["Time_Point"], row["Treatment_State"]): row["p"] > 0.05
    for index, row in normality_df.iterrows()
}

# Add the column "Is_Normal_State" to the summed_df data frame by applying the normality_vsBase_dict_State dictionary
summed_df["Is_Normal_State"] = summed_df.apply(
    lambda row: normality_vsBase_dict_State[
        (row["Time_Point"], row["Treatment_State"])
    ],
    axis=1,
)


pVals_summed_df_base_State_notNormalized = AlSt.calculate_p_values_vs_baseline_State(
    summed_df, normalized=False, log_comp=False
)

# Adjust p-values for multiple comparisons
import statsmodels.stats.multitest as smm

pVals_summed_df_base_State_notNormalized["p_adjusted"] = smm.multipletests(
    pVals_summed_df_base_State_notNormalized["P_Value"], method="bonferroni"
)[1]

# Filter the pVals_summed_df_base_State_notNormalized data frame to only include the rows where the "Treatment_State" column is "Sham"
pVals_summed_df_base_State_notNormalized_sham = (
    pVals_summed_df_base_State_notNormalized.query("Treatment_State == 'Sham'")
)

# Check if any of the p-values are less than 0.05 for every time point and every motif
if pVals_summed_df_base_State_notNormalized_sham["p_adjusted"].lt(0.05).any():
    console.print(
        "Sham data significantly changes over time, cannot normalize to sham (or can we....?)"
    )
    # Return all of the rows where the p-value is less than 0.05
    pVals_summed_df_base_State_notNormalized_sham.query("p_adjusted < 0.05")

# ------------------------------------------------------------------------------------ #
# Mean and Standard Deviation of each motif at each time point for each Tretment_State #
# ------------------------------------------------------------------------------------ #

baseline_time_points = ["Baseline_1", "Baseline_2"]

baselines = summed_df[(summed_df["Time_Point"].isin(baseline_time_points))]

State_baseline_means = (
    baselines.groupby(["Animal_ID", "Motif"])["Value"].mean()
).reset_index()

State_baseline_means.columns = ["Animal_ID", "Motif", "Mean_Value_Baseline_ID"]

# Drop the rows where "Treatment_State" is "Sham" and "Time_Point" is in baseline_time_points
no_baseline_df = summed_df.drop(
    index=summed_df.query("Time_Point in @baseline_time_points").index
)

# Add a new column 'Time_Point' to 'State_baseline_means' and set it to 'Week_00'
State_baseline_means["Time_Point"] = "Week_00"

# Rename 'Mean_Value_Baseline_ID' to 'Value' in 'State_baseline_means' to match 'no_baseline_df'
State_baseline_means.rename(columns={"Mean_Value_Baseline_ID": "Value"}, inplace=True)

# Concatenate 'no_baseline_df' and 'State_baseline_means'
no_baseline_df = pd.concat([no_baseline_df, State_baseline_means], ignore_index=True)

sham_animal_ids = no_baseline_df[no_baseline_df["Treatment_State"] == "Sham"]["Animal_ID"].unique()

State_baseline_means["Animal_ID"].isin(sham_animal_ids).any()

# Define the columns to be filled for each grouping variable
animal_id_cols = ["Treatment_State", "Group"]
motif_cols = [
    "Moving_Quickly",
    "Predominant_Behavior",
    "Secondary_Descriptor",
    "Category",
    "Exclude",
    "Predominant_Behavior_Manual_Cluster",
]

# Fill NaN values for 'Treatment_State' and 'Group' based on the mode of each 'Animal_ID' group
for col in animal_id_cols:
    no_baseline_df[col] = no_baseline_df.groupby("Animal_ID")[col].apply(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
    ).reset_index(level=0, drop=True)

no_baseline_df[(no_baseline_df["Time_Point"] == "Week_00") & (no_baseline_df["Group"] == "Sham")]["Animal_ID"].isin(sham_animal_ids).count()

# Extract the integer from the 'Time_Point' column and convert to numeric
no_baseline_df["Time_Point_Int"] = pd.to_numeric(
    no_baseline_df["Time_Point"].str.extract("(\d+)")[0], errors="coerce"
)

# Apply the conditions to correctly set the 'Treatment_State' column
no_baseline_df["Treatment_State"] = np.where(
    (no_baseline_df["Group"] == "Sham"),
    "Sham",
    np.where(
        (no_baseline_df["Group"] == "Injured"),
        "Injured",
        np.where(
            (no_baseline_df["Group"] == "ABX"),
            "ABX",
            np.where(
                (no_baseline_df["Group"] == "Treated")
                & (no_baseline_df["Time_Point_Int"] > 8),
                "Treated",
                "Injured",
            ),
        ),
    ),
)

# Print all the unique "Treatment_State" 's in no_baseline_df when the "Time_Point_Int" is <= 8
if len(no_baseline_df.query("(Treatment_State == 'Treated') & (Time_Point_Int <= 8)")):
    console.print(
        "The adjustment for Treatment_State values is not correct for the first 8 weeks of treatment"
    )

# Drop the 'Time_Point_Int' column
no_baseline_df.drop(columns=["Time_Point_Int"], inplace=True)

# Fill NaN values for 'Moving_Quickly', 'Predominant_Behavior', 'Category', 'Exclude', 'Predominant_Behavior_Manual_Cluster' based on the mode of each 'Motif' group
for col in motif_cols:
    no_baseline_df[col] = no_baseline_df.groupby("Motif")[col].apply(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
    ).reset_index(level=0, drop=True)
    

# Fill 'Is_Normal_State' with False
no_baseline_df["Is_Normal_State"] = no_baseline_df["Is_Normal_State"].fillna(False)

# If "Moving_Quickly" is NaN, set it to "False"
no_baseline_df["Moving_Quickly"] = no_baseline_df["Moving_Quickly"].fillna(False)

# If "Moving_Quickly" is 0.0 or 1.0, set it to False or True, respectively
no_baseline_df["Moving_Quickly"] = no_baseline_df["Moving_Quickly"].astype(bool)

no_baseline_df = no_baseline_df[(no_baseline_df["Exclude"] == False)]

no_baseline_df[(no_baseline_df["Treatment_State"] == "Sham")]["Time_Point"].unique()

summed_baseline_df = no_baseline_df.copy()

# Print the unique values in the "Time_Point" column, ensure Week_00 is present
console.print(summed_baseline_df["Time_Point"].unique())

grouped_columns = ["Motif", "Treatment_State", "Time_Point"]

averaged_motif_usage_df = AlSt.calculate_mean_and_sd(
    summed_baseline_df, group_columns=grouped_columns, normalization=False
)

# Create a barchart of a specific motif at all time points, color by Treatment_State
# This is a good way to visualize the distribution of motifs over time
import plotly.express as px

# * Use averaged_motif_usage_df instead of summed_baseline_df if making barplots instead of boxplots

summed_baseline_df[(summed_baseline_df["Treatment_State"] == "Sham") & (summed_baseline_df["Time_Point"] == "Week_00")]


for i in range(summed_baseline_df["Motif"].nunique()):
    fig = AlSt.plot_motif_data(summed_baseline_df, motif=i, cluster=None, plot_type='boxplot', y_label='Motif Frequency', title= 'Usage Over Time for Motif ' + str(i))
    fig_name = 'q1_hmm_Motif_Usage_Boxplot_' + str(i) + '.html'
    fig_path = "/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/motif_usage_plots_hmm"
    fig.write_html(fig_path + '/' + fig_name)


# -------------------------------------------------------------------------------------- #
# Mean and Standard Deviation of each cluster at each time point for each Tretment_State #
# -------------------------------------------------------------------------------------- #

grouped_columns[0] = "Predominant_Behavior_Manual_Cluster"

# This determines which clusters for each group and time point do not have enough values to calculate a standard deviation
cluster_usage_counts = (
    summed_baseline_df.groupby(grouped_columns)["Value"]
    .count()
    .reset_index()
    .rename(columns={"Value": "Count"})
    .query("Count < 2")
    .reset_index(drop=True)
    .copy()
)

averaged_cluster_usage_df = AlSt.calculate_mean_and_sd(
    summed_baseline_df, group_columns=grouped_columns, normalization=False
)

# Extract all rows where std is NaN 
nan_std_rows = averaged_cluster_usage_df[averaged_cluster_usage_df["std"].isna()] #* This confirms what was seen above

# Rename the Predominant_Behavior_Manual_Cluster column to Cluster
summed_baseline_df.rename(columns={"Predominant_Behavior_Manual_Cluster": "Cluster"}, inplace=True)

fig = AlSt.plot_motif_data(df=summed_baseline_df, motif=None, cluster=2.0, plot_type = "violin", y_label="ylab", title="temp")
fig.show()


for i in range(summed_baseline_df["Cluster"].nunique()):
    behavior = summed_baseline_df[summed_baseline_df["Cluster"] == i]["Predominant_Behavior"].unique()
    fig = AlSt.plot_motif_data(summed_baseline_df, motif=None, cluster=i, plot_type='boxplot', y_label='Cluster Frequency', title= 'Usage Over Time for Cluster ' + str(i) +', Behavior:'+ str(behavior))
    fig_name = 'q1_hmm_Cluster_Usage_Boxplot_' + str(i) + '.html'
    fig_path = "/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/motif_usage_plots_hmm"
    fig.write_html(fig_path + '/' + fig_name)
    

q1_stats_df = summed_baseline_df.copy()

# Drop "ABX" and "Treated" "Treatment_States"
q1_stats_df = q1_stats_df[(q1_stats_df["Treatment_State"]!= "ABX") & (q1_stats_df["Treatment_State"]!= "Treated")]

import scipy.stats

normality_results = []
clusters = q1_stats_df["Cluster"].unique()

for cluster in clusters:
    # Create a dataframe for each cluster
    cluster_df = q1_stats_df[q1_stats_df["Cluster"] == cluster]
    console.print(cluster_df)
    shapiro_results = scipy.stats.shapiro(cluster_df['Value'])
    
    normality_results.append(
        {
            "Cluster": cluster,
            "Behavior": cluster_df["Predominant_Behavior"].unique()[0],
            #"Treatment_State": state,
            #"Time_Point": time_point,
            "W": shapiro_results[0],
            "p": shapiro_results[1],
        }
    )

# Save the normality results
pd.DataFrame(normality_results).to_csv("/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/q1_clusters_normality_results_hmm.csv")

cluster_stats_vs_sham = AlSt.calculate_p_values_vs_sham_State(q1_stats_df, motif_or_cluster="cluster", log_comp=False, normalized=False)


df["Predominant_Behavior"].unique()