import concurrent.futures
import csv
import glob
import logging
import os
import pickle
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from importlib import reload
from pathlib import Path
from typing import List, Union

import community as community_louvain
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import scipy
import scipy.cluster.hierarchy as sch
import scipy.signal
import seaborn as sns
import umap
from icecream import ic
from matplotlib.pylab import f
from mpl_toolkits.mplot3d import Axes3D
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import dendrogram, fcluster, to_tree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import kruskal, mannwhitneyu, ttest_ind
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    jaccard_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

import vame.custom.ALR_analysis as ana
import vame.custom.ALR_helperFunctions as AlHf
import vame.custom.ALR_kinematics as kin
import vame.custom.ALR_plottingFunctions as AlPf
from vame.analysis.behavior_structure import get_adjacency_matrix
from vame.analysis.community_analysis import compute_transition_matrices, get_labels
from vame.analysis.pose_segmentation import load_latent_vectors
from vame.analysis.tree_hierarchy import (
    draw_tree,
    graph_to_tree,
    hierarchy_pos,
    traverse_tree_cutline,
)
from vame.util.auxiliary import read_config

reload(AlPf)
from rich import pretty

pretty.install()
from rich.console import Console

console = Console()

# Set the Matplotlib backend based on the environment.
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use(
        "Agg"
    )  # Use this backend for headless environments (e.g., Google Colab, some remote servers)
else:
    matplotlib.use("Qt5Agg")  # Use this backend for environments with a display server


""" NORMALIZATION FUNCTIONS """


def handle_missing_baseline_values(df_long, baseline_time_points):
    df_long = df_long.copy()
    from rich.console import Console

    console = Console()
    # Get all of the unique animal IDs
    animal_ids = df_long["Animal_ID"].unique()

    # Get all of the unique motifs
    all_motifs = df_long["Motif"].unique()

    # Define a set to store all motifs without baseline
    all_motifs_without_baseline = set()

    # Check if animal ids don't have a value for each motif in at least one of the baseline time points
    for animal_id in animal_ids:
        # Get all of the motifs for the animal_id in the baseline time points
        motifs_with_baseline = df_long[
            (df_long["Animal_ID"] == animal_id)
            & (df_long["Time_Point"].isin(baseline_time_points))
        ]["Motif"].unique()
        # Get the motifs that don't have a value for the animal_id in at least one of the baseline time points
        motifs_without_baseline = [
            motif for motif in all_motifs if motif not in motifs_with_baseline
        ]
        if len(motifs_without_baseline) > 0:
            # If there are motifs without a baseline value, set the Normalized_Value to NaN
            df_long.loc[
                (df_long["Animal_ID"] == animal_id)
                & (df_long["Motif"].isin(motifs_without_baseline)),
                "Normalized_Value",
            ] = np.nan

            console.print(motifs_without_baseline, style="bold red")
            # Update the set of all motifs without baseline
            all_motifs_without_baseline.update(motifs_without_baseline)

    # If there were any motifs without a baseline value, remove all rows with that Motif from the data frame
    df_long = df_long[~df_long["Motif"].isin(all_motifs_without_baseline)]

    return df_long


def log_normalize_values(df_long):
    df_long = df_long.copy()
    if (df_long["value"] <= 0).any():
        # Handle non-positive values. Here we add a small constant (e.g., 1) to all values.
        df_long["log_value"] = np.log(df_long["value"] + 1)
    else:
        # If all values are positive, we can directly apply the log transformation
        df_long["log_value"] = np.log(df_long["value"])

    return df_long


def normalize_to_baseline_log(df_long):
    baseline_time_points = ["Baseline_1", "Baseline_2"]
    df_long = df_long.copy()
    df_long = handle_missing_baseline_values(df_long, baseline_time_points)
    # Calculate the baseline mean for log-transformed values
    if (df_long["value"] <= 0).any():
        df_long["log_value"] = np.log(df_long["value"] + 1)  # Adding 1 to avoid log(0)
    else:
        df_long["log_value"] = np.log(df_long["value"])

    # Calculate the baseline mean for log-transformed values only for baseline time points
    baseline_means = (
        df_long[df_long["Time_Point"].isin(baseline_time_points)]
        .groupby(["Motif", "Animal_ID"])["log_value"]
        .mean()
        .reset_index(name="Baseline_Log_Mean")
    )

    # Merge the baseline means back onto the original dataframe
    df_long = df_long.merge(baseline_means, on=["Motif", "Animal_ID"], how="left")

    # Normalize the log-transformed values to the log-transformed baseline mean
    df_long["Log_Normalized_Value"] = (
        df_long["log_value"] - df_long["Baseline_Log_Mean"]
    )

    # When the Time_Point is Baseline_1 or Baseline_2, the Log_Normalized_Value should be 0
    df_long.loc[
        df_long["Time_Point"].isin(baseline_time_points), "Log_Normalized_Value"
    ] = 0

    return df_long


def normalize_to_baseline(df_long):
    baseline_time_points = ["Baseline_1", "Baseline_2"]
    df_long = df_long.copy()
    df_long = handle_missing_baseline_values(df_long, baseline_time_points)

    df_long["Baseline_Mean"] = (
        df_long.loc[df_long["Time_Point"].isin(baseline_time_points)]
        .groupby(["Motif", "Animal_ID"])["value"]
        .transform("mean")
    )

    # For each animal, motif, and timepoint, ensure the baseline mean is filled in
    df_long["Baseline_Mean"] = df_long.groupby(["Motif", "Animal_ID"])[
        "Baseline_Mean"
    ].transform(lambda x: x.fillna(x.mean()))

    # Fill in any remaining NaN values with 1
    df_long["Baseline_Mean"] = df_long["Baseline_Mean"].fillna(1)

    # Normalize the values to the baseline mean
    df_long["Normalized_Value"] = df_long["value"] / df_long["Baseline_Mean"]

    # When the Time_Point is Baseline_1 or Baseline_2, the Normalized_Value should be 1
    df_long.loc[
        df_long["Time_Point"].isin(baseline_time_points), "Normalized_Value"
    ] = 1

    return df_long


def normalize_to_baseline_sham_log(df_long):
    df_long = df_long.copy()
    baseline_time_points = ["Baseline_1", "Baseline_2"]
    df_long = handle_missing_baseline_values(df_long, baseline_time_points)

    # Calculate the baseline mean for log-transformed values
    if (df_long["value"] <= 0).any():
        df_long["log_value"] = np.log(df_long["value"] + 1)  # Adding 1 to avoid log(0)
    else:
        df_long["log_value"] = np.log(df_long["value"])

    # Step 2: Filter Sham Group Data and compute log-transformed means
    sham_df = df_long[df_long["Group"] == "Sham"]
    sham_means = (
        sham_df.groupby(["Time_Point", "Motif"])["log_value"]
        .mean()
        .reset_index(name="Sham_Log_Mean")
    )

    # Step 3: Merge Sham Log Means with Original Data
    df_long = df_long.merge(sham_means, on=["Time_Point", "Motif"], how="left")

    # Step 4: Normalize Log-Transformed Values
    df_long["Log_Normalized_Value"] = df_long["log_value"] - df_long["Sham_Log_Mean"]

    # Set Log_Normalized_Value to 0 for the sham group to represent no change
    df_long.loc[df_long["Group"] == "Sham", "Log_Normalized_Value"] = 0

    return df_long


def normalize_to_baseline_sham(df_long):
    df_long = df_long.copy()
    baseline_time_points = ["Baseline_1", "Baseline_2"]
    df_long = handle_missing_baseline_values(df_long, baseline_time_points)
    # Step 1: Filter Sham Group Data
    sham_df = df_long[df_long["Group"] == "Sham"]

    # Step 2: Compute Sham Group Means
    sham_means = (
        sham_df.groupby(["Time_Point", "Motif"])["value"]
        .mean()
        .reset_index(name="Sham_Mean")
    )

    # Step 3: Merge Sham Means with Original Data
    df_long = df_long.merge(sham_means, on=["Time_Point", "Motif"], how="left")

    # Step 4: Normalize Values
    df_long["Normalized_Value"] = df_long["value"] / df_long["Sham_Mean"]

    # Set Sham_Normalized_Value to 1 for the sham group
    df_long.loc[df_long["Group"] == "Sham", "Normalized_Value"] = 1

    return df_long


def calculate_mean_and_sd(df_long, normalization=True, type="Group"):
    if type == "Group":
        if normalization:
            # Calculate the mean and standard deviation for each motif, time point, and group
            stats = (
                df_long.groupby(["Motif", "Time_Point", "Group"])["Normalized_Value"]
                .agg(["mean", "std"])
                .reset_index()
            )
        else:
            stats = (
                df_long.groupby(["Motif", "Time_Point", "Group"])["value"]
                .agg(["mean", "std"])
                .reset_index()
            )

    elif type == "State":
        if normalization:
            # Calculate the mean and standard deviation for each motif, time point, and group
            stats = (
                df_long.groupby(["Motif", "Time_Point", "Treatment_State"])[
                    "Normalized_Value"
                ]
                .agg(["mean", "std"])
                .reset_index()
            )
        else:
            stats = (
                df_long.groupby(["Motif", "Time_Point", "Treatment_State"])["value"]
                .agg(["mean", "std"])
                .reset_index()
            )

        # Sort the Time_Point values
        sorted_time_points = sorted(stats["Time_Point"].unique())
        stats["Time_Point"] = pd.Categorical(
            stats["Time_Point"], categories=sorted_time_points, ordered=True
        )

    return stats


# Assuming distance_traveled_normality_results is your dictionary
def shapiro_normality_results_to_polars_df(normality_results):
    # Prepare a list to collect rows for the DataFrame
    rows_list = []

    for group, time_points in normality_results.items():
        for time_point, (stat, p_value) in time_points.items():
            # Append a dictionary for each row
            rows_list.append(
                {
                    "group": group,
                    "time_point": time_point,
                    "shapiro_stat": stat,
                    "shapiro_p_value": p_value,
                }
            )

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
    total_distance_by_individual = master_df.groupby(
        ["group", "time_point", "rat_id"]
    ).agg(pl.col(col_to_test).sum().alias("total_distance"))

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
                normality_results[group][time_point] = (
                    None,
                    None,
                )  # Not enough data for the test

    return normality_results


def check_normality_group_timepoint_motif(df, meta_data):
    normality_results = []
    # number of motifs
    num_motifs = df.shape[0]

    # Normality in Each Group and
    # Iterate over each unique group and time point and motif combination
    for group in meta_data["Group"].unique():
        for time_point in meta_data["Time_Point"].unique():
            # Filter meta_data for the current group and time point
            filtered_meta = meta_data[
                (meta_data["Group"] == group) & (meta_data["Time_Point"] == time_point)
            ]

            # Iterate over each motif
            for motif in range(df.shape[0]):
                # Collect data for the current motif from all relevant columns
                motif_data = df.loc[motif, filtered_meta["File_Name"]]

                # Fill any NaN values with 0
                motif_data = np.nan_to_num(motif_data)

                # Check if there are enough data points for the test
                if len(motif_data) >= 3:
                    # Perform the Shapiro-Wilk test
                    shapiro_results = scipy.stats.shapiro(motif_data)
                    # Add the results to the DataFrame
                    normality_results.append(
                        {
                            "Motif": motif,
                            "Group": group,
                            "Time_Point": time_point,
                            "W": shapiro_results[0],
                            "p": shapiro_results[1],
                        }
                    )
                else:
                    # Handle cases with insufficient data
                    normality_results.append(
                        {
                            "Motif": motif,
                            "Group": group,
                            "Time_Point": time_point,
                            "W": None,
                            "p": None,
                        }
                    )

    return pd.DataFrame(normality_results)


def check_normality_state_timepoint_motif(df, meta_data):
    normality_results = []
    # number of motifs
    num_motifs = df.shape[0]

    # Normality in Each Group and
    # Iterate over each unique group and time point and motif combination
    for state in meta_data["Treatment_State"].unique():
        for time_point in meta_data["Time_Point"].unique():
            # Filter meta_data for the current group and time point
            filtered_meta = meta_data[
                (meta_data["Treatment_State"] == state)
                & (meta_data["Time_Point"] == time_point)
            ]

            # Iterate over each motif
            for motif in range(df.shape[0]):
                # Collect data for the current motif from all relevant columns
                motif_data = df.loc[motif, filtered_meta["File_Name"]]

                # Fill any NaN values with 0
                motif_data = np.nan_to_num(motif_data)

                # Check if there are enough data points for the test
                if len(motif_data) >= 3:
                    # Perform the Shapiro-Wilk test
                    shapiro_results = scipy.stats.shapiro(motif_data)
                    # Add the results to the DataFrame
                    normality_results.append(
                        {
                            "Motif": motif,
                            "Treatment_State": state,
                            "Time_Point": time_point,
                            "W": shapiro_results[0],
                            "p": shapiro_results[1],
                        }
                    )
                else:
                    # Handle cases with insufficient data
                    normality_results.append(
                        {
                            "Motif": motif,
                            "Treatment_State": state,
                            "Time_Point": time_point,
                            "W": None,
                            "p": None,
                        }
                    )

    return pd.DataFrame(normality_results)


def check_normality_by_group(df, meta_data):
    normality_results = []

    for group in meta_data["Group"].unique():
        filtered_meta = meta_data[meta_data["Group"] == group]
        group_data = df.loc[:, filtered_meta["File_Name"]].values.flatten()

        # Fill any NaN values with 0
        group_data = np.nan_to_num(group_data)

        # Check for constant data or NaN values
        if np.all(group_data == group_data[0]) or np.isnan(group_data).any():
            normality_results.append(
                {"Group": group, "W": None, "p": None, "Note": "Constant data or NaN"}
            )
            continue

        # Check if there are enough data points for the test
        if len(group_data) >= 3:
            # Import the stats module from scipy
            from scipy import stats

            shapiro_results = stats.shapiro(group_data)
            normality_results.append(
                {"Group": group, "W": shapiro_results[0], "p": shapiro_results[1]}
            )
        else:
            normality_results.append(
                {"Group": group, "W": None, "p": None, "Note": "Insufficient data"}
            )

    return pd.DataFrame(normality_results)


def check_normality_by_group_and_time(df, meta_data, type_c="Group"):
    normality_results = []

    for group in meta_data[type_c].unique():
        for time_point in meta_data["Time_Point"].unique():
            # Filter meta_data for the current group and time point
            filtered_meta = meta_data[
                (meta_data[type_c] == group) & (meta_data["Time_Point"] == time_point)
            ]

            # Check if the filtered meta_data is not empty
            if not filtered_meta.empty:
                # Get the corresponding names for filtering the df
                names = filtered_meta["File_Name"].tolist()
                # Filter the df for the current group and time point
                group_time_data = df[names].values.flatten()  # * Original line

                console.print(group_time_data)

                # Handle case where all data may be NaN after filtering
                if group_time_data.size == 0 or np.isnan(group_time_data).all():
                    normality_results.append(
                        {
                            type_c: group,
                            "Time_Point": time_point,
                            "W": None,
                            "p": None,
                            "Note": "No data or all data is NaN",
                        }
                    )
                    continue

                # Fill any NaN values with 0
                group_time_data = np.nan_to_num(group_time_data)

                # Check for constant data
                if np.all(group_time_data == group_time_data[0]):
                    normality_results.append(
                        {
                            type_c: group,
                            "Time_Point": time_point,
                            "W": None,
                            "p": None,
                            "Note": "Constant data",
                        }
                    )
                    continue

                # Check if there are enough data points for the test
                if len(group_time_data) >= 3:
                    shapiro_results = scipy.stats.shapiro(group_time_data)
                    normality_results.append(
                        {
                            type_c: group,
                            "Time_Point": time_point,
                            "W": shapiro_results[0],
                            "p": shapiro_results[1],
                        }
                    )
                else:
                    normality_results.append(
                        {
                            type_c: group,
                            "Time_Point": time_point,
                            "W": None,
                            "p": None,
                            "Note": "Insufficient data",
                        }
                    )
            else:
                # Handle case where the filtered meta_data is empty
                normality_results.append(
                    {
                        type_c: group,
                        "Time_Point": time_point,
                        "W": None,
                        "p": None,
                        "Note": "No matching data for group/time point",
                    }
                )

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
    sham_totals = (
        df.filter(pl.col("group") == "Sham")
        .groupby(["time_point", "rat_id"])
        .agg(pl.col(col_to_test).sum().alias("total_distance"))
    )

    # Iterate over each time point
    for time_point in sham_totals.get_column("time_point").unique().to_list():
        # Get the total distance for the Sham group at this time point
        sham_total_distances = sham_totals.filter(pl.col("time_point") == time_point)[
            "total_distance"
        ].to_numpy()

        # Iterate over each group except Sham
        for group in df.get_column("group").unique().to_list():
            if group != "Sham":
                # Calculate the total distance for the current group at this time point
                group_totals = (
                    df.filter(
                        (pl.col("group") == group)
                        & (pl.col("time_point") == time_point)
                    )
                    .groupby(["rat_id"])
                    .agg(pl.col(col_to_test).sum().alias("total_distance"))
                )

                # Get the total distance for the current group at this time point
                group_total_distances = group_totals["total_distance"].to_numpy()

                # Perform the independent t-test between the Sham group and the current group
                p_value, d = perform_independent_t_test(
                    sham_total_distances, group_total_distances
                )
                independent_comparison_results.append((group, time_point, p_value, d))

    return independent_comparison_results


def perform_paired_t_test(before, after):
    # Perform the paired t-test
    t_stat, p_value = scipy.stats.ttest_rel(before, after)
    # Calculate Cohen's d for effect size
    d = (np.mean(after) - np.mean(before)) / np.std(after - before, ddof=1)
    return p_value, d


def within_group_across_total_distance_time_paired_t_test(
    df, baseline_data, col_to_test="distance"
):
    # Perform within-group comparisons
    paired_comparison_results = []
    for group in df.get_column("group").unique().to_list():
        # Get baseline total distances for the group
        baseline_totals = (
            baseline_data.filter(pl.col("group") == group)
            .groupby("rat_id")
            .agg(pl.col(col_to_test).sum().alias("total_baseline_distance"))
        )

        # Perform comparisons for each time point
        for time_point in df.get_column("time_point").unique().to_list():
            if time_point not in ["Baseline_1", "Baseline_2"]:
                # Get total distances for the current time point
                time_point_totals = (
                    df.filter(
                        (pl.col("group") == group)
                        & (pl.col("time_point") == time_point)
                    )
                    .groupby("rat_id")
                    .agg(pl.col(col_to_test).sum().alias("total_time_point_distance"))
                )

                # Ensure we have matching rats before performing paired t-test
                if baseline_totals.shape[0] == time_point_totals.shape[0]:
                    # Perform the paired t-test
                    p_value, d = perform_paired_t_test(
                        baseline_totals["total_baseline_distance"].to_numpy(),
                        time_point_totals["total_time_point_distance"].to_numpy(),
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
        rows_list.append(
            {
                "group": group,
                "time_point": time_point,
                "p_value": p_value,
                "effect_size": d,
            }
        )

    # Create a DataFrame from the list of dictionaries
    df = pl.DataFrame(rows_list)
    return df


# Function to apply Shapiro-Wilk test to each group
def test_normality(group, value_col="Log_Normalized_Value"):
    from scipy.stats import shapiro

    data = group[value_col]
    if data.nunique() > 1:  # Check if data has variance
        stat, p = shapiro(data)
    else:  # If data has no variance, return default values
        stat, p = np.nan, np.nan
    return pd.Series({"W": stat, "p": p})


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
    aovrm = AnovaRM(
        df,
        depvar=dependent_var,
        subject=subject,
        within=[within, between],
        aggregate_func="mean",
    )
    res = aovrm.fit()
    return res


def run_mixed_effects_model(
    df, dependent_var, groups, re_formula, fe_formula="1", maxiter=100
):
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

    model = MixedLM.from_formula(
        f"{dependent_var} ~ {fe_formula}",
        groups=df[groups],
        re_formula=re_formula,
        data=df,
    )
    optimizers = ["lbfgs", "bfgs", "nm", "powell", "cg", "newton"]

    for optimizer in optimizers:
        try:
            result = model.fit(method=optimizer, maxiter=maxiter)
            if result.converged:
                print(f"Converged with {optimizer}")
                return result
        except np.linalg.LinAlgError:
            print(f"Failed with {optimizer}")

    raise ValueError("None of the optimizers converged")


def apply_normality_status_Group(row, normality_dict):
    return normality_dict.get(
        (row["Time_Point"], row["Group"], row["Motif"]), True
    )  # Default to True if not found


# Function to apply the normality status based on the dictionary for Treatment_State
def apply_normality_status_State(row, normality_dict):
    return normality_dict.get(
        (row["Time_Point"], row["Treatment_State"], row["Motif"]), True
    )  # Default to True if not found


def calculate_p_values_vs_sham_Group(df, log_comp=False):
    new_rows = []
    time_points = df["Time_Point"].unique()
    groups = df["Group"].unique()
    motifs = df["Motif"].unique()

    # Iterate over each time point and group
    for time_point in time_points:
        for group in groups:
            if group == "Sham":
                continue
            for motif in motifs:
                # Filter the DataFrame for the current group, time point, and motif
                group_data = df[
                    (df["Time_Point"] == time_point)
                    & (df["Group"] == group)
                    & (df["Motif"] == motif)
                ]
                sham_data = df[
                    (df["Time_Point"] == time_point)
                    & (df["Group"] == "Sham")
                    & (df["Motif"] == motif)
                ]

                # Check if both distributions are normal
                if (
                    group_data["Is_Normal_Group"].all()
                    and sham_data["Is_Normal_Group"].all()
                ):
                    test = "t_stat"
                else:
                    test = "u_stat"

                # Select the appropriate value column based on log_comp
                value_column = (
                    "Log_Normalized_Value" if log_comp else "Normalized_Value"
                )
                group_values = group_data[value_column]
                sham_values = sham_data[value_column]

                if not group_values.empty and not sham_values.empty:
                    if test == "t_stat":
                        t_stat, p_val = ttest_ind(
                            group_values,
                            sham_values,
                            equal_var=False,
                            nan_policy="omit",
                        )
                        new_row = {
                            "Animal_ID": group_data["Animal_ID"].iloc[0],
                            "Time_Point": time_point,
                            "Group": group,
                            "Motif": motif,
                            "P_Value": p_val,
                            "Stat_Value": t_stat,
                            "Test_Type": "t-test",
                        }
                    elif test == "u_stat":
                        u_stat, p_val = mannwhitneyu(
                            group_values, sham_values, alternative="two-sided"
                        )
                        new_row = {
                            "Animal_ID": group_data["Animal_ID"].iloc[0],
                            "Time_Point": time_point,
                            "Group": group,
                            "Motif": motif,
                            "P_Value": p_val,
                            "Stat_Value": u_stat,
                            "Test_Type": "Mann-Whitney",
                        }
                    new_rows.append(new_row)

    # Create a DataFrame from the list of new rows
    p_values_df = pd.DataFrame(new_rows)

    return p_values_df


def calculate_p_values_vs_sham_State(df, log_comp=False):
    new_rows = []
    time_points = df["Time_Point"].unique()
    states = df["Treatment_State"].unique()
    motifs = df["Motif"].unique()

    # Iterate over each time point and state
    for time_point in time_points:
        for state in states:
            if state == "Sham":
                continue
            for motif in motifs:
                # Filter the DataFrame for the current state, time point, and motif
                state_data = df[
                    (df["Time_Point"] == time_point)
                    & (df["Treatment_State"] == state)
                    & (df["Motif"] == motif)
                ]
                sham_data = df[
                    (df["Time_Point"] == time_point)
                    & (df["Treatment_State"] == "Sham")
                    & (df["Motif"] == motif)
                ]

                # Check if both distributions are normal
                if (
                    state_data["Is_Normal_State"].all()
                    and sham_data["Is_Normal_State"].all()
                ):
                    test = "t_stat"
                else:
                    test = "u_stat"

                # Select the appropriate value column based on log_comp
                value_column = (
                    "Log_Normalized_Value" if log_comp else "Normalized_Value"
                )
                state_values = state_data[value_column]
                sham_values = sham_data[value_column]

                if not state_values.empty and not sham_values.empty:
                    if test == "t_stat":
                        t_stat, p_val = ttest_ind(
                            state_values,
                            sham_values,
                            equal_var=False,
                            nan_policy="omit",
                        )
                        new_row = {
                            "Animal_ID": state_data["Animal_ID"].iloc[0],
                            "Time_Point": time_point,
                            "Treatment_State": state,
                            "Motif": motif,
                            "P_Value": p_val,
                            "Stat_Value": t_stat,
                            "Test_Type": "t-test",
                        }
                    elif test == "u_stat":
                        u_stat, p_val = mannwhitneyu(
                            state_values, sham_values, alternative="two-sided"
                        )
                        new_row = {
                            "Animal_ID": state_data["Animal_ID"].iloc[0],
                            "Time_Point": time_point,
                            "Treatment_State": state,
                            "Motif": motif,
                            "P_Value": p_val,
                            "Stat_Value": u_stat,
                            "Test_Type": "Mann-Whitney",
                        }
                    new_rows.append(new_row)

    # Create a DataFrame from the list of new rows
    p_values_df = pd.DataFrame(new_rows)

    return p_values_df


def calculate_p_values_vs_baseline_Group(df, log_comp=False):
    new_rows = []
    time_points = df["Time_Point"].unique()
    groups = df["Group"].unique()
    motifs = df["Motif"].unique()

    # Iterate over each group and motif
    for group in groups:
        for motif in motifs:
            # Get baseline values for the current group and motif
            baseline_group_data = df[
                (df["Time_Point"].isin(["Baseline_1", "Baseline_2"]))
                & (df["Group"] == group)
                & (df["Motif"] == motif)
            ]
            baseline_value_column = (
                "Log_Normalized_Value" if log_comp else "Normalized_Value"
            )
            baseline_values = baseline_group_data[baseline_value_column]

            # Iterate over each time point
            for time_point in time_points:
                if time_point in ["Baseline_1", "Baseline_2"] or (
                    time_point == "Drug_Trt" and group == "ABX"
                ):
                    continue

                # Get values for the current time point, group, and motif
                group_data = df[
                    (df["Time_Point"] == time_point)
                    & (df["Group"] == group)
                    & (df["Motif"] == motif)
                ]
                group_values = group_data[baseline_value_column]

                # Check if both distributions are normal
                if (
                    baseline_group_data["Is_Normal_Group"].all()
                    and group_data["Is_Normal_Group"].all()
                ):
                    test = "t_stat"
                else:
                    test = "u_stat"

                if not group_values.empty and not baseline_values.empty:
                    if test == "t_stat":
                        t_stat, p_val = ttest_ind(
                            group_values,
                            baseline_values,
                            equal_var=False,
                            nan_policy="omit",
                        )
                        new_row = {
                            "Animal_ID": group_data["Animal_ID"].iloc[0],
                            "Time_Point": time_point,
                            "Group": group,
                            "Motif": motif,
                            "P_Value": p_val,
                            "Stat_Value": t_stat,
                            "Test_Type": "t-test",
                        }
                    elif test == "u_stat":
                        u_stat, p_val = mannwhitneyu(
                            group_values, baseline_values, alternative="two-sided"
                        )
                        new_row = {
                            "Animal_ID": group_data["Animal_ID"].iloc[0],
                            "Time_Point": time_point,
                            "Group": group,
                            "Motif": motif,
                            "P_Value": p_val,
                            "Stat_Value": u_stat,
                            "Test_Type": "Mann-Whitney",
                        }
                    new_rows.append(new_row)

    # Create a DataFrame from the list of new rows
    p_values_df = pd.DataFrame(new_rows)

    return p_values_df


def calculate_p_values_vs_baseline_State(df, log_comp=False):
    new_rows = []
    time_points = df["Time_Point"].unique()
    states = df["Treatment_State"].unique()
    motifs = df["Motif"].unique()

    # Iterate over each group and motif
    for state in states:
        for motif in motifs:
            # Get baseline values for the current group and motif

            if state == "Treated":
                baseline_state_data = df[
                    (df["Time_Point"].isin(["Baseline_1", "Baseline_2"]))
                    & (df["Treatment_State"] == "Injured")
                    & (df["Motif"] == motif)
                ]
            else:
                baseline_state_data = df[
                    (df["Time_Point"].isin(["Baseline_1", "Baseline_2"]))
                    & (df["Treatment_State"] == state)
                    & (df["Motif"] == motif)
                ]

            baseline_value_column = (
                "Log_Normalized_Value" if log_comp else "Normalized_Value"
            )
            baseline_values = baseline_state_data[baseline_value_column]

            # Iterate over each time point
            for time_point in time_points:
                if time_point in ["Baseline_1", "Baseline_2"] or (
                    time_point == "Drug_Trt" and state == "ABX"
                ):
                    continue

                # Get values for the current time point, group, and motif
                state_data = df[
                    (df["Time_Point"] == time_point)
                    & (df["Treatment_State"] == state)
                    & (df["Motif"] == motif)
                ]
                state_values = state_data[baseline_value_column]

                # Check if both distributions are normal
                if (
                    baseline_state_data["Is_Normal_State"].all()
                    and state_data["Is_Normal_State"].all()
                ):
                    test = "t_stat"
                else:
                    test = "u_stat"

                if not state_values.empty and not baseline_values.empty:
                    if test == "t_stat":
                        t_stat, p_val = ttest_ind(
                            state_values,
                            baseline_values,
                            equal_var=False,
                            nan_policy="omit",
                        )
                        new_row = {
                            "Animal_ID": state_data["Animal_ID"].iloc[0],
                            "Time_Point": time_point,
                            "Treatment_State": state,
                            "Motif": motif,
                            "P_Value": p_val,
                            "Stat_Value": t_stat,
                            "Test_Type": "t-test",
                        }
                    elif test == "u_stat":
                        u_stat, p_val = mannwhitneyu(
                            state_values, baseline_values, alternative="two-sided"
                        )
                        new_row = {
                            "Animal_ID": state_data["Animal_ID"].iloc[0],
                            "Time_Point": time_point,
                            "Treatment_State": state,
                            "Motif": motif,
                            "P_Value": p_val,
                            "Stat_Value": u_stat,
                            "Test_Type": "Mann-Whitney",
                        }
                    new_rows.append(new_row)

    # Create a DataFrame from the list of new rows
    p_values_df = pd.DataFrame(new_rows)

    return p_values_df
