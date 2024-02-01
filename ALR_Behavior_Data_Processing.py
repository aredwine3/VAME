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
from vame.custom.ALR_latent_vector_cluster_functions import (
    calculate_mean_latent_vector_for_motifs,
    create_tsne_projection,
    create_umap_projection,
)
from vame.util.auxiliary import read_config

pretty.install()
from rich.console import Console

console = Console()

config = "D:\\Users\\tywin\\VAME\\config.yaml"

fps = 30

df_hmm_650: Union[pl.DataFrame, pd.DataFrame] = AlHf.create_andOR_get_master_df(
    config, fps=fps, create_new_df=False, df_kind="pandas"
)

df_classifications = pd.read_csv(
    "D:\\Users\\tywin\\VAME\\results\\videos\\hmm-40-650\\ALR_manual_motif_to_behavior_classification_hmm_650.csv"
)

df_hmm_650 = df_hmm_650.merge(
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

df_hmm_650.columns = df_hmm_650.columns.str.replace(" ", "_")

df_hmm_650.rename(columns={"motif": "motif_hmm_650"}, inplace=True)

profile = ProfileReport(df_hmm_650, title="Profiling Report")
# profile.to_file("your_report.html")

config = "D:\\Users\\tywin\\VAME\\config-kmeans-30.yaml"

# Done after manual classification of behaviors
# Add classifications to the master data frame (made during the motif videos creation)

fps = 30

df_kmeans_30: Union[pl.DataFrame, pd.DataFrame] = AlHf.create_andOR_get_master_df(
    config, fps=fps, create_new_df=False, df_kind="pandas"
)

# Rename the motif column to motif_kmeans_30
df_kmeans_30.rename(columns={"motif": "motif_kmeans_30"}, inplace=True)


# Merge motif_kmeans_30 into df_hmm_650 based on the file name, frame, and rat_id
df_hmm_650 = df_hmm_650.merge(
    df_kmeans_30[["motif_kmeans_30", "file_name", "frame", "rat_id"]],
    left_on=["file_name", "frame", "rat_id"],
    right_on=["file_name", "frame", "rat_id"],
    how="left",
).drop_duplicates()

df = df_hmm_650.copy()

# Drop the motif column
df.drop(columns=["Motif"], inplace=True)

# Create a dictionary of motif_kmeans_30 values and the corresponding motif_hmm_650 values
motif_kmeans_30_to_motif_hmm_650 = (
    df.groupby("motif_kmeans_30")["motif_hmm_650"].unique().to_dict()
)

# Preview that dictionary as a json
import json


def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


print(json.dumps(motif_kmeans_30_to_motif_hmm_650, default=ndarray_to_list, indent=4))
config = "D:\\Users\\tywin\\VAME\\config.yaml"

file_names = AlHf.get_files(config)


cfg = read_config(config)

# Add the full path to the file names

from vame.custom import helperFunctions as hf

motif_usage_hmm_650 = hf.combineBehavior(
    config, save=True, cluster_method="hmm", legacy=False
)
meta_data = AlHf.create_meta_data_df(motif_usage_hmm_650)

# Rename the 'Name' column to 'File_Name' in the meta_data data frame
meta_data.rename(columns={"Name": "File_Name"}, inplace=True)

normality_results_Group = AlSt.check_normality_by_group(motif_usage_hmm_650, meta_data)
normality_results_GroupTime = AlSt.check_normality_by_group_and_time(
    motif_usage_hmm_650, meta_data
)
normality_results_GroupTimeMotif = AlSt.check_normality_group_timepoint_motif(
    motif_usage_hmm_650, meta_data
)
normality_results_StateTimeMotif = AlSt.check_normality_state_timepoint_motif(
    motif_usage_hmm_650, meta_data
)


total_motifs_Group = len(normality_results_GroupTimeMotif)
normal_motifs_Group = normality_results_GroupTimeMotif[
    normality_results_GroupTimeMotif["p"] > 0.05
]
non_normal_motifs_Group = normality_results_GroupTimeMotif[
    normality_results_GroupTimeMotif["p"] <= 0.05
]
percent_normal_Group = (len(normal_motifs_Group) / total_motifs_Group) * 100

total_motifs_State = len(normality_results_StateTimeMotif)
normal_motifs_State = normality_results_StateTimeMotif[
    normality_results_StateTimeMotif["p"] > 0.05
]
non_normal_motifs_State = normality_results_StateTimeMotif[
    normality_results_StateTimeMotif["p"] <= 0.05
]
percent_normal_State = (len(normal_motifs_State) / total_motifs_State) * 100


df.rename(
    columns={
        "file_name": "File_Name",
        "rat_id": "Animal_ID",
        "group": "Group",
        "time_point": "Time_Point",
    },
    inplace=True,
)

# Add the Treatment_State column from meta_data to df by merging on Animal_ID
df = df.merge(
    meta_data[["Animal_ID", "Treatment_State"]],
    left_on="Animal_ID",
    right_on="Animal_ID",
    how="left",
).drop_duplicates()


# Determine the total usage of each motif for each animal at each time point
temp_series = df.groupby(["Animal_ID", "Time_Point", "motif_hmm_650"]).size()

# Convert the series to a data frame, giving the column a name
per_video_total_motif_usage = (temp_series.to_frame("value")).reset_index()

# Add Group, Treatment_State, Moving_Quickly, Predominant_Behavior, Secondary_Descriptor, and Category to the data frame by merging on Animal_ID, Time_Point, and motif_hmm_650
per_video_total_motif_usage = per_video_total_motif_usage.merge(
    df[
        [
            "File_Name",
            "Animal_ID",
            "Time_Point",
            "motif_hmm_650",
            "Group",
            "Treatment_State",
            "Moving_Quickly",
            "Predominant_Behavior",
            "Secondary_Descriptor",
            "Category",
            "Exclude",
        ]
    ],
    left_on=["Animal_ID", "Time_Point", "motif_hmm_650"],
    right_on=["Animal_ID", "Time_Point", "motif_hmm_650"],
    how="left",
).drop_duplicates()


# Assign each Predominant_Behavior a positive integer, starting at 0
per_video_total_motif_usage["Predominant_Behavior_Manual_Cluster"] = (
    per_video_total_motif_usage["Predominant_Behavior"].astype("category").cat.codes + 1
)

# Ensure that each predominant behavior was always assigned the same integer
assert (
    per_video_total_motif_usage.groupby("Predominant_Behavior")[
        "Predominant_Behavior_Manual_Cluster"
    ]
    .nunique()
    .max()
    == 1
)

# Reset the index
per_video_total_motif_usage.reset_index(inplace=True)
# Rename motif_hmm_650 to Motif
per_video_total_motif_usage.rename(columns={"motif_hmm_650": "Motif"}, inplace=True)

# Drop the 'Treatment_State' column
per_video_total_motif_usage.drop(columns=["Treatment_State", "index"], inplace=True)


per_video_total_motif_usage = per_video_total_motif_usage.drop_duplicates()
per_video_total_motif_usage.reset_index(inplace=True)


# Add a 'Treatment_State' column.
# If the animal is in the Sham group, Treatment_State is 'Sham'
# If the animal is in the Injured group, Treatment_State is 'Injured'
# If the animal is in the Treated group, and the integer in the 'Time_Point' column is less than or equal to 8, Treatment_State is 'Injured'
# If the animal is in the Treated group, and the integer in the 'Time_Point' column is greater than 8, Treatment_State is 'Treated'

# Extract the integer from the 'Time_Point' column and convert to numeric
per_video_total_motif_usage["Time_Point_Int"] = pd.to_numeric(
    per_video_total_motif_usage["Time_Point"].str.extract("(\d+)")[0], errors="coerce"
)


# Apply the conditions to set the 'Treatment_State' column
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
                & (per_video_total_motif_usage["Time_Point_Int"] <= 8),
                "Injured",
                "Treated",
            ),
        ),
    ),
)
# Drop the 'Time_Point_Int' column
per_video_total_motif_usage.drop(columns=["Time_Point_Int"], inplace=True)

# Drop the level_0 and index columns
per_video_total_motif_usage.drop(columns=["index"], inplace=True)


# * Finding where the baseline values are NaN'
# Return rows in per_video_total_motif_usage where the the Time_Points "Baseline_1" or "Baseline_2" have NaN in the value column
zero_value_baseline = per_video_total_motif_usage[
    (per_video_total_motif_usage["Time_Point"].isin(["Baseline_1", "Baseline_2"]))
    & (per_video_total_motif_usage["value"] == 0)
]

print(zero_value_baseline)

if len(zero_value_baseline) > 0:
    print("There are zero values in the baseline time points")
else:
    df_long_norm_base = AlSt.normalize_to_baseline(per_video_total_motif_usage)
    df_long_logNorm_base = AlSt.normalize_to_baseline_log(per_video_total_motif_usage)
    df_long_norm_sham = AlSt.normalize_to_baseline_sham(per_video_total_motif_usage)
    df_long_logNorm_sham = AlSt.normalize_to_baseline_sham_log(
        per_video_total_motif_usage
    )

result = AlSt.run_mixed_effects_model(
    df_long_norm_base,
    dependent_var="Normalized_Value",
    groups="Animal_ID",
    re_formula="1",
    fe_formula="Time_Point * Group",
    maxiter=100,
)
print(result.summary())

result_logNorm_base = AlSt.run_mixed_effects_model(
    df_long_logNorm_base,
    dependent_var="Log_Normalized_Value",
    groups="Animal_ID",
    re_formula="1",
    fe_formula="Time_Point * Group",
    maxiter=100,
)
print(result_logNorm_base.summary())

result = AlSt.run_kruskal_wallis_test(df_long, groups="Treatment_State", values="value")
print(result)

# Group the data by 'Time_Point', 'Group', and 'Motif' and apply the test
normality_results_vsSham_Group = (
    df_long_logNorm_sham.groupby(["Time_Point", "Group", "Motif"])
    .apply(AlHf.test_normality, value_col="Log_Normalized_Value")
    .reset_index()
)
normality_results_vsBase_Group = (
    df_long_logNorm_base.groupby(["Time_Point", "Group", "Motif"])
    .apply(AlHf.test_normality, value_col="Log_Normalized_Value")
    .reset_index()
)
normality_results_vsSham_State = (
    df_long_logNorm_sham.groupby(["Time_Point", "Treatment_State", "Motif"])
    .apply(AlHf.test_normality, value_col="Log_Normalized_Value")
    .reset_index()
)
normality_results_vsBase_State = (
    df_long_logNorm_base.groupby(["Time_Point", "Treatment_State", "Motif"])
    .apply(AlHf.test_normality, value_col="Log_Normalized_Value")
    .reset_index()
)

normal_motifs_vsSham_Group = normality_results_vsSham_Group[
    normality_results_vsSham_Group["p"] > 0.05
]
non_normal_motifs_vsSham_Group = normality_results_vsSham_Group[
    normality_results_vsSham_Group["p"] <= 0.05
]
percent_normal_vsSham_Group = (
    len(normal_motifs_vsSham_Group) / len(normality_results_vsSham_Group) * 100
)

normal_motifs_vsBase_Group = normality_results_vsBase_Group[
    normality_results_vsBase_Group["p"] > 0.05
]
non_normal_motifs_vsBase_Group = normality_results_vsBase_Group[
    normality_results_vsBase_Group["p"] <= 0.05
]
percent_normal_vsBase_Group = (
    len(normal_motifs_vsBase_Group) / len(normality_results_vsBase_Group) * 100
)

normal_motifs_vsSham_State = normality_results_vsSham_State[
    normality_results_vsSham_State["p"] > 0.05
]
non_normal_motifs_vsSham_State = normality_results_vsSham_State[
    normality_results_vsSham_State["p"] <= 0.05
]
percent_normal_vsSham_State = (
    len(normal_motifs_vsSham_State) / len(normality_results_vsSham_State) * 100
)

normal_motifs_vsBase_State = normality_results_vsBase_State[
    normality_results_vsBase_State["p"] > 0.05
]
non_normal_motifs_vsBase_State = normality_results_vsBase_State[
    normality_results_vsBase_State["p"] <= 0.05
]
percent_normal_vsBase_State = (
    len(normal_motifs_vsBase_State) / len(normality_results_vsBase_State) * 100
)

print(percent_normal_vsSham_Group)
print(percent_normal_vsBase_Group)

print(percent_normal_vsSham_State)
print(percent_normal_vsBase_State)


normality_vsSham_dict_Group = {
    (row["Time_Point"], row["Group"], row["Motif"]): row["p"] > 0.05
    for index, row in normality_results_vsSham_Group.iterrows()
}
normality_vsBase_dict_Group = {
    (row["Time_Point"], row["Group"], row["Motif"]): row["p"] > 0.05
    for index, row in normality_results_vsBase_Group.iterrows()
}

normality_vsSham_dict_State = {
    (row["Time_Point"], row["Treatment_State"], row["Motif"]): row["p"] > 0.05
    for index, row in normality_results_vsSham_State.iterrows()
}
normality_vsBase_dict_State = {
    (row["Time_Point"], row["Treatment_State"], row["Motif"]): row["p"] > 0.05
    for index, row in normality_results_vsBase_State.iterrows()
}


# Apply the function to create a new column 'Is_Normal_Group'
df_long_logNorm_sham["Is_Normal_Group"] = df_long_logNorm_sham.apply(
    AlSt.apply_normality_status_Group,
    axis=1,
    normality_dict=normality_vsSham_dict_Group,
)
df_long_logNorm_base["Is_Normal_Group"] = df_long_logNorm_base.apply(
    AlSt.apply_normality_status_Group,
    axis=1,
    normality_dict=normality_vsBase_dict_Group,
)


# Apply the function to create a new column 'Is_Normal_State'
df_long_logNorm_sham["Is_Normal_State"] = df_long_logNorm_sham.apply(
    AlSt.apply_normality_status_State,
    axis=1,
    normality_dict=normality_vsSham_dict_State,
)
df_long_logNorm_base["Is_Normal_State"] = df_long_logNorm_base.apply(
    AlSt.apply_normality_status_State,
    axis=1,
    normality_dict=normality_vsBase_dict_State,
)

pVals_longNorm_sham_Group = AlSt.calculate_p_values_vs_sham_Group(
    df_long_logNorm_sham, log_comp=True
)
pVals_longNorm_base_Group = AlSt.calculate_p_values_vs_baseline_Group(
    df_long_logNorm_base, log_comp=True
)
pVals_longNorm_sham_State = AlSt.calculate_p_values_vs_sham_State(
    df_long_logNorm_sham, log_comp=True
)
pVals_longNorm_base_State = AlSt.calculate_p_values_vs_baseline_State(
    df_long_logNorm_base, log_comp=True
)

from statsmodels.stats.multitest import multipletests

# Adjusting p-values for the multiple tests:
p_vals = pVals_longNorm_sham_Group["P_Value"].values
adjusted_p_vals = multipletests(p_vals, alpha=0.05, method="bonferroni")
pVals_longNorm_sham_Group["Adjusted_P_Value"] = adjusted_p_vals[1]

p_vals = pVals_longNorm_base_Group["P_Value"].values
adjusted_p_vals = multipletests(p_vals, alpha=0.05, method="bonferroni")
pVals_longNorm_base_Group["Adjusted_P_Value"] = adjusted_p_vals[1]

p_vals = pVals_longNorm_sham_State["P_Value"].values
adjusted_p_vals = multipletests(p_vals, alpha=0.05, method="bonferroni")
pVals_longNorm_sham_State["Adjusted_P_Value"] = adjusted_p_vals[1]

p_vals = pVals_longNorm_base_State["P_Value"].values
adjusted_p_vals = multipletests(p_vals, alpha=0.05, method="bonferroni")
pVals_longNorm_base_State["Adjusted_P_Value"] = adjusted_p_vals[1]

# Merge the group p-values
df_long_logNorm_sham = df_long_logNorm_sham.merge(
    pVals_longNorm_sham_Group[
        [
            "Time_Point",
            "Group",
            "Motif",
            "P_Value",
            "Adjusted_P_Value",
            "Stat_Value",
            "Test_Type",
        ]
    ],
    on=["Time_Point", "Group", "Motif"],
    how="left",
    suffixes=("", "_group"),
)

# Merge the state p-values
df_long_logNorm_sham = df_long_logNorm_sham.merge(
    pVals_longNorm_sham_State[
        [
            "Time_Point",
            "Treatment_State",
            "Motif",
            "P_Value",
            "Adjusted_P_Value",
            "Stat_Value",
            "Test_Type",
        ]
    ],
    on=["Time_Point", "Treatment_State", "Motif"],
    how="left",
    suffixes=("", "_state"),
)

# Rename the columns to make them differentiable
df_long_logNorm_sham.rename(
    columns={
        "P_Value": "P_Value_Group",
        "Stat_Value": "Stat_Value_Group",
        "Test_Type": "Test_Type_Group",
        "P_Value_state": "P_Value_State",
        "Stat_Value_state": "Stat_Value_State",
        "Test_Type_state": "Test_Type_State",
        "Adjusted_P_Value_group": "Adjusted_P_Value_Group",
        "Adjusted_P_Value_state": "Adjusted_P_Value_State",
    },
    inplace=True,
)


# Merge the group p-values
df_long_logNorm_base = df_long_logNorm_base.merge(
    pVals_longNorm_base_Group[
        [
            "Time_Point",
            "Group",
            "Motif",
            "P_Value",
            "Adjusted_P_Value",
            "Stat_Value",
            "Test_Type",
        ]
    ],
    on=["Time_Point", "Group", "Motif"],
    how="left",
    suffixes=("", "_group"),
)

# Merge the state p-values
df_long_logNorm_base = df_long_logNorm_base.merge(
    pVals_longNorm_base_State[
        [
            "Time_Point",
            "Treatment_State",
            "Motif",
            "P_Value",
            "Adjusted_P_Value",
            "Stat_Value",
            "Test_Type",
        ]
    ],
    on=["Time_Point", "Treatment_State", "Motif"],
    how="left",
    suffixes=("", "_state"),
)

# Rename the columns to make them differentiable
df_long_logNorm_base.rename(
    columns={
        "P_Value": "P_Value_Group",
        "Stat_Value": "Stat_Value_Group",
        "Test_Type": "Test_Type_Group",
        "P_Value_state": "P_Value_State",
        "Stat_Value_state": "Stat_Value_State",
        "Test_Type_state": "Test_Type_State",
        "Adjusted_P_Value_group": "Adjusted_P_Value_Group",
        "Adjusted_P_Value_state": "Adjusted_P_Value_State",
    },
    inplace=True,
)

# Count the number of significant motifs for each Group at each Time_Point when comparing to sham
# and list the significant motifs
significant_counts_vsSham_Group = (
    pVals_longNorm_sham_Group[pVals_longNorm_sham_Group["P_Value"] <= 0.05]
    .groupby(["Time_Point", "Group"])["Motif"]
    .apply(list)
    .reset_index(name="Significant_Motifs_Group")
)
significant_counts_vsSham_Group[
    "Significant_Motif_Count_Group"
] = significant_counts_vsSham_Group["Significant_Motifs_Group"].apply(len)

significant_counts_vsBase_Group = (
    pVals_longNorm_base_Group[pVals_longNorm_base_Group["P_Value"] <= 0.05]
    .groupby(["Time_Point", "Group"])["Motif"]
    .apply(list)
    .reset_index(name="Significant_Motifs_Group")
)
significant_counts_vsBase_Group[
    "Significant_Motif_Count_Group"
] = significant_counts_vsBase_Group["Significant_Motifs_Group"].apply(len)

significant_counts_vsSham_State = (
    pVals_longNorm_sham_State[pVals_longNorm_sham_State["P_Value"] <= 0.05]
    .groupby(["Time_Point", "Treatment_State"])["Motif"]
    .apply(list)
    .reset_index(name="Significant_Motifs_State")
)
significant_counts_vsSham_State[
    "Significant_Motif_Count_State"
] = significant_counts_vsSham_State["Significant_Motifs_State"].apply(len)

significant_counts_vsBase_State = (
    pVals_longNorm_base_State[pVals_longNorm_base_State["P_Value"] <= 0.05]
    .groupby(["Time_Point", "Treatment_State"])["Motif"]
    .apply(list)
    .reset_index(name="Significant_Motifs_State")
)
significant_counts_vsBase_State[
    "Significant_Motif_Count_State"
] = significant_counts_vsBase_State["Significant_Motifs_State"].apply(len)

import vame.custom.ALR_plottingFunctions as AlPf

# Rename Predominant_Behavior_Manual_Cluster to Cluster
df_long_logNorm_base.rename(
    columns={"Predominant_Behavior_Manual_Cluster": "Cluster"}, inplace=True
)

df_long_logNorm_sham.rename(
    columns={"Predominant_Behavior_Manual_Cluster": "Cluster"}, inplace=True
)


AlPf.plot_normalized_values_by_group_and_timepoint_3(
    df_long_logNorm_base,
    pVals_longNorm_base_Group,
    significance_level=0.05,
    specific_time_point=None,
    log_norm=True,
    c_type="Group",
    adj_Pvals=True,
)

df_long_logNorm_base.to_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/df_long_logNorm_base.csv"
)
pVals_longNorm_base_Group.to_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/pVals_longNorm_base_Group.csv"
)
pVals_longNorm_base_State.to_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/pVals_longNorm_base_State.csv"
)

df_long_logNorm_sham.to_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/df_long_logNorm_sham.csv"
)

pVals_longNorm_sham_Group.to_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/pVals_longNorm_sham_Group.csv"
)
pVals_longNorm_sham_State.to_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/pVals_longNorm_sham_State.csv"
)

df_long_logNorm_base = pd.read_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/df_long_logNorm_base.csv"
)
pVals_longNorm_base_Group = pd.read_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/pVals_longNorm_base_Group.csv"
)
pVals_longNorm_base_State = pd.read_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/pVals_longNorm_base_State.csv"
)

df_long_logNorm_sham = pd.read_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/df_long_logNorm_sham.csv"
)
pVals_longNorm_sham_Group = pd.read_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/pVals_longNorm_sham_Group.csv"
)
pVals_longNorm_sham_State = pd.read_csv(
    "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/pVals_longNorm_sham_State.csv"
)
