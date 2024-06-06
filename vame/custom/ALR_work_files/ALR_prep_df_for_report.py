import numpy as np
import pandas as pd
from rich import pretty
from sklearn import base

import vame
import vame.custom.ALR_helperFunctions as AlHf

pretty.install()
from rich.console import Console

console = Console()


def adjust_treatment_state(console, df):
    import re
    df.loc[df["Time_Point"] != "Drug_Trt", "Time_Point_Int"] = (
        df.loc[df["Time_Point"] != "Drug_Trt", "Time_Point"]
        .str.extract("(\d+)")
        .astype(int)
    )
    df.loc[df["Time_Point"] == "Drug_Trt", "Time_Point_Int"] = 16
    
    # For all rows where "Time_Point" isnt "Drug_Treat", extract the integers from the string and set them as the "Time_Point_Int" column.
    for i in range(len(df)):
        if df.loc[i, "Time_Point"]!= "Drug_Trt":
            df.loc[i, "Time_Point_Int"] = int(
                re.findall("\\d+", df.loc[i, "Time_Point"])[0]
            )
        else:
            df.loc[i, "Time_Point_Int"] = 16

    conditions = [
        (df["Group"] == "Sham"),
        (df["Group"] == "Injured"),
        (df["Group"] == "ABX"),
        (
            (df["Group"] == "Treated")
            & (df["Time_Point_Int"] > 8)
            #& (df["Time_Point"] != "Drug_Trt")
        ),
        (
            (df["Group"] == "Treated")
            & (df["Time_Point_Int"] <= 8)
        ),
    ]
    
    choices = ["Sham", "Injured", "ABX", "Treated", "Injured"]
    df["Treatment_State"] = np.select(conditions, choices, default="Injured")

    # Check if the adjustment for Treatment_State values is correct for the first 8 weeks of treatment
    if ((df["Treatment_State"] == "Treated") & (df["Time_Point_Int"] <= 8)).any():
        console.print(
            "The adjustment for Treatment_State values is not correct for the first 8 weeks of treatment"
        )
    
    console.print(
        df["Treatment_State"].value_counts(),
        style="bold green"
    )

    return df


def mean_baseline_values(df):
    baseline_time_points = ["Baseline_1", "Baseline_2"]

    # Calculate the mean value for each Animal_ID and Motif during the baseline time points
    State_baseline_means = (
        df[df["Time_Point"].isin(baseline_time_points)]
        .groupby(["Animal_ID", "Motif"])["Value"]
        .mean()
        .reset_index()
    )
    State_baseline_means.columns = ["Animal_ID", "Motif", "Value"]
    State_baseline_means["Time_Point"] = "Week_00"

    # Remove the baseline time points from the original dataframe
    no_baseline_df = df[~df["Time_Point"].isin(baseline_time_points)]

    # Concatenate the baseline means with the original dataframe
    no_baseline_df = pd.concat(
        [no_baseline_df, State_baseline_means], ignore_index=True
    )

    # Define the columns to be filled for each grouping variable
    group_cols = {
        "Animal_ID": ["Treatment_State", "Group"],
        "Motif": [
            "Moving_Quickly",
            "Predominant_Behavior",
            "Secondary_Descriptor",
            "Category",
            "Exclude",
            "Predominant_Behavior_Manual_Cluster",
        ],
    }

    # Save the rows where "Time_Point" is "Drug_Trt" to a new data frame
    #treatment_df = no_baseline_df[no_baseline_df["Time_Point"] == "Drug_Trt"]
    #console.print(treatment_df)

    #no_baseline_df = no_baseline_df[no_baseline_df["Time_Point"] != "Drug_Trt"]

    # Fill NaN values based on the mode of each group
    for group, cols in group_cols.items():
        for col in cols:
            no_baseline_df[col] = no_baseline_df.groupby(group)[col].transform(
                lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
            )

    no_baseline_df = adjust_treatment_state(console, no_baseline_df)

    # If "Moving_Quickly" is NaN, set it to "False"
    no_baseline_df["Moving_Quickly"] = (
        no_baseline_df["Moving_Quickly"].fillna(False).astype(bool)
    )

    # Exclude rows where "Exclude" is True
    no_baseline_df = no_baseline_df[no_baseline_df["Exclude"] == False]

    # Rename "Predominant_Behavior_Manual_Cluster" to "Cluster"
    no_baseline_df.rename(
        columns={"Predominant_Behavior_Manual_Cluster": "Cluster"}, inplace=True
    )

    return no_baseline_df

# --------------------------------------------------------------------------- #
# Section 1: Data Preparation
# --------------------------------------------------------------------------- #

master_df = pd.read_csv(
    "/Users/adanredwine/Desktop/needed_files/all_sequences_hmm-40-650.csv"
)  # This is made with AlHf.create_andOR_get_master_df

master_df = master_df.query("rat_id != 'A1'")

# 1.1 Making motif usages per video dataframe (this can be alternatively be made with hf.combineBehavior)
temp_df = master_df.copy()

temp_df.columns = temp_df.columns.str.strip()

df_class_temp = temp_df.groupby(["file_name", "motif"]).size().reset_index()

df_class_temp.columns = ["file_name", "motif", "value"]

df_pivot = df_class_temp.pivot(index="motif", columns="file_name", values="value")

df_pivot.fillna(0, inplace=True)

motif_usages_df = df_pivot.copy()

meta_data = AlHf.create_meta_data_df(motif_usages_df)

meta_data.rename(columns={"Name": "File_Name"}, inplace=True)
# ---

df_classifications = pd.read_csv(
    "/Users/adanredwine/Desktop/needed_files/ALR_manual_motif_to_behavior_classification_hmm_650.csv"
)

df = master_df.merge(
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

df.drop_duplicates()

df.columns = [col.replace("_", " ").title().replace(" ", "_") for col in df.columns]

df.rename(
    columns={"Animal_Id": "Animal_ID"},
    inplace=True,
)

# --------------------------------------------------------------------------- #
# Make data easier to work with
# --------------------------------------------------------------------------- #
# Rename Rat_Id to Animal_ID
df.rename(columns={"Rat_Id": "Animal_ID"}, inplace=True)

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
per_video_total_motif_usage = adjust_treatment_state(
    console, per_video_total_motif_usage
)

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

# Get a list of columns that end with "_x" and drop them
merged_df.drop(
    columns=[col for col in merged_df.columns if col.endswith("_x")], inplace=True
)

# Rename the remaining columns to remove the "_y" suffix
merged_df.columns = merged_df.columns.str.replace("_y$", "", regex=True)

# Drop any rows where "Exclude" is True
merged_df = merged_df[merged_df.Exclude == False]

# Assign each Predominant_Behavior a positive integer, starting at 0
merged_df["Predominant_Behavior_Manual_Cluster"] = (
    merged_df["Predominant_Behavior"].astype("category").cat.codes + 1
)

if "A1" in merged_df["Animal_ID"].unique():
    merged_df = merged_df.query("Animal_ID != 'A1'")

# Reset the index
summed_df = merged_df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Mean baseline 1 & 2 values
# --------------------------------------------------------------------------- #
baseline_summed_df = mean_baseline_values(summed_df)


df = baseline_summed_df.copy()

# Add three columns, is_injured, is_treated, is_abx (this may help with mixed effects models)
for r in df.index:
    
    if df.loc[r, "Group"] in ["Injured", "Treated", "ABX"]:
        df.loc[r, "is_injured"] = True
    else:
        df.loc[r, "is_injured"] = False

    if (df.loc[r, "Group"] in ["Injured", "Treated", "ABX"]) & (df.loc[r, "Time_Point_Int"].astype(float) == 0.0):
        df.loc[r, "is_injured_true"] = False
    else:
        df.loc[r, "is_injured_true"] = df.loc[r, "is_injured"]
    
    if df.loc[r, "Treatment_State"] == "ABX":
        df.loc[r, "is_abx"] = True
        df.loc[r, "is_treated"] = False
    else:
        df.loc[r, "is_abx"] = False

    if ((df.loc[r, "Group"] == "Treated") & (df.loc[r, "Time_Point_Int"].astype(float) > 8.0)):
        df.loc[r, "is_abx"] = False
        df.loc[r, "is_treated"] = True
    else:
        df.loc[r, "is_treated"] = False

if df.columns.str.contains("Unnamed").any():
    df.drop(columns=["Unnamed: 0"], inplace=True)

df['Desc_Behavior'] = df.apply(lambda row: row['Predominant_Behavior'] if pd.isnull(row['Secondary_Descriptor']) else row['Predominant_Behavior'] + "-" + row['Secondary_Descriptor'], axis=1)



# Define the mapping from Animal_ID to Recording_Block
mapping_recBlock = {
    "Block_1": ("A2", "B1", "B2"),
    "Block_2": ("C1", "C2", "D1", "D2"),
    "Block_3": ("E1", "E2", "F1", "F2"),
    "Block_4": ("G1", "G2", "H1", "H2"),
    "Block_5": ("I1", "I2", "J1", "J2"),
    "Block_6": ("K1", "K2", "L1", "L2"),
    "Block_7": ("M1", "M2", "N1", "N2"),
    "Block_8": ("O1", "O2", "P1", "P2"),
    "Block_9": ("Q1", "Q2", "R1", "R2"),
    "Block_10": ("S1", "S2", "T1", "T2"),
    "Block_11": ("U1", "U2", "W1", "W2"),
    "Block_12": ("X1", "X2", "Y1", "Y2"),
}

df["Recording_Block"] = "NaN"
for block, ids in mapping_recBlock.items():
    df.loc[df["Animal_ID"].isin(ids), "Recording_Block"] = block


mapping_sugDate = {
    "22-01-18": {"A1", 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2', 'E1', 'E2', 'F1', 'F2'},
    "22-01-19": {"G1", 'G2', 'H1', 'H2', 'I1', 'I2', 'J1', 'J2', 'K1', 'K2', 'L1', 'L2'},
    "22-01-20": {"M1", 'M2', 'N1', 'N2', 'O1', 'O2', 'P1', 'P2', 'Q1', 'Q2', 'R1', 'R2'},
    "22-01-21": {"S1", 'S2', 'T1', 'T2', 'U1', 'U2', 'W1', 'W2', 'X1', 'X2', 'Y1', 'Y2'}
}

df["Sug_Date"] = "NaN"
for block in mapping_sugDate.keys():
    df.loc[df["Animal_ID"].isin(mapping_sugDate[block]), "Sug_Date"] = block

mapping_postOp = {
    '0.5': {'C2', 'H1', 'I2', 'L1', 'O2', 'P1', 'Q2', 'S1', 'T1', 'W1', 'R2', 'U1', 'U2', 'W1', 'Y1'},
    '1.0': {'M1', 'Q1', 'T2', 'S2', 'W2', 'X2'},
    '2.0': {'F2', 'K2', 'M2', 'R1', 'Y2'},
    "3.0": {'B1'},
}

df["PostSug_Prob"] = "0"
for block in mapping_postOp.keys():
    df.loc[df["Animal_ID"].isin(mapping_postOp[block]), "PostSug_Prob"] = block

# Adding recording_date column, by getting the first 8 characters of the "File_Name" column
df["Recording_Date"] = df["File_Name"].str[:8]

# if any rows of df["Time_Point"] are "Week_00", and the corresponding rows of df["Recording_Date"] "NaN", set the "Recording_Date" "22-01-11"
df.loc[(df["Time_Point"] == "Week_00") & (df["Recording_Date"].isna()), "Recording_Date"] = "22-01-11"

# Convert the "Recording_Date" and "Sug_Date" columns to datetime
df["Recording_Date"] = pd.to_datetime(df["Recording_Date"], format='%y-%m-%d')
df["Sug_Date"] = pd.to_datetime(df["Sug_Date"], format='%y-%m-%d')

# Calculate the difference in days between the "Recording_Date" and "Sug_Date" columns
df["Days_Post_Surgery"] = (df["Recording_Date"] - df["Sug_Date"]).dt.days

# Group the data frame by "Time_Point" and put a 1 in the "Recording_Wk_Day" column for lowest "Recording_Date" value, and a 2 for the next lowest, and so on
df["Recording_Wk_Day"] = df.groupby("Time_Point")["Recording_Date"].rank("dense", ascending=True)

# Save the dataframe,
df.to_csv(
    "/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/baseline_summed_df_hmm.csv"
)

