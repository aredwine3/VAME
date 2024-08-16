import os
import sys
from importlib import reload
from sqlite3 import Time
from typing import Union

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from cv2 import sumElems
from matplotlib.pyplot import cla
from rich import pretty
from scipy.stats import kurtosis, shapiro, skew
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

# Load the configuration file
config = "/Users/adanredwine/VAME/config_files/config.yaml"

# Create data frame (set create_new_df to True to create a new data frame or if you have not created one yet)
df = AlHf.create_andOR_get_master_df(config, fps=30, create_new_df=False, df_kind="polars")

# Drop all rows where Animal is A1
df = df.filter(df['rat_id'] != 'A1')

# Load in data frame with manual classifications
classifications = pl.read_csv("/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/baseline_summed_df_hmm.csv")

# Keep columns of interest
classifications = classifications[['Motif', 'Moving_Quickly', 'Predominant_Behavior', 'Secondary_Descriptor', 'Category', 'Exclude', 'Cluster', 'Desc_Behavior']]

# Only keep distinct rows
classifications = classifications.unique()

# Make the Motif column lowercase, this is so we can join df and classifications on this column
classifications = classifications.rename({"Motif": "motif"})

# Join the two data frames on the motif column, this will add the manual classifications to the data frame
df = df.join(classifications, on='motif', how='left')

# Capitalist all first letters of the column names
df.columns = [col.replace("_", " ").title().replace(" ", "_") for col in df.columns]

# Rename the 'Rat_Id' column to 'Animal_ID'
df = df.rename({"Rat_Id": "Animal_ID"})


# Save the data frame to a csv file
df.write_csv("/Users/adanredwine/Desktop/ALR_master_df.csv")


#-----------------------------------------------------------------#
#                     Motif Usage Data Frame                      #
#-----------------------------------------------------------------#
""" 
This data frame that gives the frequency of motif usage for each animal at each time point.
"""

motif_usage_df = df.groupby(['Animal_ID', 'Time_Point', 'Motif']).agg(pl.col('Motif').count().alias('Frequency')).sort('Time_Point')

# Add the manual classifications to the motif_usage_df
classifications = classifications.rename({"motif": "Motif"})
motif_usage_df = motif_usage_df.join(classifications, on='Motif', how='left')

# Extract baseline time points
baseline_motif_usage_df = motif_usage_df.filter((motif_usage_df['Time_Point'] == 'Baseline_1') | (motif_usage_df['Time_Point'] == 'Baseline_2'))

# Find the Mean frequency of each motif for each animal during the Baseline period
baseline_motif_usage_df = baseline_motif_usage_df.groupby(['Animal_ID', 'Motif']).agg(pl.col('Frequency').mean().alias('Frequency')).sort('Animal_ID')

# Make sure all values in the 'Frequency' column are integers
baseline_motif_usage_df = baseline_motif_usage_df.with_columns(baseline_motif_usage_df['Frequency'].cast(pl.Int64).alias('Frequency'))

# Add a column called 'Time_Point' with the value '0'
baseline_motif_usage_df = baseline_motif_usage_df.with_columns(pl.lit('0').alias('Time_Point'))

# Merge the baseline_motif_usage_df with the classifications_df
baseline_motif_usage_df = baseline_motif_usage_df.join(classifications, on='Motif', how='left')

# Remove rows with Time_Point equal to Baseline_1 or Baseline_2 in the motif_usage_df
motif_usage_df = motif_usage_df.filter((motif_usage_df['Time_Point'] != 'Baseline_1') & (motif_usage_df['Time_Point'] != 'Baseline_2'))

# Get the order of the columns in the motif_usage_df
columns = motif_usage_df.columns

# Set the order of the columns in the baseline_motif_usage_df to match the motif_usage_df
baseline_motif_usage_df = baseline_motif_usage_df.select(columns)

# Get the column types of each df
column_types = motif_usage_df.dtypes()

# Merge the baseline_motif_usage_df with the motif_usage_df
motif_usage_df = motif_usage_df.extend(baseline_motif_usage_df)

# Save the data frame to a csv file
motif_usage_df.write_csv("/Users/adanredwine/Desktop/Open_Arena/ALR_summed_motif_usage.csv")

#-----------------------------------------------------------------#
#                       Kinematics Analysis                       #
#-----------------------------------------------------------------#

##################### TOTAL DISTANCE TRAVELED #####################
## By Week and Group

distance_df = df.groupby(['Animal_ID', 'Time_Point', 'Group']).agg(pl.col('Distance').sum().alias('Total_Distance')).sort('Time_Point')

""" Creating a single Baseline Time Point """
# Convert Baseline 1 and Baseline 2 to a single Baseline Time_Point
baseline_dist_df = distance_df.filter((distance_df['Time_Point'] == 'Baseline_1') | (distance_df['Time_Point'] == 'Baseline_2'))

# Find the Mean distance traveled for each animal during the Baseline period
baseline_dist_df = baseline_dist_df.groupby(['Animal_ID', 'Group']).agg(pl.col('Total_Distance').mean().alias('Total_Distance')).sort('Animal_ID')

# Add a column called 'Time_Point' with the value 'Baseline'
baseline_dist_df = baseline_dist_df.with_columns(pl.lit('0').alias('Time_Point'))

# Set the columns to the same order as the distance_df
baseline_dist_df = baseline_dist_df.select('Animal_ID', 'Time_Point', 'Group', 'Total_Distance')

# Drop rows with Time_Point equal to Baseline_1 or Baseline_2 in the distance_df
distance_df = distance_df.filter((distance_df['Time_Point'] != 'Baseline_1') & (distance_df['Time_Point'] != 'Baseline_2'))

# Merge the baseline_dist_df with the distance_df
distance_df = distance_df.extend(baseline_dist_df)

# Rename the 'Drug_Treat' Time_Point to '16'
distance_df = distance_df.with_columns(distance_df['Time_Point'].map_elements(lambda x: "16" if x == "Drug_Trt" else x).alias('Time_Point'))

# Only keep integer values in the 'Time_Point' column
def to_int(value):
    try:
        if "Week_" in value:
            return int(value.split('_')[1])
        else:
            return int(value)
    except (ValueError, IndexError):
        return None

distance_df_temp = distance_df.with_columns(distance_df['Time_Point'].map_elements(to_int).alias('Time_Point'))

# Convert the data frame to a pandas data frame (easier to work with IMO)
distance_df = distance_df_temp.to_pandas()

# Save the data frame to a csv file
distance_df.to_csv("/Users/adanredwine/Desktop/Open_Arena/Distance_Traveled/Total/ALR_total_distance_traveled.csv")

########### LATENCY TO ENTER CENTER ZONE ############
# Filter rows where 'In_Center' is True
df_center = df.filter(df['In_Center'] == True)

# Group by 'File_Name' and get the minimum 'Frame' for each group
first_center_entry = df_center.groupby('File_Name', 'Animal_ID', 'Group', 'Time_Point').agg(
    first_center_frame=pl.col('Frame').min()
)

# Print the result
print(first_center_entry) #? How do we want to handle when the data for an animal starts in the center zone?

# Count the number of times 'first_center_frame' is 0
print(first_center_entry.filter(first_center_entry['first_center_frame'] == 0).count())

# Count the percentage of frames where 'first_center_frame' is greater than 30
print(first_center_entry.filter(first_center_entry['first_center_frame'] > 900).count() / first_center_entry.count())


""" Total Distance Statistics """
# Run a 2-way ANOVA by Time_Point and Group
# Fit the model
model = ols('Total_Distance ~ C(Time_Point) * C(Group)', data=distance_df).fit()

# Extract residuals
residuals = model.resid

# Perform the Shapiro-Wilk test for normality
stat, p = shapiro(residuals)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# Calculate skewness and kurtosis
skewness = skew(residuals)
kurt = kurtosis(residuals)

print('Skewness=%.3f, Kurtosis=%.3f' % (skewness, kurt))


# Plot histogram
plt.hist(residuals, bins=30)
plt.title('Histogram of Residuals')
plt.show()

# Plot Q-Q plot
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

# Although the residuals are not normally distributed, we will continue with the ANOVA as the QQ plot and
# histogram show that the residuals are approximately normally distributed

# Perform the two-way ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

# Perform Tukey HSD test for 'Time_Point'
posthoc = pairwise_tukeyhsd(endog=distance_df['Total_Distance'], groups=distance_df['Time_Point'], alpha=0.05)
print(posthoc)


for group in distance_df['Group'].unique():
    distance_df_group = distance_df[distance_df['Group'] == group]
    # Perform Tukey HSD test for 'Time_Point'
    posthoc = pairwise_tukeyhsd(endog=distance_df_group['Total_Distance'], groups=distance_df_group['Time_Point'], alpha=0.05) 
    # Save the post hoc results to a text file
    with open(f"/Users/adanredwine/Desktop/Open_Arena/Distance_Traveled/Total/ALR_total_distance_{group}_post_hoc.txt", 'w') as f:
        f.write(str(posthoc))


for point in distance_df['Time_Point'].unique():
    distance_df_point = distance_df[distance_df['Time_Point'] == point]
    # Perform Tukey HSD test for 'Group'
    posthoc = pairwise_tukeyhsd(endog=distance_df_point['Total_Distance'], groups=distance_df_point['Group'], alpha=0.05) 
    # Save the post hoc results to a text file
    with open(f"/Users/adanredwine/Desktop/Open_Arena/Distance_Traveled/Total/ALR_total_distance_Week_{point}_post_hoc.txt", 'w') as f:
        f.write(str(posthoc))
    
# Get the baseline values
baseline_df = distance_df[distance_df['Time_Point'] == 0].set_index('Animal_ID')['Total_Distance']

# Normalize the total distance
distance_df['Normalized_Distance'] = distance_df.apply(lambda row: row['Total_Distance'] / baseline_df.loc[row['Animal_ID']], axis=1)

import matplotlib.pyplot as plt

# Create a line plot using seaborn
import seaborn as sns

# Define the color mapping
color_dict = {'Sham': 'green', 'Treated': 'purple', 'Injured': 'red', 'ABX': 'blue'}

# Calculate the mean values
mean_values = distance_df.groupby(['Time_Point', 'Group'])['Normalized_Distance'].mean().reset_index()

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=distance_df, x='Time_Point', y='Normalized_Distance', hue='Group', palette=color_dict, err_style='band', ci='sd')
sns.scatterplot(data=mean_values, x='Time_Point', y='Normalized_Distance', hue='Group', palette=color_dict, legend=False)

# Add the mean values to the plot
for _, row in mean_values.iterrows():
    plt.text(row['Time_Point'], row['Normalized_Distance'], f"{row['Normalized_Distance']:.2f}")

plt.title('Mean Distance Traveled by Group')
plt.ylabel('Baseline Normalized Distance')
plt.xlabel('Week')
plt.tight_layout()
# Save the plot
plt.savefig("/Users/adanredwine/Desktop/Open_Arena/Distance_Traveled/Total/ALR_total_distance_line_plot.svg")