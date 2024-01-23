import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy
import scipy.cluster.hierarchy as sch
import scipy.signal
import seaborn as sns
import umap
from collections import defaultdict
from icecream import ic
from vame.util.auxiliary import read_config
from pathlib import Path

def plot_distribution_of_motif(df, meta_data, motif=None, group=None, time_point=None):
    motifs = range(df.shape[0]) if motif is None else [motif]
    groups = meta_data['Group'].unique() if group is None else [group]
    time_points = meta_data['Time_Point'].unique() if time_point is None else [time_point]

    save_dir = "/Users/adanredwine/Desktop/VAME_normality_figs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for motif in motifs:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        axes = axes.flatten()
        fig.suptitle(f"Motif {motif}")

    # Create a plot for each motif using data from all time points and groups
    for motif in motifs:
        # Create a figure with 4 subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        # Flatten the axes array
        axes = axes.flatten()
        # Set the title for the figure
        fig.suptitle(f"Motif {motif}")
        
        # Iterate over each group
        for i, group in enumerate(groups):
            # Filter the meta data for the current group
            filtered_meta = meta_data[meta_data['Group'] == group]
            # Get the data for the current motif and group
            group_data = df.loc[motif, filtered_meta['Name']]
            # Flatten the data
            group_data = group_data.values.flatten()
            # Remove NaN values
            group_data = group_data[~np.isnan(group_data)]
            
            # Plot the data in the appropriate subplot
            sns.histplot(group_data, ax=axes[i])
            axes[i].set_title(group)
            
        # Iterate over each time point
        for i, time_point in enumerate(time_points):
            if i + 2 < len(axes):
                # Filter the meta data for the current time point
                filtered_meta = meta_data[meta_data['Time_Point'] == time_point]
                # Get the data for the current motif and time point
                time_point_data = df.loc[motif, filtered_meta['Name']]
                # Flatten the data
                time_point_data = time_point_data.values.flatten()
                # Remove NaN values
                time_point_data = time_point_data[~np.isnan(time_point_data)]
                
                # Plot the data in the appropriate subplot
                sns.histplot(time_point_data, ax=axes[i+2])
                axes[i+2].set_title(time_point)
            
        fig_filename = f"Motif_{motif}_plot.png"
        fig_path = os.path.join(save_dir, fig_filename)
        print(f"Saving plot to {fig_path}")  # Debugging print
        fig.savefig(fig_path)
        plt.close(fig)
        
        
def plot_distribution_of_motif_2(df, meta_data, motif=None):
    motifs = range(df.shape[0]) if motif is None else [motif]
    groups = meta_data['Group'].unique()
    time_points = meta_data['Time_Point'].unique()
    save_dir = "/Users/adanredwine/Desktop/VAME_normality_figs"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for motif in motifs:
        # Create a figure for each motif with a subplot for each group and time point
        fig, axes = plt.subplots(nrows=len(groups), ncols=len(time_points), figsize=(len(time_points) * 5, len(groups) * 5))
        fig.suptitle(f"Motif {motif}", fontsize=16)

        for i, group in enumerate(groups):
            for j, time_point in enumerate(time_points):
                ax = axes[i, j]

                # Filter the data for the current group and time point
                filtered_meta = meta_data[(meta_data['Group'] == group) & (meta_data['Time_Point'] == time_point)]
                motif_data = df.loc[motif, filtered_meta['Name']]
                motif_data = motif_data.values.flatten()
                motif_data = motif_data[~np.isnan(motif_data)]

                sns.histplot(motif_data, ax=ax, kde=True)
                ax.set_title(f"{group} - {time_point}")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_filename = f"Motif_{motif}_plot.png"
        fig_path = os.path.join(save_dir, fig_filename)
        fig.savefig(fig_path)
        plt.close(fig)
    

def plot_distribution_of_motif_3(df, meta_data, motif=None):
    motifs = range(df.shape[0]) if motif is None else [motif]
    groups = meta_data['Group'].unique()
    time_points = meta_data['Time_Point'].unique()
    save_dir = "/Users/adanredwine/Desktop/VAME_normality_figs"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for motif in motifs:
        fig, axes = plt.subplots(nrows=len(time_points), ncols=1, figsize=(10, 5 * len(time_points)))
        if len(time_points) == 1:
            axes = [axes]
        
        fig.suptitle(f"Motif {motif}", fontsize=16)

        for i, time_point in enumerate(time_points):
            ax = axes[i]
            
            for group in groups:
                filtered_meta = meta_data[(meta_data['Group'] == group) & (meta_data['Time_Point'] == time_point)]
                motif_data = df.loc[motif, filtered_meta['Name']]
                motif_data = motif_data.values.flatten()
                motif_data = motif_data[~np.isnan(motif_data)]

                # Use kdeplot for a smoother distribution curve
                sns.kdeplot(motif_data, ax=ax, label=group, fill=True, alpha=0.5)
            
            ax.set_title(f"Time Point: {time_point}", fontsize=14)
            ax.set_xlabel('Motif Usage')
            ax.set_ylabel('Density')
            ax.legend(title='Group')
            ax.grid(True)
            ax.set_xlim(left=0)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_filename = f"Motif_{motif}_plot.png"
        fig_path = os.path.join(save_dir, fig_filename)
        fig.savefig(fig_path)
        plt.close(fig)




def plot_line_chart(df_long):
    save_dir = "/Users/adanredwine/Desktop/VAME_line_plots"
    colors = {'Sham': '#2ca02c', 'ABX': '#1f77b4', 'Treated': '#b662ff', 'Injured': '#d42163'}
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Calculate mean and standard deviation for each group, time point, and motif
    stats = calculate_mean_and_sd(df_long, normalization=True)

    # Sort the Time_Point values
    sorted_time_points = sorted(stats['Time_Point'].unique())
    stats['Time_Point'] = pd.Categorical(stats['Time_Point'], categories=sorted_time_points, ordered=True)
    
    # Plotting
    for motif in stats['Motif'].unique():
        plt.figure(figsize=(10, 6))
        motif_stats = stats[stats['Motif'] == motif]
        
        # Plot each group with its error band
        for group in motif_stats['Group'].unique():
            group_stats = motif_stats[motif_stats['Group'] == group]
            sns.lineplot(data=group_stats, x='Time_Point', y='mean', label=group, color=colors[group], marker='o')
            plt.fill_between(group_stats['Time_Point'], group_stats['mean'] - group_stats['std'], group_stats['mean'] + group_stats['std'], color=colors[group], alpha=0.2)
        
        plt.title(f'Normalized Values Over Time for Motif {motif}')
        plt.xlabel('Time Point')
        plt.ylabel('Mean Normalized Value')
        plt.xticks(rotation=45)
        plt.legend(title='Group', loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        plt.tight_layout()

        plt_filename = f"Motif_{motif}_line_plot.png"
        plt_path = os.path.join(save_dir, plt_filename)
        plt.savefig(plt_path)
        plt.close()


def plot_bar_chart(df_long):
    
    save_dir = "/Users/adanredwine/Desktop/VAME_bar_plots"
    colors = {'Sham': '#2ca02c', 'ABX': '#1f77b4', 'Treated': '#b662ff', 'Injured': '#d42163'}
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Calculate mean normalized values for each group, time point, and motif
    mean_normalized = df_long.groupby(['Motif', 'Time_Point', 'Group'])['Normalized_Value'].mean().reset_index()
    
    # Sort the Time_Point column
    time_points = mean_normalized['Time_Point'].unique()
    time_points = sorted(time_points, key=lambda x: x if x != 'Drug_Trt' else 'ZDrug_Trt')
    mean_normalized['Time_Point'] = pd.Categorical(mean_normalized['Time_Point'], categories=time_points, ordered=True)
    
    # Plotting
    unique_motifs = df_long['Motif'].unique()
    
    for motif in unique_motifs:
        plt.figure(figsize=(10, 6))
        motif_data = mean_normalized[mean_normalized['Motif'] == motif]
        
        barplot = sns.barplot(data=motif_data, x='Time_Point', y='Normalized_Value', hue='Group', palette=colors)
        
        plt.title(f'Normalized Values Over Time for Motif {motif}')
        plt.xlabel('Time Point')
        plt.ylabel('Mean Normalized Value')
        plt.xticks(rotation=45)
        plt.legend(title='Group', loc='upper left', bbox_to_anchor=(1, 1))  # Move the legend outside the plot
        plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        plt.tight_layout()

        plt_filename = f"Motif_{motif}_bar_plot.png"
        plt_path = os.path.join(save_dir, plt_filename)
        plt.savefig(plt_path)
        plt.close()


def plot_bar_chart_2(df_long):
    save_dir = "/Users/adanredwine/Desktop/VAME_bar_plots"
    colors = {'Sham': '#2ca02c', 'ABX': '#1f77b4', 'Treated': '#b662ff', 'Injured': '#d42163'}
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Calculate mean and standard deviation for each group and time point

    stats = calculate_mean_and_sd(df_long, normalization=True)

    # Plotting
    for motif in stats['Motif'].unique():
        plt.figure(figsize=(10, 6))
        motif_stats = stats[stats['Motif'] == motif].copy()
        
        # Create a barplot
        barplot = sns.barplot(
            data=motif_stats,
            x='Time_Point',
            y='mean',
            hue='Group',
            palette=colors,
            capsize=.1
        )
        
        # Iterate over the bars and add error bars
        for bar, stddev in zip(barplot.patches, motif_stats['std']):
            bar.set_width(0.25)
            plt.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=stddev,
                ecolor='black',
                capsize=3,
                fmt='none'
            )
        
        # Adjust x-ticks and labels
        plt.xticks(
            rotation=45,
            horizontalalignment='right'
        )
        
        plt.title(f'Normalized Values Over Time for Motif {motif}')
        plt.xlabel('Time Point')
        plt.ylabel('Mean Normalized Value')
        plt.legend(title='Group', loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        plt.tight_layout()

        plt_filename = f"Motif_{motif}_bar_plot.png"
        plt_path = os.path.join(save_dir, plt_filename)
        plt.savefig(plt_path)
        plt.close()


def plot_violin_chart(df_long):
    save_dir = "/Users/adanredwine/Desktop/VAME_violin_plots"
    colors = {'Sham': '#2ca02c', 'ABX': '#1f77b4', 'Treated': '#b662ff', 'Injured': '#d42163'}
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Plotting
    for motif in df_long['Motif'].unique():
        plt.figure(figsize=(10, 6))
        motif_stats = df_long[df_long['Motif'] == motif].copy()
        
        # Create a violin plot
        sns.violinplot(
            data=motif_stats,
            x='Time_Point',
            y='Normalized_Value',  # Assuming 'value' is the column with the data to plot
            hue='Group',
            palette=colors,
            split=False,  # Set to True if you want to compare two groups side by side
            inner='quartile'  # Show the quartiles inside the violin
        )
        
        # Adjust x-ticks and labels
        plt.xticks(
            rotation=45,
            horizontalalignment='right'
        )
        
        plt.title(f'Normalized Values Over Time for Motif {motif}')
        plt.xlabel('Time Point')
        plt.ylabel('Normalized Value')
        plt.legend(title='Group', loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        plt.tight_layout()

        plt_filename = f"Motif_{motif}_violin_plot.png"
        plt_path = os.path.join(save_dir, plt_filename)
        plt.savefig(plt_path)
        plt.close()

# Call the function with your DataFrame
# plot_violin_chart(df_long)

def plot_aggregated_community_sizes(config, aggregated_community_sizes):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    parameterization = cfg['parameterization']
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    hmm_iters = cfg.get('hmm_iters', 0)
    
    # Sort communities by size
    sorted_communities = sorted(aggregated_community_sizes.items(), key=lambda item: item[1], reverse=True)
    
    # Unpack the sorted items for plotting
    communities, sizes = zip(*sorted_communities)
    
    plt.figure(figsize=(10, 5))
    plt.bar(communities, sizes)
    plt.xlabel('Community')
    plt.ylabel('Number of Motifs')
    plt.title('Aggregated Community Sizes Across All Files')

    path_to_fig =  os.path.join(cfg['project_path'], 'results')

    if not os.path.exists(os.path.join(path_to_fig,"community_louvain")):
        os.mkdir(os.path.join(path_to_fig,"community_louvain"))

    if parameterization == 'hmm':
        plt.savefig(os.path.join(path_to_fig, "community_louvain", f"Aggregated_communities-{parameterization}-{n_cluster}-{hmm_iters}.svg"))
    else:
        plt.savefig(os.path.join(path_to_fig, "community_louvain", f"Aggregated_communities-{parameterization}-{n_cluster}.svg"))
    plt.close()


def plot_3d_transition_heatmap(transition_matrix, cmap='flare'):
    """
    Plots a 3D surface plot of the transition matrix.

    Parameters:
    transition_matrix (np.array): A 2D numpy array representing the transition probabilities.
    cmap (str): The colormap for the surface plot.

    Returns:
    matplotlib.figure.Figure: The figure object for the plot.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create grid
    x = np.arange(transition_matrix.shape[1])
    y = np.arange(transition_matrix.shape[0])
    x, y = np.meshgrid(x, y)

    # Create surface plot
    surf = ax.plot_surface(x, y, transition_matrix, cmap=cmap, rstride=1, cstride=1)

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Set labels
    ax.set_xlabel('Motif at t')
    ax.set_ylabel('Motif at t+1')
    ax.set_zlabel('Transition Probability')

    return fig


def create_cluster_metric_heatmap(comparison_metrics, weeks, metric_key):
    # Sort weeks so that "Drug_Trt" is last
    weeks_sorted = sorted(week for week in weeks if week != "Drug_Trt") + ["Drug_Trt"] if "Drug_Trt" in weeks else weeks

    # Initialize an empty matrix for the heatmap data
    heatmap_data = np.full((len(weeks_sorted), len(weeks_sorted)), np.nan)

    # Populate the heatmap data with the metric values
    for i, week_i in enumerate(weeks_sorted):
        for j, week_j in enumerate(weeks_sorted):
            if i != j:
                week_pair = (week_i, week_j)
                if week_pair in comparison_metrics:
                    heatmap_data[i, j] = comparison_metrics[week_pair][metric_key]

    # Create the heatmap
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='flare', xticklabels=weeks_sorted, yticklabels=weeks_sorted)
    plt.title(f'Heatmap of {metric_key}')
    plt.xlabel('Weeks')
    plt.ylabel('Weeks')

    return fig

def plot_motif_statistics(avg_motif_duration, avg_motif_frequency, avg_motif_speed, clusters):
    """
    Plots the average motif duration, frequency, and speed on the same bar chart, arranged by cluster.

    Parameters:
    avg_motif_duration (dict): Average duration of each motif.
    avg_motif_frequency (dict): Frequency of each motif.
    avg_motif_speed (dict): Average speed of each motif.
    clusters (np.array): Array of cluster labels for each motif.
    """
    # Create a list of motifs sorted by cluster
    motifs_sorted_by_cluster = sorted(avg_motif_duration.keys(), key=lambda motif: clusters[motif])
    
    # Extract the values in the same order for each statistic
    durations = [avg_motif_duration[motif] for motif in motifs_sorted_by_cluster]
    frequencies = [avg_motif_frequency[motif] for motif in motifs_sorted_by_cluster]
    speeds = [avg_motif_speed[motif] for motif in motifs_sorted_by_cluster]
    
    # Set the positions of the bars
    x = np.arange(len(motifs_sorted_by_cluster))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 8))
    
    rects1 = ax.bar(x - width, durations, width, label='Duration')
    rects2 = ax.bar(x, frequencies, width, label='Frequency')

    # Create a second y-axis for the average speed
    ax2 = ax.twinx()
    rects3 = ax2.bar(x + width, speeds, width, label='Speed (cm/s)', color='green')

    # Determine the unique clusters and their colors
    unique_clusters = np.unique(clusters)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_clusters)))  # Colormap for clusters

    # Color mapping for each cluster
    cluster_colors = {cluster: color for cluster, color in zip(unique_clusters, colors)}

    # Add colored spans for each cluster
    for cluster in unique_clusters:

        # Find the indices of motifs in the current cluster
        cluster_indices = [i for i, motif in enumerate(motifs_sorted_by_cluster) if clusters[motif] == cluster]

        if cluster_indices:
            # Find the start and end positions for the cluster span
            start_pos = min(cluster_indices) - 2.5 * width
            end_pos = max(cluster_indices) + 1.5 * width
            
            # Add the colored span for the cluster
            ax.axvspan(start_pos, end_pos, color=cluster_colors[cluster], alpha=0.20)

    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_xlabel('Motifs')
    ax.set_ylabel('Duration/Frequency')
    ax2.set_ylabel('Speed (cm/s)', color='green')  # Set the label for the second y-axis
    ax.set_title('Average Motif Statistics by Cluster')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{motif} (C{clusters[motif]})" for motif in motifs_sorted_by_cluster], rotation=45)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Create proxy artists for the cluster colors
    cluster_lines = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    cluster_labels = [f'Cluster {cluster}' for cluster in unique_clusters]
    ax.legend(lines + lines2 + cluster_lines, labels + labels2 + cluster_labels, loc='upper left')

    return fig

def plot_3d_jaccard_indices(comparison_metrics, weeks):
    # Sort weeks so that "Drug_Trt" is last
    weeks_sorted = sorted(week for week in weeks if week != "Drug_Trt") + ["Drug_Trt"] if "Drug_Trt" in weeks else weeks

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Create a colormap
    cmap = plt.get_cmap('flare')

    # Initialize lists to store color and size data for the colorbar
    colors = []
    sizes = []

    # Iterate over each pair of sorted weeks
    for i, week_i in enumerate(weeks_sorted):
        for j, week_j in enumerate(weeks_sorted):
            if i != j:
                week_pair = (week_i, week_j)
                if week_pair in comparison_metrics:
                    jaccard_indices = comparison_metrics[week_pair]['Jaccard Indices']
                    for z, jaccard_index in enumerate(jaccard_indices):
                        # Size and color based on the Jaccard index value
                        size = jaccard_index * 100  # Example scaling factor for size
                        color = cmap(jaccard_index)  # Map the Jaccard index to a color
                        
                        # Store color and size for the colorbar
                        colors.append(color)
                        sizes.append(size)
                        
                        # Plot a sphere at the (i, j, z) position
                        ax.scatter(i, j, z, s=size, color=color, alpha=0.65)

    # Create colorbar for the colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(jaccard_indices), vmax=max(jaccard_indices)))
    sm.set_array([])  # You have to set_array for newer versions of matplotlib
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Jaccard Index')

    # Create size legend for the sizes
    # Generate representative handles for the legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'{s/100:.2f}',
                          markerfacecolor='k', markersize=np.sqrt(s)) for s in np.unique(sizes)]
    ax.legend(handles=handles, title="Jaccard Index Size", loc='upper left')

    # Set labels and title
    ax.set_xlabel('Weeks')
    ax.set_ylabel('Weeks')
    ax.set_zlabel('Jaccard Index')
    ax.set_title('3D Plot of Jaccard Indices')

    # Set the tick labels for x and y axes to the sorted week names
    ax.set_xticks(range(len(weeks_sorted)))
    ax.set_xticklabels(weeks_sorted)
    ax.set_yticks(range(len(weeks_sorted)))
    ax.set_yticklabels(weeks_sorted)

    return fig


def plot_transition_counts_heatmap(comparison_metrics, week_pair):
    transition_counts = comparison_metrics[week_pair]['Transition Counts']

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(transition_counts, annot=True, fmt='g')
    plt.title(f'Transition Counts: {week_pair[0]} vs {week_pair[1]}')
    plt.xlabel(f'Clusters in {week_pair[1]}')
    plt.ylabel(f'Clusters in {week_pair[0]}')
    plt.tight_layout()

    return fig


def plot_3d_weekly_transition_counts(comparison_metrics, weeks):
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')

    num_weeks = len(weeks)
    num_clusters = max(max(len(metrics['Transition Counts'][0]), len(metrics['Transition Counts'][1])) for metrics in comparison_metrics.values())

    # Initialize z as zeros for each pair of weeks and clusters
    z = np.zeros((num_weeks, num_weeks, num_clusters), dtype=float)

    # Populate z with transition counts
    for (week_i, week_j), metrics in comparison_metrics.items():
        i = weeks.index(week_i)
        j = weeks.index(week_j)
        transition_counts = metrics['Transition Counts']
        padded_transition_counts = np.pad(transition_counts, ((0, 0), (0, num_clusters - transition_counts.shape[1])), 'constant')
        z[i, j, :] = padded_transition_counts.sum(axis=0)

    # Create a mask for non-zero transition counts
    non_zero_mask = z > 0

    # Apply the mask to z and get the corresponding x, y indices
    x, y, z = np.nonzero(non_zero_mask)
    z_values = z[non_zero_mask]

    # Plot bars
    ax.bar3d(x, y, np.zeros_like(z_values), 0.4, 0.4, z_values, shade=True)

    # Set labels and title
    ax.set_xlabel('Weeks')
    ax.set_ylabel('Clusters')
    ax.set_zlabel('Transition Count')
    ax.set_title('3D Transition Counts Between Weeks')

    # Set ticks for weeks and clusters
    ax.set_xticks(range(num_weeks))
    ax.set_xticklabels(weeks)
    ax.set_yticks(range(num_clusters))
    ax.set_yticklabels(range(num_clusters))

    return fig


def plot_normalized_values_by_group_and_timepoint(df, p_values_df, significance_level=0.05, specific_time_point=None, log_norm = False):
    
    # Define a color mapping for groups
    group_colors = {'Sham': '#2ca02c', 'ABX': '#1f77b4', 'Treated': '#b662ff', 'Injured': '#d42163'}
    
    df = df.merge(p_values_df, on=['Animal_ID', 'Time_Point', 'Group', 'Motif'], how='left')

    # If a specific time point is provided, filter the DataFrame to only include that time point
    if specific_time_point is not None:
        df = df[df['Time_Point'] == specific_time_point]

    if log_norm == 'False':
        if specific_time_point is not None:
            grouped = df.groupby(['Cluster', 'Motif', 'Group']).agg({
            'Normalized_Value': 'mean',
            'P_Value': 'mean'
            }).reset_index()
        else:
            # Group by the necessary columns and calculate the mean 'Normalized_Value' and mean 'P_Value'
            grouped = df.groupby(['Time_Point', 'Cluster', 'Motif', 'Group']).agg({
                'Normalized_Value': 'mean',
                'P_Value': 'mean'
            }).reset_index()
    else:
        if specific_time_point is not None:
            grouped = df.groupby(['Cluster', 'Motif', 'Group']).agg({
            'Log_Normalized_Value': 'mean',
            'P_Value': 'mean'
            }).reset_index()
        else:
            # Group by the necessary columns and calculate the mean 'Normalized_Value' and mean 'P_Value'
            grouped = df.groupby(['Time_Point', 'Cluster', 'Motif', 'Group']).agg({
            'Log_Normalized_Value': 'mean',
            'P_Value': 'mean'
            }).reset_index()


    # Sort the DataFrame by 'Cluster' and 'Motif' for plotting
    grouped_sorted = grouped.sort_values(by=['Cluster', 'Motif'])
    time_points = grouped_sorted['Time_Point'].unique() if specific_time_point is None else [specific_time_point]
    groups = grouped_sorted['Group'].unique()

    # Determine global y-axis limits
    if log_norm == 'False':
        global_min = grouped_sorted['Normalized_Value'].min()
        global_max = grouped_sorted['Normalized_Value'].max() + 2
    else:
        global_min = grouped_sorted['Log_Normalized_Value'].min()
        global_max = grouped_sorted['Log_Normalized_Value'].max() + 2

    # Create subplots for each time point
    nrows = 1 if specific_time_point is not None else len(time_points)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 6 * nrows), sharex=True) # type: ignore

    # If there's only one time point, axes will not be an array, so we wrap it in a list
    if len(time_points) == 1:
        axes = [axes]

    # Plot each time point in a separate subplot
    for ax, time_point in zip(axes, time_points):
            # Calculate the offset as a fraction of the distance between motifs
        offset_fraction = 0.05  # Adjust this value as needed to prevent overlap
        offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * offset_fraction / len(groups)
        
        # Filter the DataFrame for the current time point
        if specific_time_point is None:
            data_time_point = grouped_sorted[grouped_sorted['Time_Point'] == time_point]
        else:
            data_time_point = grouped_sorted

        # Plot each group as a separate line in the subplot
        for group_idx, group in enumerate(groups):
            data_group = data_time_point[data_time_point['Group'] == group]
            group_color = group_colors.get(group, 'gray')  # Default to 'gray' if group not in dictionary
            
            if log_norm == 'False':
                line = sns.lineplot(data=data_group, x='Motif', y='Normalized_Value', ax=ax, label=group, marker='o', color = group_color)
            else:
                line = sns.lineplot(data=data_group, x='Motif', y='Log_Normalized_Value', ax=ax, label=group, marker='o', color = group_color)
            #group_color = ax.lines[-1].get_color() 
            group_color = line.get_lines()[-1].get_color() 
            
            # Filter significant points for the current group and time point
            if specific_time_point is None:
                #significant_points = df[(df['Time_Point'] == time_point) & (df['Group'] == group) & (df['P_Value'] <= significance_level)]
                significant_points = data_group[(data_group['Time_Point'] == time_point) & (data_group['Group'] == group) & (data_group['P_Value'] <= significance_level)]
            else:
                #significant_points = df[(df['Group'] == group) & (df['P_Value'] <= significance_level)]
                significant_points = data_group[(data_group['Group'] == group) & (data_group['P_Value'] <= significance_level)]


            # Initialize a dictionary to keep track of the number of significant points plotted for each motif
            motif_significant_count = defaultdict(int)

            # Plot the significant points directly above the motif they are marking as significant
            for _, point in significant_points.iterrows():
                motif = point['Motif']
                # Increment the count for the motif
                motif_significant_count[motif] += 1
                # Calculate the x position based on the motif and the offset for the group
                x_value = motif # + (group_idx - len(groups) / 2) * offset
                if log_norm == 'False':
                # Find the maximum normalized value for the current motif across all groups within the time point
                    max_normalized_value = data_time_point[data_time_point['Motif'] == motif]['Normalized_Value'].max()
                else:
                    max_normalized_value = data_time_point[data_time_point['Motif'] == motif]['Log_Normalized_Value'].max()

                # Calculate the y position based on the number of significant points already plotted for this motif
                #y_value = global_max + 0.1 * motif_significant_count[motif]
                y_value = (max_normalized_value + 0.5) + motif_significant_count[motif]

                # Ensure the y position is within the y-axis limits
                y_value = min(y_value, ax.get_ylim()[1])
                # Plot the significant marker
                ax.scatter(x_value, y_value, color=group_color, marker='*', s=50)

        # Set the title and labels
        ax.set_xlabel('Motif')
        if log_norm == 'False':
            ax.set_ylabel('Normalized Value')
        else:
            ax.set_ylabel('Log Normalized Value')

        # Set the same y-axis limits for all subplots
        ax.set_ylim((global_min - 0.5), (global_max + 0.5))

        # Set motif labels on the x-axis
        motifs = data_time_point['Motif'].unique()
        ax.set_xticks(motifs)
        ax.set_xticklabels(motifs)

        # Move the legend to the right of the plot
        ax.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Place the Time Point text in the upper right corner of the subplot
        ax.text(0.95, 0.95, time_point, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust the right edge to make room for the legend
    plt.show()


def plot_normalized_values_by_group_and_timepoint_2(df, p_values_df, significance_level=0.05, specific_time_point=None, log_norm=False):
    # Define a color mapping for groups
    group_colors = {'Sham': '#2ca02c', 'ABX': '#1f77b4', 'Treated': '#b662ff', 'Injured': '#d42163'}
    
    df = df.merge(p_values_df, on=['Animal_ID', 'Time_Point', 'Group', 'Motif'], how='left')

    # If a specific time point is provided, filter the DataFrame to only include that time point
    if specific_time_point is not None:
        df = df[df['Time_Point'] == specific_time_point]

    # Determine the value column to use based on log_norm flag
    value_column = 'Log_Normalized_Value' if log_norm else 'Normalized_Value'

    # Group by the necessary columns and calculate the mean value and mean 'P_Value'
    grouped = df.groupby(['Cluster', 'Motif', 'Group', 'Time_Point']).agg({
        value_column: 'mean',
        'P_Value': 'mean'
    }).reset_index()

    # Sort the DataFrame by 'Cluster' and 'Motif' for plotting
    grouped_sorted = grouped.sort_values(by=['Cluster', 'Motif'])
    time_points = grouped_sorted['Time_Point'].unique()
    groups = grouped_sorted['Group'].unique()

    # Determine global y-axis limits
    global_min = grouped_sorted[value_column].min()
    global_max = grouped_sorted[value_column].max() + 2

    # Create subplots for each time point
    nrows = len(time_points) if specific_time_point is None else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 6 * nrows), sharex=True)

    # If there's only one time point, axes will not be an array, so we wrap it in a list
    if nrows == 1:
        axes = [axes]

    # Plot each time point in a separate subplot
    for ax, time_point in zip(axes, time_points):
        offset_fraction = 0.05  # Adjust this value as needed to prevent overlap
        offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * offset_fraction / len(groups)
        
        # Filter the DataFrame for the current time point
        data_time_point = grouped_sorted[grouped_sorted['Time_Point'] == time_point]

        # Plot each group as a separate line in the subplot
        for group_idx, group in enumerate(groups):
            data_group = data_time_point[data_time_point['Group'] == group]
            group_color = group_colors.get(group, 'gray')  # Default to 'gray' if group not in dictionary
            
            sns.lineplot(data=data_group, x='Motif', y=value_column, ax=ax, label=group, marker='o', color=group_color)

            # Filter significant points for the current group and time point
            significant_points = data_group[data_group['P_Value'] <= significance_level]

            # Plot the significant points directly above the motif they are marking as significant
            for _, point in significant_points.iterrows():
                motif = point['Motif']
                
                cluster = point['Cluster']
                # Find x position based on cluster and motif, adjusting for group offset
                cluster_motifs = data_time_point[data_time_point['Cluster'] == cluster]['Motif'].unique()
                motif_index = list(cluster_motifs).index(motif)
                
                #x_value = motif
                
                x_value = cluster_motifs[motif_index] + (group_idx - len(groups) / 2) * offset

                max_value = data_time_point[data_time_point['Motif'] == motif][value_column].max()
                y_value = max_value + 0.1  # Slightly above the max value for visibility
                ax.scatter(x_value, y_value, color=group_color, marker='*', s=50)

        # Set the title and labels
        ax.set_title(f'Time Point: {time_point}')
        ax.set_xlabel('Motif')
        ax.set_ylabel(value_column)

        # Set the same y-axis limits for all subplots
        ax.set_ylim((global_min - 0.5), (global_max + 0.5))

        # Set motif labels on the x-axis
        motifs = data_time_point['Motif'].unique()
        ax.set_xticks(motifs)
        ax.set_xticklabels(motifs)

        # Move the legend to the right of the plot
        ax.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust the right edge to make room for the legend
    if specific_time_point:
        plt.suptitle(f'Normalized Values for Time Point: {specific_time_point}', y=1.02)
    else:
        plt.suptitle('Normalized Values by Time Point', y=1.02)
    
    plt.show()


def plot_normalized_values_by_group_and_timepoint_3(df, p_values_df, significance_level=0.05, specific_time_point=None, log_norm=False):
    # Define a color mapping for groups
    group_colors = {'Sham': '#2ca02c', 'ABX': '#1f77b4', 'Treated': '#b662ff', 'Injured': '#d42163'}
    
    df = df.merge(p_values_df, on=['Animal_ID', 'Time_Point', 'Group', 'Motif'], how='left')

    # If a specific time point is provided, filter the DataFrame to only include that time point
    if specific_time_point is not None:
        df = df[df['Time_Point'] == specific_time_point]

    # Determine the value column to use based on log_norm flag
    value_column = 'Log_Normalized_Value' if log_norm else 'Normalized_Value'

    # Group by the necessary columns and calculate the mean value and mean 'P_Value'
    grouped = df.groupby(['Cluster', 'Motif', 'Group', 'Time_Point']).agg({
        value_column: 'mean',
        'P_Value': 'mean'
    }).reset_index()

    # Sort the DataFrame by 'Cluster' and 'Motif' for plotting
    grouped_sorted = grouped.sort_values(by=['Cluster', 'Motif'])
    time_points = grouped_sorted['Time_Point'].unique()
    groups = grouped_sorted['Group'].unique()

    # Determine global y-axis limits
    global_min = grouped_sorted[value_column].min()
    global_max = grouped_sorted[value_column].max() + 2

    # Create subplots for each time point
    nrows = len(time_points) if specific_time_point is None else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 6 * nrows), sharex=True)

    # If there's only one time point, axes will not be an array, so we wrap it in a list
    if nrows == 1:
        axes = [axes]

    epsilon = 1e-6  # A small value to ensure the line is plotted

    # Add epsilon to the Sham group's Log_Normalized_Value before plotting
    grouped_sorted.loc[grouped_sorted['Group'] == 'Sham', value_column] += epsilon
    
    grouped_sorted = grouped_sorted.copy()

    cluster_motif_combinations = grouped_sorted[['Cluster', 'Motif']].drop_duplicates()
    cluster_motif_combinations['x_value'] = range(len(cluster_motif_combinations))
    mapping = cluster_motif_combinations.set_index(['Cluster', 'Motif'])['x_value'].to_dict()

    # Create a new column 'x_value' using the mapping dictionary
    grouped_sorted['x_value'] = grouped_sorted.apply(lambda row: mapping[(row['Cluster'], row['Motif'])], axis=1)

    # Plot each time point in a separate subplot
    for ax, time_point in zip(axes, time_points):
        offset_fraction = 0.05  # Adjust this value as needed to prevent overlap
        offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * offset_fraction / len(groups)


        # Filter the DataFrame for the current time point
        data_time_point = grouped_sorted[grouped_sorted['Time_Point'] == time_point]

        # Plot each group as a separate line in the subplot
        for group_idx, group in enumerate(groups):
            data_group = data_time_point[data_time_point['Group'] == group]
            #data_group['x_value'] = data_group.apply(lambda row: mapping[(row['Cluster'], row['Motif'])], axis=1)
            group_color = group_colors.get(group, 'gray')  # Default to 'gray' if group not in dictionary
            
            #sns.lineplot(data=data_group, x='Motif', y=value_column, ax=ax, label=group, marker='o', color=group_color)
            sns.lineplot(data=data_group, x='x_value', y=value_column, ax=ax, label=group, marker='o', color=group_color)

            # Filter significant points for the current group and time point
            significant_points = data_group[data_group['P_Value'] <= significance_level]
            motif_significant_count = defaultdict(int)
            # Plot the significant points directly above the motif they are marking as significant
            for _, point in significant_points.iterrows():
                motif = point['Motif']
                motif_significant_count[motif] += 1
                x_value = motif

                max_value = data_time_point[data_time_point['Motif'] == motif][value_column].max()
                y_value = (max_value + 0.1) + (motif_significant_count[motif])
                ax.scatter(x_value, y_value, color=group_color, marker='*', s=50)

        # Set the title and labels
        ax.set_title(f'Time Point: {time_point}')
        ax.set_xlabel('Motif')
        ax.set_ylabel(value_column)

        # Set the same y-axis limits for all subplots
        ax.set_ylim((global_min - 0.5), (global_max + 0.5))

        # Set motif labels on the x-axis
        motifs = data_time_point['Motif'].unique()
        # Update x-axis ticks and labels to reflect cluster-motif ordering
        ax.set_xticks(cluster_motif_combinations['x_value'])
        ax.set_xticklabels([f"{row['Cluster']}_{row['Motif']}" for _, row in cluster_motif_combinations.iterrows()], rotation=90)

        #ax.set_xticks(motifs)
        #ax.set_xticklabels(motifs)

        # Move the legend to the right of the plot
        ax.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust the right edge to make room for the legend
    if specific_time_point:
        plt.suptitle(f'Normalized Values for Time Point: {specific_time_point}', y=1.02)
    else:
        plt.suptitle('Normalized Values by Time Point', y=1.02)
    
    plt.show()


def plot_heatmap_by_group(df, clustered_motifs):
    # Create a mapping from motif to cluster order
    motif_to_cluster = {motif: cluster for cluster, motifs in clustered_motifs.items() for motif in motifs}
    
    # Work with a copy to avoid SettingWithCopyWarning
    df_filtered = df[~df['Time_Point'].isin(['Baseline_1', 'Baseline_2'])].copy()
    df_filtered['Cluster_Order'] = df_filtered['Motif'].map(motif_to_cluster)
    
    # Group by the necessary columns and calculate the mean 'Normalized_Value'
    grouped = df_filtered.groupby(['Time_Point', 'Cluster_Order', 'Motif', 'Group'])['Normalized_Value'].mean().reset_index()

    # Sort the DataFrame by 'Cluster_Order' and 'Motif' for plotting
    grouped_sorted = grouped.sort_values(by=['Cluster_Order', 'Motif'])

    # Get unique groups and time points, excluding baseline and ensuring 'Drug_Trt' is last
    groups = grouped_sorted['Group'].unique()
    time_points = [tp for tp in grouped_sorted['Time_Point'].unique() if tp != 'Drug_Trt'] + ['Drug_Trt']
    
    # Determine the global color scale limits
    global_min = grouped_sorted['Normalized_Value'].min()
    global_max = grouped_sorted['Normalized_Value'].max()

    # Create subplots for each group
    fig, axes = plt.subplots(nrows=1, ncols=len(groups), figsize=(5 * len(groups), 10), sharey=True)
    
    if len(groups) == 1:
        axes = [axes]
    
    # Plot a heatmap for each group
    for ax, group in zip(axes, groups):
        # Use pivot_table to handle duplicate entries
        heatmap_data = grouped_sorted[grouped_sorted['Group'] == group].pivot_table(
            #index='Cluster_Order',
            index='Motif', 
            columns='Time_Point', 
            values='Normalized_Value', 
            aggfunc='mean'
        )
        
        # Sort the index by 'Cluster_Order' to maintain the cluster grouping
        heatmap_data = heatmap_data.sort_index()
        
        # Reindex the DataFrame to ensure 'Drug_Trt' is the last column
        heatmap_data = heatmap_data.reindex(time_points, axis=1)
        
        # Plot the heatmap with a consistent color scale
        sns.heatmap(heatmap_data, ax=ax, cmap='flare', cbar_kws={'label': 'Normalized Value'}, vmin=global_min, vmax=global_max)
        
        # Set the title and adjust the axis
        ax.set_title(f'Group: {group}')
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Motif')
    
    # Adjust the layout
    plt.tight_layout()
    plt.show()
