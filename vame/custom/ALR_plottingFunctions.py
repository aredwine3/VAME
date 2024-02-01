import itertools
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import scipy
import scipy.cluster.hierarchy as sch
import scipy.signal
import seaborn as sns
import umap
from icecream import ic

# Get the parent directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)

# Get the grandparent directory, which should contain the 'vame' directory
grandparent_dir = os.path.dirname(os.path.dirname(script_dir))

# Add the grandparent directory to sys.path
sys.path.append(grandparent_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from vame.util.auxiliary import read_config

if os.environ.get("DISPLAY", "") == "":
    print("no display found. Using non-interactive Agg backend")
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import vame
import vame.custom.ALR_statsFunctions as AlSt


def plot_distribution_of_motif(df, meta_data, motif=None, group=None, time_point=None):
    motifs = range(df.shape[0]) if motif is None else [motif]
    groups = meta_data["Group"].unique() if group is None else [group]
    time_points = (
        meta_data["Time_Point"].unique() if time_point is None else [time_point]
    )

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
            filtered_meta = meta_data[meta_data["Group"] == group]
            # Get the data for the current motif and group
            group_data = df.loc[motif, filtered_meta["Name"]]
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
                filtered_meta = meta_data[meta_data["Time_Point"] == time_point]
                # Get the data for the current motif and time point
                time_point_data = df.loc[motif, filtered_meta["Name"]]
                # Flatten the data
                time_point_data = time_point_data.values.flatten()
                # Remove NaN values
                time_point_data = time_point_data[~np.isnan(time_point_data)]

                # Plot the data in the appropriate subplot
                sns.histplot(time_point_data, ax=axes[i + 2])
                axes[i + 2].set_title(time_point)

        fig_filename = f"Motif_{motif}_plot.png"
        fig_path = os.path.join(save_dir, fig_filename)
        print(f"Saving plot to {fig_path}")  # Debugging print
        fig.savefig(fig_path)
        plt.close(fig)


def plot_distribution_of_motif_2(df, meta_data, motif=None):
    motifs = range(df.shape[0]) if motif is None else [motif]
    groups = meta_data["Group"].unique()
    time_points = meta_data["Time_Point"].unique()
    save_dir = "/Users/adanredwine/Desktop/VAME_normality_figs"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for motif in motifs:
        # Create a figure for each motif with a subplot for each group and time point
        fig, axes = plt.subplots(
            nrows=len(groups),
            ncols=len(time_points),
            figsize=(len(time_points) * 5, len(groups) * 5),
        )
        fig.suptitle(f"Motif {motif}", fontsize=16)

        for i, group in enumerate(groups):
            for j, time_point in enumerate(time_points):
                ax = axes[i, j]

                # Filter the data for the current group and time point
                filtered_meta = meta_data[
                    (meta_data["Group"] == group)
                    & (meta_data["Time_Point"] == time_point)
                ]
                motif_data = df.loc[motif, filtered_meta["Name"]]
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
    groups = meta_data["Group"].unique()
    time_points = meta_data["Time_Point"].unique()
    save_dir = "/Users/adanredwine/Desktop/VAME_normality_figs"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for motif in motifs:
        fig, axes = plt.subplots(
            nrows=len(time_points), ncols=1, figsize=(10, 5 * len(time_points))
        )
        if len(time_points) == 1:
            axes = [axes]

        fig.suptitle(f"Motif {motif}", fontsize=16)

        for i, time_point in enumerate(time_points):
            ax = axes[i]

            for group in groups:
                filtered_meta = meta_data[
                    (meta_data["Group"] == group)
                    & (meta_data["Time_Point"] == time_point)
                ]
                motif_data = df.loc[motif, filtered_meta["Name"]]
                motif_data = motif_data.values.flatten()
                motif_data = motif_data[~np.isnan(motif_data)]

                # Use kdeplot for a smoother distribution curve
                sns.kdeplot(motif_data, ax=ax, label=group, fill=True, alpha=0.5)

            ax.set_title(f"Time Point: {time_point}", fontsize=14)
            ax.set_xlabel("Motif Usage")
            ax.set_ylabel("Density")
            ax.legend(title="Group")
            ax.grid(True)
            ax.set_xlim(left=0)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_filename = f"Motif_{motif}_plot.png"
        fig_path = os.path.join(save_dir, fig_filename)
        fig.savefig(fig_path)
        plt.close(fig)


def plot_line_chart(df_long):
    save_dir = "/Users/adanredwine/Desktop/VAME_line_plots"
    colors = {
        "Sham": "#2ca02c",
        "ABX": "#1f77b4",
        "Treated": "#b662ff",
        "Injured": "#d42163",
    }

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate mean and standard deviation for each group, time point, and motif
    stats = AlSt.calculate_mean_and_sd(df_long, normalization=True)

    # Sort the Time_Point values
    sorted_time_points = sorted(stats["Time_Point"].unique())
    stats["Time_Point"] = pd.Categorical(
        stats["Time_Point"], categories=sorted_time_points, ordered=True
    )

    # Plotting
    for motif in stats["Motif"].unique():
        plt.figure(figsize=(10, 6))
        motif_stats = stats[stats["Motif"] == motif]

        # Plot each group with its error band
        for group in motif_stats["Group"].unique():
            group_stats = motif_stats[motif_stats["Group"] == group]
            sns.lineplot(
                data=group_stats,
                x="Time_Point",
                y="mean",
                label=group,
                color=colors[group],
                marker="o",
            )
            plt.fill_between(
                group_stats["Time_Point"],
                group_stats["mean"] - group_stats["std"],
                group_stats["mean"] + group_stats["std"],
                color=colors[group],
                alpha=0.2,
            )

        plt.title(f"Normalized Values Over Time for Motif {motif}")
        plt.xlabel("Time Point")
        plt.ylabel("Mean Normalized Value")
        plt.xticks(rotation=45)
        plt.legend(title="Group", loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")
        plt.tight_layout()

        plt_filename = f"Motif_{motif}_line_plot.png"
        plt_path = os.path.join(save_dir, plt_filename)
        plt.savefig(plt_path)
        plt.close()


def plot_bar_chart(df_long):
    save_dir = "/Users/adanredwine/Desktop/VAME_bar_plots"
    colors = {
        "Sham": "#2ca02c",
        "ABX": "#1f77b4",
        "Treated": "#b662ff",
        "Injured": "#d42163",
    }

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate mean normalized values for each group, time point, and motif
    mean_normalized = (
        df_long.groupby(["Motif", "Time_Point", "Group"])["Normalized_Value"]
        .mean()
        .reset_index()
    )

    # Sort the Time_Point column
    time_points = mean_normalized["Time_Point"].unique()
    time_points = sorted(
        time_points, key=lambda x: x if x != "Drug_Trt" else "ZDrug_Trt"
    )
    mean_normalized["Time_Point"] = pd.Categorical(
        mean_normalized["Time_Point"], categories=time_points, ordered=True
    )

    # Plotting
    unique_motifs = df_long["Motif"].unique()

    for motif in unique_motifs:
        plt.figure(figsize=(10, 6))
        motif_data = mean_normalized[mean_normalized["Motif"] == motif]

        barplot = sns.barplot(
            data=motif_data,
            x="Time_Point",
            y="Normalized_Value",
            hue="Group",
            palette=colors,
        )

        plt.title(f"Normalized Values Over Time for Motif {motif}")
        plt.xlabel("Time Point")
        plt.ylabel("Mean Normalized Value")
        plt.xticks(rotation=45)
        plt.legend(
            title="Group", loc="upper left", bbox_to_anchor=(1, 1)
        )  # Move the legend outside the plot
        plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")
        plt.tight_layout()

        plt_filename = f"Motif_{motif}_bar_plot.png"
        plt_path = os.path.join(save_dir, plt_filename)
        plt.savefig(plt_path)
        plt.close()


def plot_bar_chart_2(df_long):
    save_dir = "/Users/adanredwine/Desktop/VAME_bar_plots"
    colors = {
        "Sham": "#2ca02c",
        "ABX": "#1f77b4",
        "Treated": "#b662ff",
        "Injured": "#d42163",
    }

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate mean and standard deviation for each group and time point

    stats = AlSt.calculate_mean_and_sd(df_long, normalization=True)

    # Plotting
    for motif in stats["Motif"].unique():
        plt.figure(figsize=(10, 6))
        motif_stats = stats[stats["Motif"] == motif].copy()

        # Create a barplot
        barplot = sns.barplot(
            data=motif_stats,
            x="Time_Point",
            y="mean",
            hue="Group",
            palette=colors,
            capsize=0.1,
        )

        # Iterate over the bars and add error bars
        for bar, stddev in zip(barplot.patches, motif_stats["std"]):
            bar.set_width(0.25)
            plt.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=stddev,
                ecolor="black",
                capsize=3,
                fmt="none",
            )

        # Adjust x-ticks and labels
        plt.xticks(rotation=45, horizontalalignment="right")

        plt.title(f"Normalized Values Over Time for Motif {motif}")
        plt.xlabel("Time Point")
        plt.ylabel("Mean Normalized Value")
        plt.legend(title="Group", loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")
        plt.tight_layout()

        plt_filename = f"Motif_{motif}_bar_plot.png"
        plt_path = os.path.join(save_dir, plt_filename)
        plt.savefig(plt_path)
        plt.close()


def plot_violin_chart(df_long):
    save_dir = "/Users/adanredwine/Desktop/VAME_violin_plots"
    colors = {
        "Sham": "#2ca02c",
        "ABX": "#1f77b4",
        "Treated": "#b662ff",
        "Injured": "#d42163",
    }

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plotting
    for motif in df_long["Motif"].unique():
        plt.figure(figsize=(10, 6))
        motif_stats = df_long[df_long["Motif"] == motif].copy()

        # Create a violin plot
        sns.violinplot(
            data=motif_stats,
            x="Time_Point",
            y="Normalized_Value",  # Assuming 'value' is the column with the data to plot
            hue="Group",
            palette=colors,
            split=False,  # Set to True if you want to compare two groups side by side
            inner="quartile",  # Show the quartiles inside the violin
        )

        # Adjust x-ticks and labels
        plt.xticks(rotation=45, horizontalalignment="right")

        plt.title(f"Normalized Values Over Time for Motif {motif}")
        plt.xlabel("Time Point")
        plt.ylabel("Normalized Value")
        plt.legend(title="Group", loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")
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
    parameterization = cfg["parameterization"]
    model_name = cfg["model_name"]
    n_cluster = cfg["n_cluster"]
    hmm_iters = cfg.get("hmm_iters", 0)

    # Sort communities by size
    sorted_communities = sorted(
        aggregated_community_sizes.items(), key=lambda item: item[1], reverse=True
    )

    # Unpack the sorted items for plotting
    communities, sizes = zip(*sorted_communities)

    plt.figure(figsize=(10, 5))
    plt.bar(communities, sizes)
    plt.xlabel("Community")
    plt.ylabel("Number of Motifs")
    plt.title("Aggregated Community Sizes Across All Files")

    path_to_fig = os.path.join(cfg["project_path"], "results")

    if not os.path.exists(os.path.join(path_to_fig, "community_louvain")):
        os.mkdir(os.path.join(path_to_fig, "community_louvain"))

    if parameterization == "hmm":
        plt.savefig(
            os.path.join(
                path_to_fig,
                "community_louvain",
                f"Aggregated_communities-{parameterization}-{n_cluster}-{hmm_iters}.svg",
            )
        )
    else:
        plt.savefig(
            os.path.join(
                path_to_fig,
                "community_louvain",
                f"Aggregated_communities-{parameterization}-{n_cluster}.svg",
            )
        )
    plt.close()


def plot_3d_transition_heatmap(transition_matrix, cmap="flare"):
    """
    Plots a 3D surface plot of the transition matrix.

    Parameters:
    transition_matrix (np.array): A 2D numpy array representing the transition probabilities.
    cmap (str): The colormap for the surface plot.

    Returns:
    matplotlib.figure.Figure: The figure object for the plot.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Create grid
    x = np.arange(transition_matrix.shape[1])
    y = np.arange(transition_matrix.shape[0])
    x, y = np.meshgrid(x, y)

    # Create surface plot
    surf = ax.plot_surface(x, y, transition_matrix, cmap=cmap, rstride=1, cstride=1)

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Set labels
    ax.set_xlabel("Motif at t")
    ax.set_ylabel("Motif at t+1")
    ax.set_zlabel("Transition Probability")

    return fig


def create_cluster_metric_heatmap(comparison_metrics, weeks, metric_key):
    # Sort weeks so that "Drug_Trt" is last
    weeks_sorted = (
        sorted(week for week in weeks if week != "Drug_Trt") + ["Drug_Trt"]
        if "Drug_Trt" in weeks
        else weeks
    )

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
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="flare",
        xticklabels=weeks_sorted,
        yticklabels=weeks_sorted,
    )
    plt.title(f"Heatmap of {metric_key}")
    plt.xlabel("Weeks")
    plt.ylabel("Weeks")

    return fig


def plot_motif_statistics(
    avg_motif_duration, avg_motif_frequency, avg_motif_speed, clusters
):
    """
    Plots the average motif duration, frequency, and speed on the same bar chart, arranged by cluster.

    Parameters:
    avg_motif_duration (dict): Average duration of each motif.
    avg_motif_frequency (dict): Frequency of each motif.
    avg_motif_speed (dict): Average speed of each motif.
    clusters (np.array): Array of cluster labels for each motif.
    """
    # Create a list of motifs sorted by cluster
    motifs_sorted_by_cluster = sorted(
        avg_motif_duration.keys(), key=lambda motif: clusters[motif]
    )

    # Extract the values in the same order for each statistic
    durations = [avg_motif_duration[motif] for motif in motifs_sorted_by_cluster]
    frequencies = [avg_motif_frequency[motif] for motif in motifs_sorted_by_cluster]
    speeds = [avg_motif_speed[motif] for motif in motifs_sorted_by_cluster]

    # Set the positions of the bars
    x = np.arange(len(motifs_sorted_by_cluster))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 8))

    rects1 = ax.bar(x - width, durations, width, label="Duration")
    rects2 = ax.bar(x, frequencies, width, label="Frequency")

    # Create a second y-axis for the average speed
    ax2 = ax.twinx()
    rects3 = ax2.bar(x + width, speeds, width, label="Speed (cm/s)", color="green")

    # Determine the unique clusters and their colors
    unique_clusters = np.unique(clusters)
    colors = plt.cm.jet(
        np.linspace(0, 1, len(unique_clusters))
    )  # Colormap for clusters

    # Color mapping for each cluster
    cluster_colors = {cluster: color for cluster, color in zip(unique_clusters, colors)}

    # Add colored spans for each cluster
    for cluster in unique_clusters:
        # Find the indices of motifs in the current cluster
        cluster_indices = [
            i
            for i, motif in enumerate(motifs_sorted_by_cluster)
            if clusters[motif] == cluster
        ]

        if cluster_indices:
            # Find the start and end positions for the cluster span
            start_pos = min(cluster_indices) - 2.5 * width
            end_pos = max(cluster_indices) + 1.5 * width

            # Add the colored span for the cluster
            ax.axvspan(start_pos, end_pos, color=cluster_colors[cluster], alpha=0.20)

    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_xlabel("Motifs")
    ax.set_ylabel("Duration/Frequency")
    ax2.set_ylabel("Speed (cm/s)", color="green")  # Set the label for the second y-axis
    ax.set_title("Average Motif Statistics by Cluster")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{motif} (C{clusters[motif]})" for motif in motifs_sorted_by_cluster],
        rotation=45,
    )

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Create proxy artists for the cluster colors
    cluster_lines = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    cluster_labels = [f"Cluster {cluster}" for cluster in unique_clusters]
    ax.legend(
        lines + lines2 + cluster_lines,
        labels + labels2 + cluster_labels,
        loc="upper left",
    )

    return fig


def plot_3d_jaccard_indices(comparison_metrics, weeks):
    # Sort weeks so that "Drug_Trt" is last
    weeks_sorted = (
        sorted(week for week in weeks if week != "Drug_Trt") + ["Drug_Trt"]
        if "Drug_Trt" in weeks
        else weeks
    )

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection="3d")

    # Create a colormap
    cmap = plt.get_cmap("flare")

    # Initialize lists to store color and size data for the colorbar
    colors = []
    sizes = []

    # Iterate over each pair of sorted weeks
    for i, week_i in enumerate(weeks_sorted):
        for j, week_j in enumerate(weeks_sorted):
            if i != j:
                week_pair = (week_i, week_j)
                if week_pair in comparison_metrics:
                    jaccard_indices = comparison_metrics[week_pair]["Jaccard Indices"]
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
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=min(jaccard_indices), vmax=max(jaccard_indices)),
    )
    sm.set_array([])  # You have to set_array for newer versions of matplotlib
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label("Jaccard Index")

    # Create size legend for the sizes
    # Generate representative handles for the legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{s/100:.2f}",
            markerfacecolor="k",
            markersize=np.sqrt(s),
        )
        for s in np.unique(sizes)
    ]
    ax.legend(handles=handles, title="Jaccard Index Size", loc="upper left")

    # Set labels and title
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Weeks")
    ax.set_zlabel("Jaccard Index")
    ax.set_title("3D Plot of Jaccard Indices")

    # Set the tick labels for x and y axes to the sorted week names
    ax.set_xticks(range(len(weeks_sorted)))
    ax.set_xticklabels(weeks_sorted)
    ax.set_yticks(range(len(weeks_sorted)))
    ax.set_yticklabels(weeks_sorted)

    return fig


def plot_transition_counts_heatmap(comparison_metrics, week_pair):
    transition_counts = comparison_metrics[week_pair]["Transition Counts"]

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(transition_counts, annot=True, fmt="g")
    plt.title(f"Transition Counts: {week_pair[0]} vs {week_pair[1]}")
    plt.xlabel(f"Clusters in {week_pair[1]}")
    plt.ylabel(f"Clusters in {week_pair[0]}")
    plt.tight_layout()

    return fig


def plot_3d_weekly_transition_counts(comparison_metrics, weeks):
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection="3d")

    num_weeks = len(weeks)
    num_clusters = max(
        max(len(metrics["Transition Counts"][0]), len(metrics["Transition Counts"][1]))
        for metrics in comparison_metrics.values()
    )

    # Initialize z as zeros for each pair of weeks and clusters
    z = np.zeros((num_weeks, num_weeks, num_clusters), dtype=float)

    # Populate z with transition counts
    for (week_i, week_j), metrics in comparison_metrics.items():
        i = weeks.index(week_i)
        j = weeks.index(week_j)
        transition_counts = metrics["Transition Counts"]
        padded_transition_counts = np.pad(
            transition_counts,
            ((0, 0), (0, num_clusters - transition_counts.shape[1])),
            "constant",
        )
        z[i, j, :] = padded_transition_counts.sum(axis=0)

    # Create a mask for non-zero transition counts
    non_zero_mask = z > 0

    # Apply the mask to z and get the corresponding x, y indices
    x, y, z = np.nonzero(non_zero_mask)
    z_values = z[non_zero_mask]

    # Plot bars
    ax.bar3d(x, y, np.zeros_like(z_values), 0.4, 0.4, z_values, shade=True)

    # Set labels and title
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Clusters")
    ax.set_zlabel("Transition Count")
    ax.set_title("3D Transition Counts Between Weeks")

    # Set ticks for weeks and clusters
    ax.set_xticks(range(num_weeks))
    ax.set_xticklabels(weeks)
    ax.set_yticks(range(num_clusters))
    ax.set_yticklabels(range(num_clusters))

    return fig


def plot_normalized_values_by_group_and_timepoint(
    df, p_values_df, significance_level=0.05, specific_time_point=None, log_norm=False
):
    # Define a color mapping for groups
    group_colors = {
        "Sham": "#2ca02c",
        "ABX": "#1f77b4",
        "Treated": "#b662ff",
        "Injured": "#d42163",
    }

    df = df.merge(
        p_values_df, on=["Animal_ID", "Time_Point", "Group", "Motif"], how="left"
    )

    # If a specific time point is provided, filter the DataFrame to only include that time point
    if specific_time_point is not None:
        df = df[df["Time_Point"] == specific_time_point]

    if log_norm == "False":
        if specific_time_point is not None:
            grouped = (
                df.groupby(["Cluster", "Motif", "Group"])
                .agg({"Normalized_Value": "mean", "P_Value": "mean"})
                .reset_index()
            )
        else:
            # Group by the necessary columns and calculate the mean 'Normalized_Value' and mean 'P_Value'
            grouped = (
                df.groupby(["Time_Point", "Cluster", "Motif", "Group"])
                .agg({"Normalized_Value": "mean", "P_Value": "mean"})
                .reset_index()
            )
    else:
        if specific_time_point is not None:
            grouped = (
                df.groupby(["Cluster", "Motif", "Group"])
                .agg({"Log_Normalized_Value": "mean", "P_Value": "mean"})
                .reset_index()
            )
        else:
            # Group by the necessary columns and calculate the mean 'Normalized_Value' and mean 'P_Value'
            grouped = (
                df.groupby(["Time_Point", "Cluster", "Motif", "Group"])
                .agg({"Log_Normalized_Value": "mean", "P_Value": "mean"})
                .reset_index()
            )

    # Sort the DataFrame by 'Cluster' and 'Motif' for plotting
    grouped_sorted = grouped.sort_values(by=["Cluster", "Motif"])
    time_points = (
        grouped_sorted["Time_Point"].unique()
        if specific_time_point is None
        else [specific_time_point]
    )
    groups = grouped_sorted["Group"].unique()

    # Determine global y-axis limits
    if log_norm == "False":
        global_min = grouped_sorted["Normalized_Value"].min()
        global_max = grouped_sorted["Normalized_Value"].max() + 2
    else:
        global_min = grouped_sorted["Log_Normalized_Value"].min()
        global_max = grouped_sorted["Log_Normalized_Value"].max() + 2

    # Create subplots for each time point
    nrows = 1 if specific_time_point is not None else len(time_points)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 6 * nrows), sharex=True)  # type: ignore

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
            data_time_point = grouped_sorted[grouped_sorted["Time_Point"] == time_point]
        else:
            data_time_point = grouped_sorted

        # Plot each group as a separate line in the subplot
        for group_idx, group in enumerate(groups):
            data_group = data_time_point[data_time_point["Group"] == group]
            group_color = group_colors.get(
                group, "gray"
            )  # Default to 'gray' if group not in dictionary
            if not data_group.empty:
                if log_norm == "False":
                    line = sns.lineplot(
                        data=data_group,
                        x="Motif",
                        y="Normalized_Value",
                        ax=ax,
                        label=group,
                        marker="o",
                        color=group_color,
                    )
                else:
                    line = sns.lineplot(
                        data=data_group,
                        x="Motif",
                        y="Log_Normalized_Value",
                        ax=ax,
                        label=group,
                        marker="o",
                        color=group_color,
                    )
                # group_color = ax.lines[-1].get_color()
                group_color = line.get_lines()[-1].get_color()

            # Filter significant points for the current group and time point
            if specific_time_point is None:
                # significant_points = df[(df['Time_Point'] == time_point) & (df['Group'] == group) & (df['P_Value'] <= significance_level)]
                significant_points = data_group[
                    (data_group["Time_Point"] == time_point)
                    & (data_group["Group"] == group)
                    & (data_group["P_Value"] <= significance_level)
                ]
            else:
                # significant_points = df[(df['Group'] == group) & (df['P_Value'] <= significance_level)]
                significant_points = data_group[
                    (data_group["Group"] == group)
                    & (data_group["P_Value"] <= significance_level)
                ]

            # Initialize a dictionary to keep track of the number of significant points plotted for each motif
            motif_significant_count = defaultdict(int)

            # Plot the significant points directly above the motif they are marking as significant
            for _, point in significant_points.iterrows():
                motif = point["Motif"]
                # Increment the count for the motif
                motif_significant_count[motif] += 1
                # Calculate the x position based on the motif and the offset for the group
                x_value = motif  # + (group_idx - len(groups) / 2) * offset
                if log_norm == "False":
                    # Find the maximum normalized value for the current motif across all groups within the time point
                    max_normalized_value = data_time_point[
                        data_time_point["Motif"] == motif
                    ]["Normalized_Value"].max()
                else:
                    max_normalized_value = data_time_point[
                        data_time_point["Motif"] == motif
                    ]["Log_Normalized_Value"].max()

                # Calculate the y position based on the number of significant points already plotted for this motif
                # y_value = global_max + 0.1 * motif_significant_count[motif]
                y_value = (max_normalized_value + 0.5) + motif_significant_count[motif]

                # Ensure the y position is within the y-axis limits
                y_value = min(y_value, ax.get_ylim()[1])
                # Plot the significant marker
                ax.scatter(x_value, y_value, color=group_color, marker="*", s=50)

        # Set the title and labels
        ax.set_xlabel("Motif")
        if log_norm == "False":
            ax.set_ylabel("Normalized Value")
        else:
            ax.set_ylabel("Log Normalized Value")

        # Set the same y-axis limits for all subplots
        ax.set_ylim((global_min - 0.5), (global_max + 0.5))

        # Set motif labels on the x-axis
        motifs = data_time_point["Motif"].unique()
        ax.set_xticks(motifs)
        ax.set_xticklabels(motifs)

        # Move the legend to the right of the plot
        ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Place the Time Point text in the upper right corner of the subplot
        ax.text(
            0.95,
            0.95,
            time_point,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust the right edge to make room for the legend
    # Save the plot
    plt.savefig(f"Normalized Values by Time Point_Group_1.png", dpi=900)

    plt.show()


def plot_normalized_values_by_group_and_timepoint_2(
    df, p_values_df, significance_level=0.05, specific_time_point=None, log_norm=False
):
    # Define a color mapping for groups
    group_colors = {
        "Sham": "#2ca02c",
        "ABX": "#1f77b4",
        "Treated": "#b662ff",
        "Injured": "#d42163",
    }

    df = df.merge(
        p_values_df, on=["Animal_ID", "Time_Point", "Group", "Motif"], how="left"
    )

    # If a specific time point is provided, filter the DataFrame to only include that time point
    if specific_time_point is not None:
        df = df[df["Time_Point"] == specific_time_point]

    # Determine the value column to use based on log_norm flag
    value_column = "Log_Normalized_Value" if log_norm else "Normalized_Value"

    # Group by the necessary columns and calculate the mean value and mean 'P_Value'
    grouped = (
        df.groupby(["Cluster", "Motif", "Group", "Time_Point"])
        .agg({value_column: "mean", "P_Value": "mean"})
        .reset_index()
    )

    # Sort the DataFrame by 'Cluster' and 'Motif' for plotting
    grouped_sorted = grouped.sort_values(by=["Cluster", "Motif"])
    time_points = grouped_sorted["Time_Point"].unique()
    groups = grouped_sorted["Group"].unique()

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
        data_time_point = grouped_sorted[grouped_sorted["Time_Point"] == time_point]

        # Plot each group as a separate line in the subplot
        for group_idx, group in enumerate(groups):
            data_group = data_time_point[data_time_point["Group"] == group]
            group_color = group_colors.get(
                group, "gray"
            )  # Default to 'gray' if group not in dictionary

            sns.lineplot(
                data=data_group,
                x="Motif",
                y=value_column,
                ax=ax,
                label=group,
                marker="o",
                color=group_color,
            )

            # Filter significant points for the current group and time point
            significant_points = data_group[data_group["P_Value"] <= significance_level]

            # Plot the significant points directly above the motif they are marking as significant
            for _, point in significant_points.iterrows():
                motif = point["Motif"]

                cluster = point["Cluster"]
                # Find x position based on cluster and motif, adjusting for group offset
                cluster_motifs = data_time_point[data_time_point["Cluster"] == cluster][
                    "Motif"
                ].unique()
                motif_index = list(cluster_motifs).index(motif)

                # x_value = motif

                x_value = (
                    cluster_motifs[motif_index] + (group_idx - len(groups) / 2) * offset
                )

                max_value = data_time_point[data_time_point["Motif"] == motif][
                    value_column
                ].max()
                y_value = max_value + 0.1  # Slightly above the max value for visibility
                ax.scatter(x_value, y_value, color=group_color, marker="*", s=50)

        # Set the title and labels
        ax.set_title(f"Time Point: {time_point}")
        ax.set_xlabel("Motif")
        ax.set_ylabel(value_column)

        # Set the same y-axis limits for all subplots
        ax.set_ylim((global_min - 0.5), (global_max + 0.5))

        # Set motif labels on the x-axis
        motifs = data_time_point["Motif"].unique()
        ax.set_xticks(motifs)
        ax.set_xticklabels(motifs)

        # Move the legend to the right of the plot
        ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust the right edge to make room for the legend
    if specific_time_point:
        plt.suptitle(f"Normalized Values for Time Point: {specific_time_point}", y=1.02)
    else:
        plt.suptitle("Normalized Values by Time Point", y=1.02)

    plt.show()


def plot_boxplot_by_group_and_timepoint(
    df, p_values_df, significance_level=0.05, specific_time_point=None, log_norm=False
):
    # Define a color mapping for groups
    group_colors = {
        "Sham": "#2ca02c",
        "ABX": "#1f77b4",
        "Treated": "#b662ff",
        "Injured": "#d42163",
    }

    # Merge and filter data as needed
    df = df.merge(
        p_values_df, on=["Animal_ID", "Time_Point", "Group", "Motif"], how="left"
    )
    if specific_time_point is not None:
        df = df[df["Time_Point"] == specific_time_point]
    value_column = "Log_Normalized_Value" if log_norm else "Normalized_Value"

    # Group and sort data
    grouped = (
        df.groupby(["Cluster", "Motif", "Group", "Time_Point"])
        .agg({value_column: "mean", "P_Value": "mean"})
        .reset_index()
    )
    grouped_sorted = grouped.sort_values(by=["Cluster", "Motif"])

    # Create subplots
    time_points = grouped_sorted["Time_Point"].unique()
    nrows = len(time_points) if specific_time_point is None else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 6 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    # Plot boxplots
    for ax, time_point in zip(axes, time_points):
        data_time_point = grouped_sorted[grouped_sorted["Time_Point"] == time_point]
        sns.boxplot(
            data=data_time_point,
            x="Motif",
            y=value_column,
            hue="Group",
            ax=ax,
            palette=group_colors,
        )

        # Optionally, add significant markers
        for group in data_time_point["Group"].unique():
            data_group = data_time_point[data_time_point["Group"] == group]
            significant_points = data_group[data_group["P_Value"] <= significance_level]
            for _, point in significant_points.iterrows():
                x_value = point["Motif"]
                y_value = point[value_column]
                ax.scatter(
                    x=x_value, y=y_value, color=group_colors[group], marker="*", s=50
                )

        ax.set_title(f"Time Point: {time_point}")
        ax.set_xlabel("Motif")
        ax.set_ylabel(value_column)
        ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    if specific_time_point:
        plt.suptitle(
            f"Boxplot of Normalized Values for Time Point: {specific_time_point}",
            y=1.02,
        )
    else:
        plt.suptitle("Boxplot of Normalized Values by Time Point", y=1.02)

    plt.show()


def plot_normalized_values_by_group_and_timepoint_3(
    df,
    p_values_df,
    significance_level=0.05,
    specific_time_point=None,
    log_norm=False,
    c_type="Group",
    adj_Pvals=False,
):
    # Define a color mapping for groups
    group_colors = {
        "Sham": "#2ca02c",
        "ABX": "#1f77b4",
        "Treated": "#b662ff",
        "Injured": "#d42163",
    }

    df_cols = df.columns
    p_values_df_cols = p_values_df.columns

    shared_cols = set(df_cols).intersection(p_values_df_cols)

    # Merge based on c_type
    merge_on = ["Animal_ID", "Time_Point", "Motif", c_type]
    df = df.merge(p_values_df, on=merge_on, how="left")
    # Check for duplicate columns where all values are the same

    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Filter by specific time point if provided
    if specific_time_point:
        df = df[df["Time_Point"] == specific_time_point]

    # Determine the value column to use based on log_norm flag
    value_column = "Log_Normalized_Value" if log_norm else "Normalized_Value"

    # Determine P_val_col and group_by based on adj_Pvals and c_type
    P_val_col = "Adjusted_P_Value_" + c_type if adj_Pvals else "P_Value"
    group_by = ["Cluster", "Motif", c_type, "Time_Point"]

    print(df.columns)

    if value_column in df.columns and P_val_col in df.columns:
        print(f"Value column: {value_column}")
        print(f"P_val_col: {P_val_col}")
        # Group by the necessary columns and calculate the mean value and mean 'P_Value'
        grouped = (
            df.groupby(group_by)
            .agg({value_column: "mean", P_val_col: "mean"})
            .reset_index()
        )
    else:
        print(f"Error: {value_column} or {P_val_col} not in df.columns")
        return

    # Sort the DataFrame by 'Cluster' and 'Motif' for plotting
    grouped_sorted = grouped.sort_values(by=["Cluster", "Motif"])
    time_points = grouped_sorted["Time_Point"].unique()
    groups = grouped_sorted[c_type].unique()

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
    grouped_sorted.loc[grouped_sorted[c_type] == "Sham", value_column] += epsilon

    grouped_sorted = grouped_sorted.copy()

    cluster_motif_combinations = grouped_sorted[["Cluster", "Motif"]].drop_duplicates()
    cluster_motif_combinations["x_value"] = range(len(cluster_motif_combinations))
    mapping = cluster_motif_combinations.set_index(["Cluster", "Motif"])[
        "x_value"
    ].to_dict()

    # Create a new column 'x_value' using the mapping dictionary
    grouped_sorted["x_value"] = grouped_sorted.apply(
        lambda row: mapping[(row["Cluster"], row["Motif"])], axis=1
    )

    # Plot each time point in a separate subplot
    for ax, time_point in zip(axes, time_points):
        offset_fraction = 0.05  # Adjust this value as needed to prevent overlap
        offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * offset_fraction / len(groups)

        # Filter the DataFrame for the current time point
        data_time_point = grouped_sorted[grouped_sorted["Time_Point"] == time_point]

        # Plot each group as a separate line in the subplot
        for group_idx, group in enumerate(groups):
            data_group = data_time_point[data_time_point[c_type] == group]
            if data_group.empty:
                continue
            # data_group['x_value'] = data_group.apply(lambda row: mapping[(row['Cluster'], row['Motif'])], axis=1)
            group_color = group_colors.get(
                group, "gray"
            )  # Default to 'gray' if group not in dictionary

            # sns.lineplot(data=data_group, x='Motif', y=value_column, ax=ax, label=group, marker='o', color=group_color)
            sns.lineplot(
                data=data_group,
                x="x_value",
                y=value_column,
                ax=ax,
                label=group,
                marker="o",
                color=group_color,
            )

            # Filter significant points for the current group and time point
            significant_points = data_group[data_group[P_val_col] <= significance_level]
            motif_significant_count = defaultdict(int)
            # Plot the significant points directly above the motif they are marking as significant
            for _, point in significant_points.iterrows():
                motif = point["Motif"]
                motif_significant_count[motif] += 1
                x_value = motif

                max_value = data_time_point[data_time_point["Motif"] == motif][
                    value_column
                ].max()
                y_value = (max_value + 0.1) + (motif_significant_count[motif])
                ax.scatter(x_value, y_value, color=group_color, marker="*", s=50)

        # Set the title and labels
        ax.set_title(f"Time Point: {time_point}")
        ax.set_xlabel("Motif")
        ax.set_ylabel(value_column)

        # Set the same y-axis limits for all subplots
        ax.set_ylim((global_min - 0.5), (global_max + 0.5))

        # Set motif labels on the x-axis
        motifs = data_time_point["Motif"].unique()
        # Update x-axis ticks and labels to reflect cluster-motif ordering
        ax.set_xticks(cluster_motif_combinations["x_value"])
        ax.set_xticklabels(
            [
                f"{row['Cluster']}_{row['Motif']}"
                for _, row in cluster_motif_combinations.iterrows()
            ],
            rotation=90,
        )

        # ax.set_xticks(motifs)
        # ax.set_xticklabels(motifs)

        # Move the legend to the right of the plot
        ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust the right edge to make room for the legend
    if specific_time_point:
        plt.suptitle(f"Normalized Values for Time Point: {specific_time_point}", y=1.02)
    else:
        plt.suptitle("Normalized Values by Time Point", y=1.02)

    # Save the plot
    plt.savefig(f"Normalized Values by Time Point_{c_type}.png", dpi=900)

    plt.show()


def plot_motifs_all_time_points(df, comparison="Group", significance_level=0.05):
    group_colors = {
        "Sham": "#2ca02c",
        "ABX": "#1f77b4",
        "Treated": "#b662ff",
        "Injured": "#d42163",
    }

    time_points = df["Time_Point"].unique()
    nrows = len(time_points)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(15, 5 * nrows), sharex=True)

    # If there's only one time point, axes will not be an array, so we wrap it in a list
    if nrows == 1:
        axes = [axes]

    global_min, global_max = float("inf"), float("-inf")

    # Initialize an empty list to collect legend handles and labels
    legend_handles, legend_labels = [], []

    for ax, time_point in zip(axes, time_points):
        df_time = df[df["Time_Point"] == time_point]

        if comparison == "Group":
            mean_values = (
                df_time.groupby(["Group", "Motif", "Cluster"])["Log_Normalized_Value"]
                .mean()
                .reset_index()
            )
            p_value_column = "P_Value_Group"
        elif comparison == "Treatment_State":
            mean_values = (
                df_time.groupby(["Treatment_State", "Motif", "Cluster"])[
                    "Log_Normalized_Value"
                ]
                .mean()
                .reset_index()
            )
            p_value_column = "P_Value_State"

        mean_values_sorted = mean_values.sort_values(by=["Cluster", "Motif"])

        unique_clusters = mean_values_sorted["Cluster"].unique()
        motif_order = {}
        counter = 0

        for cluster in unique_clusters:
            cluster_motifs = mean_values_sorted[
                mean_values_sorted["Cluster"] == cluster
            ]["Motif"].unique()
            for motif in cluster_motifs:
                motif_order[motif] = counter
                counter += 1

        # Initialize a dictionary to keep track of the maximum y-value for each x-position
        max_y_by_x_position = {}

        for group in mean_values_sorted[comparison].unique():
            group_data = mean_values_sorted[mean_values_sorted[comparison] == group]
            group_color = group_colors[group]
            x_values = group_data["Motif"].map(motif_order)
            y_values = group_data["Log_Normalized_Value"]
            line = ax.plot(
                x_values, y_values, label=group, marker="o", color=group_color
            )

            # line_color = line[0].get_color()
            # Update the dictionary with the maximum y-values
            for x, y in zip(x_values, y_values):
                max_y_by_x_position[x] = max(
                    max_y_by_x_position.get(x, float("-inf")), y
                )

        # Initialize a dictionary to keep track of the number of asterisks plotted for each x-position
        asterisk_count = {}

        for group in mean_values_sorted[comparison].unique():
            group_data = mean_values_sorted[mean_values_sorted[comparison] == group]
            line_color = group_colors[group]
            for x, y, motif in zip(x_values, y_values, group_data["Motif"]):
                if (
                    df_time[
                        (df_time[comparison] == group) & (df_time["Motif"] == motif)
                    ][p_value_column].iloc[0]
                    <= significance_level
                ):
                    asterisk_count[x] = asterisk_count.get(x, 0) + 1
                    new_y = max_y_by_x_position[x] + 0.1 * (
                        asterisk_count[x] * 2
                    )  # Add a small offset above the max y-value
                    max_y_by_x_position[x] = new_y
                    global_max = max(global_max, new_y)

                    ax.text(
                        x,
                        new_y,
                        "*",
                        color=line_color,
                        fontsize=14,
                        ha="center",
                        va="bottom",
                    )

        # Update global_max based on the highest asterisk position
        global_max = max(global_max, max(max_y_by_x_position.values()))

        # Add clusters background color
        clusters = df_time["Cluster"].unique()
        colors = [
            "#ff9999",
            "#ffcc99",
            "#e6ff99",
            "#b3ffcc",
            "#99ffff",
            "#99ccff",
            "#cc99ff",
            "#ff99ff",
            "#ff6666",
            "#ffbb66",
            "#dfff66",
            "#66ff99",
        ]
        for i, cluster in enumerate(unique_clusters):
            cluster_motifs = mean_values_sorted[
                mean_values_sorted["Cluster"] == cluster
            ]["Motif"].unique()
            x_positions = [motif_order[motif] for motif in cluster_motifs]
            ax.axvspan(
                min(x_positions) - 0.5,
                max(x_positions) + 0.5,
                color=colors[i % len(colors)],
                alpha=0.45,
            )

            # Update global min and max for y-axis limits
            global_min = min(global_min, y_values.min())
            global_max = max(global_max, y_values.max())

        # Collect handles and labels for the legend
        for line in ax.get_lines():
            label = line.get_label()
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        # ax.set_title(f'{time_point}')
        ax.text(
            0.95,
            0.95,
            time_point,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    # Set common labels
    fig.text(0.5, 0.01, "Motif #", ha="center", va="center")
    fig.text(
        0.01, 0.5, "Log Normalized Value", ha="center", va="center", rotation="vertical"
    )

    # Set the same y-axis limits for all subplots with a margin
    y_margin = (global_max - global_min) * 0.10  # 10% margin
    for ax in axes:
        ax.set_ylim(global_min - y_margin, global_max + y_margin)

    # Set the x-axis ticks and labels based on motif order
    for ax in axes:
        ax.set_xticks(list(motif_order.values()))
        ax.set_xticklabels(list(motif_order.keys()))

    # Set the plot title
    # Customizable plot title position
    title_x = 0.5  # Horizontal position, 0.5 is center
    title_y = 0.995  # Vertical position, 0.98 is near the top
    if comparison == "Group":
        fig.suptitle("Group Motif Usage", fontsize=16, x=title_x, y=title_y)
    if comparison == "Treatment_State":
        fig.suptitle("Treatment State Motif Usage", fontsize=16, x=title_x, y=title_y)

    # Determine the maximum label length
    max_label_length = max(len(label) for label in legend_labels)

    # first_legend = fig.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

    # Create the first legend for groups or treatment states
    first_legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(1, 0.98),
        bbox_transform=plt.gcf().transFigure,
        handlelength=2,  # Adjust this as needed
        handletextpad=0.5,  # Adjust this as needed
        borderaxespad=0.5,  # Adjust this as needed
        title="Groups",
    )

    plt.gca().add_artist(first_legend)  # Add the first legend back to the current Axes

    # Define custom labels for each cluster, if desired
    # custom_labels = ['A', 'B', 'C', 'D']  # Example custom labels

    # Create custom legend handles (patches) for the cluster colors
    # Using just numbers as labels
    cluster_patches = [
        mpatches.Patch(color=color, label=f"{i+1}")
        for i, color in enumerate(colors[: len(unique_clusters)])
    ]

    # Or using custom labels
    # cluster_patches = [mpatches.Patch(color=color, label=custom_labels[i]) for i, color in enumerate(colors[:len(unique_clusters)])]

    # Create the second legend for clusters and place it below the first legend
    second_legend = fig.legend(
        handles=cluster_patches,  # The legend handles (patches) for the clusters
        loc="upper right",  # Location of the legend in the figure
        bbox_to_anchor=(0.9875, 0.90),  # The position of the legend's bounding box
        bbox_transform=plt.gcf().transFigure,  # The transformation used for the bounding box
        handlelength=2,  # The length of the legend handles, matching the first legend
        handletextpad=0.5,  # The pad between the legend handle and text, matching the first legend
        borderaxespad=0.25,  # The pad between the legend border and axes, matching the first legend
        title="Clusters",  # The title of the legend
    )

    # Create the second legend for clusters and place it below the first legend

    plt.tight_layout()

    plt.subplots_adjust(
        bottom=0.035,  # Increase the bottom padding to 0.2 (20% of the figure height)
        top=0.975,  # Adjust the top padding
        left=0.04,  # Adjust the left padding
        right=0.90,  # Adjust the right padding
        hspace=0.0,  # Adjust the vertical space between subplots
        #     wspace=0.0 # Adjust the horizontal space between subplots
    )

    plt.show()


def plot_motif_frequency(df, time_point, motifs=None):
    # Filter the DataFrame for the specified time point
    df_time = df[df["time_point"] == time_point]

    # If specific motifs are provided, filter the DataFrame to include only these motifs
    if motifs is not None:
        df_time = df_time[df_time["motif"].isin(motifs)]

    # Calculate the frequency of each motif in each group
    motif_freq = (
        df_time.groupby(["group", "motif"]).size().reset_index(name="frequency")
    )

    # Print the motif_freq DataFrame for debugging
    print(motif_freq)

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="motif", y="frequency", hue="group", data=motif_freq)
    plt.title(f"Motif Frequency at {time_point}")
    plt.show()


def plot_heatmap_by_group(df, clustered_motifs):
    # Create a mapping from motif to cluster order
    motif_to_cluster = {
        motif: cluster
        for cluster, motifs in clustered_motifs.items()
        for motif in motifs
    }

    # Work with a copy to avoid SettingWithCopyWarning
    df_filtered = df[~df["Time_Point"].isin(["Baseline_1", "Baseline_2"])].copy()
    df_filtered["Cluster_Order"] = df_filtered["Motif"].map(motif_to_cluster)

    # Group by the necessary columns and calculate the mean 'Normalized_Value'
    grouped = (
        df_filtered.groupby(["Time_Point", "Cluster_Order", "Motif", "Group"])[
            "Normalized_Value"
        ]
        .mean()
        .reset_index()
    )

    # Sort the DataFrame by 'Cluster_Order' and 'Motif' for plotting
    grouped_sorted = grouped.sort_values(by=["Cluster_Order", "Motif"])

    # Get unique groups and time points, excluding baseline and ensuring 'Drug_Trt' is last
    groups = grouped_sorted["Group"].unique()
    time_points = [
        tp for tp in grouped_sorted["Time_Point"].unique() if tp != "Drug_Trt"
    ] + ["Drug_Trt"]

    # Determine the global color scale limits
    global_min = grouped_sorted["Normalized_Value"].min()
    global_max = grouped_sorted["Normalized_Value"].max()

    # Create subplots for each group
    fig, axes = plt.subplots(
        nrows=1, ncols=len(groups), figsize=(5 * len(groups), 10), sharey=True
    )

    if len(groups) == 1:
        axes = [axes]

    # Plot a heatmap for each group
    for ax, group in zip(axes, groups):
        # Use pivot_table to handle duplicate entries
        heatmap_data = grouped_sorted[grouped_sorted["Group"] == group].pivot_table(
            # index='Cluster_Order',
            index="Motif",
            columns="Time_Point",
            values="Normalized_Value",
            aggfunc="mean",
        )

        # Sort the index by 'Cluster_Order' to maintain the cluster grouping
        heatmap_data = heatmap_data.sort_index()

        # Reindex the DataFrame to ensure 'Drug_Trt' is the last column
        heatmap_data = heatmap_data.reindex(time_points, axis=1)

        # Plot the heatmap with a consistent color scale
        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap="flare",
            cbar_kws={"label": "Normalized Value"},
            vmin=global_min,
            vmax=global_max,
        )

        # Set the title and adjust the axis
        ax.set_title(f"Group: {group}")
        ax.set_xlabel("Time Point")
        ax.set_ylabel("Motif")

    # Adjust the layout
    plt.tight_layout()
    plt.show()


def plot_motifs4(df, comparison="Group", significance_level=0.05):
    time_points = df["Time_Point"].unique()
    nrows = len(time_points)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(15, 5 * nrows), sharex=True)

    # If there's only one time point, axes will not be an array, so we wrap it in a list
    if nrows == 1:
        axes = [axes]

    global_min, global_max = float("inf"), float("-inf")

    for ax, time_point in zip(axes, time_points):
        df_time = df[df["Time_Point"] == time_point]

        if comparison == "Group":
            mean_values = (
                df_time.groupby(["Group", "Motif", "Cluster"])["Log_Normalized_Value"]
                .mean()
                .reset_index()
            )
            p_value_column = "P_Value_Group"
        elif comparison == "Treatment_State":
            mean_values = (
                df_time.groupby(["Treatment_State", "Motif", "Cluster"])[
                    "Log_Normalized_Value"
                ]
                .mean()
                .reset_index()
            )
            p_value_column = "P_Value_State"

        mean_values_sorted = mean_values.sort_values(by=["Cluster", "Motif"])

        unique_clusters = mean_values_sorted["Cluster"].unique()
        motif_order = {}
        counter = 0
        for cluster in unique_clusters:
            cluster_motifs = mean_values_sorted[
                mean_values_sorted["Cluster"] == cluster
            ]["Motif"].unique()
            for motif in cluster_motifs:
                motif_order[motif] = counter
                counter += 1

        for group in mean_values_sorted[comparison].unique():
            group_data = mean_values_sorted[mean_values_sorted[comparison] == group]

            x_values = group_data["Motif"].map(motif_order)
            y_values = group_data["Log_Normalized_Value"]
            line = ax.plot(x_values, y_values, label=group, marker="o")

            line_color = line[0].get_color()

            # Check for significance and add asterisks
            for x, y, motif in zip(x_values, y_values, group_data["Motif"]):
                if (
                    df_time[
                        (df_time[comparison] == group) & (df_time["Motif"] == motif)
                    ][p_value_column].iloc[0]
                    <= significance_level
                ):
                    ax.text(
                        x,
                        y,
                        "*",
                        color=line_color,
                        fontsize=12,
                        ha="center",
                        va="bottom",
                    )

            # Update global min and max for y-axis limits
            global_min = min(global_min, y_values.min())
            global_max = max(global_max, y_values.max())

        ax.set_title(f"{time_point}")
        ax.text(
            0.95,
            0.95,
            time_point,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    # Set common labels
    fig.text(0.5, 0.04, "Motif #", ha="center", va="center")
    fig.text(
        0.06, 0.5, "Log Normalized Value", ha="center", va="center", rotation="vertical"
    )

    # Set the same y-axis limits for all subplots with a margin
    y_margin = (global_max - global_min) * 0.05  # 5% margin
    for ax in axes:
        ax.set_ylim(global_min - y_margin, global_max + y_margin)

    # Set the x-axis ticks and labels based on motif order
    for ax in axes:
        ax.set_xticks(list(motif_order.values()))
        ax.set_xticklabels(list(motif_order.keys()))

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_motifs3(
    df, time_point, comparison="Group", significance_level=0.05, clustering="kmeans-30"
):
    # Filter the data for the given Time_Point
    df_time = df[df["Time_Point"] == time_point]

    if comparison == "Group":
        # Calculate the mean 'Log_Normalized_Value' for each group and motif
        mean_values = (
            df_time.groupby(["Group", "Motif", "Cluster"])["Log_Normalized_Value"]
            .mean()
            .reset_index()
        )
        p_value_column = "P_Value_Group"
    elif comparison == "Treatment_State":
        mean_values = (
            df_time.groupby(["Treatment_State", "Motif", "Cluster"])[
                "Log_Normalized_Value"
            ]
            .mean()
            .reset_index()
        )
        p_value_column = "P_Value_State"

    # Sort the DataFrame by 'Cluster' and 'Motif' for plotting
    mean_values_sorted = mean_values.sort_values(by=["Cluster", "Motif"])

    # Create a mapping for x-axis positions
    unique_clusters = mean_values_sorted["Cluster"].unique()
    motif_order = {}
    counter = 0
    for cluster in unique_clusters:
        cluster_motifs = mean_values_sorted[mean_values_sorted["Cluster"] == cluster][
            "Motif"
        ].unique()
        for motif in cluster_motifs:
            motif_order[motif] = counter
            counter += 1

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot data for each group or treatment state
    for group in mean_values_sorted[comparison].unique():
        group_data = mean_values_sorted[mean_values_sorted[comparison] == group]

        # Use the motif_order mapping for the x-axis positions
        x_values = group_data["Motif"].map(motif_order)
        y_values = group_data["Log_Normalized_Value"]
        # ax.plot(x_values, y_values, label=group, marker='o')
        line = ax.plot(x_values, y_values, label=group, marker="o")

        line_color = line[0].get_color()

        # Check for significance and add asterisks
        for x, y, motif in zip(x_values, y_values, group_data["Motif"]):
            if (
                df_time[(df_time[comparison] == group) & (df_time["Motif"] == motif)][
                    p_value_column
                ].iloc[0]
                <= significance_level
            ):
                ax.text(
                    x, y, "*", color=line_color, fontsize=12, ha="center", va="bottom"
                )

    # Add clusters background color
    clusters = df_time["Cluster"].unique()
    colors = ["pink", "orange", "lightgreen", "lightblue"] * (
        len(clusters) // 4 + 1
    )  # repeat colors if not enough
    for i, cluster in enumerate(unique_clusters):
        cluster_motifs = mean_values_sorted[mean_values_sorted["Cluster"] == cluster][
            "Motif"
        ].unique()
        x_positions = [motif_order[motif] for motif in cluster_motifs]
        ax.axvspan(
            min(x_positions) - 0.5,
            max(x_positions) + 0.5,
            color=colors[i % len(colors)],
            alpha=0.3,
        )

    # Customizing the plot
    ax.set_xlabel("Motif #")
    ax.set_ylabel("Log Normalized Value")
    ax.set_title(f"Mean Log Normalized Values by {comparison} at {time_point}")
    ax.set_xticks(list(motif_order.values()))
    ax.set_xticklabels(list(motif_order.keys()))
    ax.legend()

    plt.savefig(f"{clustering}-{comparison}_{time_point}.svg", format="svg")

    # plt.show()


def plot_motifs2(df, time_point, comparison="Group"):
    # Filter the data for the given Time_Point
    df_time = df[df["Time_Point"] == time_point]

    if comparison == "Group":
        # Calculate the mean 'Log_Normalized_Value' for each group and motif
        mean_values = (
            df_time.groupby(["Group", "Motif", "Cluster"])["Log_Normalized_Value"]
            .mean()
            .reset_index()
        )
    elif comparison == "Treatment_State":
        mean_values = (
            df_time.groupby(["Treatment_State", "Motif", "Cluster"])[
                "Log_Normalized_Value"
            ]
            .mean()
            .reset_index()
        )

    # Sort the DataFrame by 'Cluster' and 'Motif' for plotting
    mean_values_sorted = mean_values.sort_values(by=["Cluster", "Motif"])

    # Create a mapping for x-axis positions
    unique_clusters = mean_values_sorted["Cluster"].unique()
    motif_order = {}
    counter = 0
    for cluster in unique_clusters:
        cluster_motifs = mean_values_sorted[mean_values_sorted["Cluster"] == cluster][
            "Motif"
        ].unique()
        for motif in cluster_motifs:
            motif_order[motif] = counter
            counter += 1

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(15, 5))

    if comparison == "Group":
        groups = mean_values_sorted["Group"].unique()
    elif comparison == "Treatment_State":
        groups = mean_values_sorted["Treatment_State"].unique()

    # Plot data for each group
    for group in groups:
        if comparison == "Group":
            group_data = mean_values_sorted[mean_values_sorted["Group"] == group]
        elif comparison == "Treatment_State":
            group_data = mean_values_sorted[
                mean_values_sorted["Treatment_State"] == group
            ]

        # Use the motif_order mapping for the x-axis positions
        x_values = group_data["Motif"].map(motif_order)
        ax.plot(x_values, group_data["Log_Normalized_Value"], label=group, marker="o")

    clusters = df_time["Cluster"].unique()
    colors = ["pink", "orange", "lightgreen", "lightblue"] * (
        len(clusters) // 4 + 1
    )  # repeat colors if not enough
    # Add clusters background color
    for i, cluster in enumerate(unique_clusters):
        cluster_motifs = mean_values_sorted[mean_values_sorted["Cluster"] == cluster][
            "Motif"
        ].unique()
        x_positions = [motif_order[motif] for motif in cluster_motifs]
        ax.axvspan(
            min(x_positions) - 0.5,
            max(x_positions) + 0.5,
            color=colors[i % len(colors)],
            alpha=0.3,
        )

    # Customizing the plot
    ax.set_xlabel("Motif #")
    ax.set_ylabel("Log Normalized Value")
    if comparison == "Group":
        ax.set_title(f"Mean Log Normalized Values by Group at {time_point}")
    elif comparison == "Treatment_State":
        ax.set_title(f"Mean Log Normalized Values by Treatment State at {time_point}")
    ax.set_xticks(list(motif_order.values()))
    ax.set_xticklabels(list(motif_order.keys()))
    ax.legend()

    plt.show()


def plot_motifs(df, time_point):
    # Filter the data for the given Time_Point
    df_time = df[df["Time_Point"] == time_point]

    # Calculate the mean 'Log_Normalized_Value' for each group and motif
    mean_values = (
        df_time.groupby(["Group", "Motif"])["Log_Normalized_Value"].mean().reset_index()
    )

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(15, 5))

    # Get unique groups
    groups = mean_values["Group"].unique()

    # Plot data for each group
    for group in groups:
        group_data = mean_values[mean_values["Group"] == group].sort_values(
            by="Motif"
        )  # Sort by Motif

        # Plot the mean values for each group
        ax.plot(
            group_data["Motif"],
            group_data["Log_Normalized_Value"],
            label=group,
            marker="o",
        )

    # Add clusters background color
    clusters = df_time["Cluster"].unique()
    colors = ["pink", "orange", "lightgreen", "lightblue"] * (
        len(clusters) // 4 + 1
    )  # repeat colors if not enough
    for i, cluster in enumerate(clusters):
        cluster_data = df_time[df_time["Cluster"] == cluster]
        ax.axvspan(
            cluster_data["Motif"].min(),
            cluster_data["Motif"].max(),
            color=colors[i],
            alpha=0.3,
        )

    # Customizing the plot
    ax.set_xlabel("Motif #")
    ax.set_ylabel("Log Normalized Value")
    ax.set_title(f"Mean Log Normalized Values by Group at {time_point}")
    ax.legend()

    plt.show()


def plot_boxplot_by_group_and_timepoint_2(
    df, significance_level=0.05, specific_time_point=None, log_norm=False
):
    # Define a color mapping for groups
    group_colors = {
        "Sham": "#2ca02c",
        "ABX": "#1f77b4",
        "Treated": "#b662ff",
        "Injured": "#d42163",
    }

    # If a specific time point is provided, filter the DataFrame to only include that time point
    if specific_time_point is not None:
        df = df[df["Time_Point"] == specific_time_point]

    # Determine the value column to use based on log_norm flag
    value_column = "Log_Normalized_Value" if log_norm else "Normalized_Value"

    # Create a new DataFrame for plotting
    plot_df = df[["Motif", "Group", "Time_Point", value_column]]

    # If a specific time point is provided, adjust nrows
    time_points = plot_df["Time_Point"].unique()
    nrows = len(time_points) if specific_time_point is None else 1

    # Create subplots for each time point
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 6 * nrows), sharex=True)

    # If there's only one time point, axes will not be an array, so we wrap it in a list
    if nrows == 1:
        axes = [axes]

    # Plot each time point in a separate subplot
    for ax, time_point in zip(axes, time_points):
        # Filter the DataFrame for the current time point
        data_time_point = plot_df[plot_df["Time_Point"] == time_point]

        # Create the boxplot
        sns.boxplot(
            data=data_time_point,
            x="Motif",
            y=value_column,
            hue="Group",
            ax=ax,
            palette=group_colors,
        )

        # Set the title and labels
        ax.set_title(f"Time Point: {time_point}")
        ax.set_xlabel("Motif")
        ax.set_ylabel(value_column)

        # Move the legend to the right of the plot
        ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust the right edge to make room for the legend
    if specific_time_point:
        plt.suptitle(
            f"Boxplot of {value_column} for Time Point: {specific_time_point}", y=1.02
        )
    else:
        plt.suptitle(f"Boxplot of {value_column} by Time Point", y=1.02)

    plt.show()


if __name__ == "__main__":
    import vame

    df = pd.read_csv(
        "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/df_long_logNorm_base.csv"
    )
    p_values_df = pd.read_csv(
        "D:/Users/tywin/VAME/results/aggregated_analysis/hmm-40-650/all_data/stats/pVals_longNorm_base_Group.csv"
    )

    plot_normalized_values_by_group_and_timepoint(
        df,
        p_values_df,
        significance_level=0.05,
        specific_time_point=None,
        log_norm=True,
    )
