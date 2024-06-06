import csv
import json
import os
import re
import sys
from calendar import c
from importlib import reload
from sqlite3 import Time
from turtle import title
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# import pymer4 as pym
import scipy.stats
import seaborn as sns
from dask.base import compute
from dask.delayed import delayed
from fitter import Fitter
from icecream import ic
from loguru import logger
from regex import F
from rich import pretty

pretty.install()
from rich.console import Console
from sklearn.ensemble import IsolationForest as iforest
from sktree.ensemble import ExtendedIsolationForest as eif
from statsmodels.stats.diagnostic import het_white

# make sure python knows vame is a folder the code can be imported from
sys.path.append("/Users/adanredwine/VAME/")
import vame
import vame.custom.ALR_analysis as ana
import vame.custom.ALR_helperFunctions as AlHf
import vame.custom.ALR_plottingFunctions as AlPf
import vame.custom.ALR_statsFunctions as AlSt
from vame.custom import helperFunctions as hf
from vame.custom.ALR_latent_vector_cluster_functions import (
    calculate_mean_latent_vector_for_motifs,
    create_tsne_projection,
    create_umap_projection,
)
from vame.util.auxiliary import read_config

logger.add(
    "logs/debug_ALR_ReportPlotting.log",
    colorize=True,
    format="{time} {level} {message}",
    level="DEBUG",
)

logger.debug("New Run of ALR_ReportPlotting, nice!")

console = Console()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def check_normality_tps_checks(df, check="Motif", alpha=0.05):
    checks = df[check].unique()
    normality_results = []

    tps = df["Time_Point"].unique()

    for tp in tps:
        for k in checks:
            check_df = df[df[check] == k]
            console.print(check_df["Value"])
            # Drop any rows where "Value" is NaN
            check_df = check_df.dropna(subset=["Value"])
            shapiro_results = scipy.stats.shapiro(check_df["Value"])
            print(shapiro_results)
            normality_results.append(
                {
                    check: k,
                    "Behavior": check_df["Predominant_Behavior"].unique()[0],
                    #"Treatment_State": state,
                    "Time_Point": tp,
                    "W": shapiro_results.statistic,
                    "p": shapiro_results.pvalue,
                    "Is_Normal": shapiro_results.pvalue >= alpha,
                }
            )

    return normality_results


def delayed_fit(data):
    # f = Fitter(data, timeout=120)
    f = Fitter(
        data,
        distributions=[
            "norm",
            "expon",
            "exponpow",
            "uniform",
            "lognorm",
            "nbinom",
        ],
        timeout=240,
    )
    f.fit()
    best_fit = f.get_best()
    return best_fit


def find_data_bestFit(df, check="Motif"):
    checks = df[check].unique()

    tps = df["Time_Point"].unique()

    delayed_results = []

    for tp in tps:
        for k in checks:
            check_df = df[df[check] == k]
            check_df = check_df.dropna(subset=["Value"])
            result = delayed(delayed_fit)(check_df["Value"])
            delayed_results.append(
                (k, check_df["Predominant_Behavior"].unique()[0], tp, result)
            )

    computed_results = compute(*delayed_results)

    fit_results = []
    for (k, behavior, tp, result), fit in zip(delayed_results, computed_results):
        fit_results.append(
            {
                check: k,
                "Behavior": behavior,
                "Time_Point": tp,
                "Fit": fit,
            }
        )

    return fit_results


def plot_histogram_and_shapiro(
    df, column="Value", facet_col="Time_Point", facet_row="Treatment_State"
):
    hist_df = px.histogram(df, x=column, title=f"Histogram of {column} Column")
    shap_res = scipy.stats.shapiro(df[column])
    hist_df.add_annotation(
        x=0.5,
        y=0.95,
        text="Shapiro-Wilk Test Results: Stat: "
        + str(round(shap_res[0], 4))
        + "\n"
        + " p value: "
        + str(round(shap_res[1], 4)),
        showarrow=False,
        font=dict(size=14),
        align="center",
        xref="paper",
        yref="paper",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        opacity=0.5,
    )
    return hist_df


import scipy.stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def calculate_and_adjust_p_values(df, motif_or_cluster):
    pVals_df = AlSt.calculate_p_values_vs_sham_State(
        df,
        motif_or_cluster=motif_or_cluster,
        value_column="Value",
        log_comp=False,
        normalized=False,
    )
    reject, p_adjusted, _, _ = multipletests(
        pVals_df["P_Value"], m_or_c="fdr_bh", alpha=0.05
    )

    pVals_df["P_Value_adj"] = p_adjusted
    pVals_df.drop(columns=["Animal_ID"], inplace=True)

    return pVals_df


from functools import reduce


def add_state_means_to_df(df, group_by_column):
    state_means = {}
    for state in df["Treatment_State"].unique():
        state_name = state
        state_means[state] = df.groupby(
            ["Treatment_State", "Time_Point", group_by_column]
        )["Value"].mean()[state]
        state_means[state] = pd.DataFrame(state_means[state]).reset_index()
        state_means[state].columns = [
            "Time_Point",
            group_by_column,
            state_name + "_Mean",
        ]

    dfs = list(state_means.values())
    merged_df = reduce(
        lambda left, right: pd.merge(
            left, right, on=["Time_Point", group_by_column], how="outer"
        ),
        dfs,
    )

    return merged_df


def make_rvf_plot(name,  result, het_white_res, m_or_c="M", motif=None, cluster=None):
    G = None
    if m_or_c == "Motif":
        G = "motif"
        motif_or_cluster = motif
    elif m_or_c == "Cluster":
        G = "cluster"
        motif_or_cluster = cluster
    fig = plt.figure(figsize=(16, 9))
    ax = sns.scatterplot(y=result.resid, x=result.fittedvalues)
    ax.set_title("RVF Plot")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.text(
        0.01,
        0.95,
        f"White's Test:\nLM Statistic = {het_white_res[0]:.3f}\nLM-Test p-value = {het_white_res[1]:.3f}\nF-Statistic = {het_white_res[2]:.3f}\nF-Test p-value = {het_white_res[3]:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )
    
    fig.savefig(
        f"/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/{name}_{G}_{motif_or_cluster}_rvf_plot.png"
    )


def make_density_plot(name, result, norm_res, m_or_c="M", motif=None, cluster=None):
    G = None
    motif_or_cluster = None
    if m_or_c == "Motif":
        G = "motif"
        motif_or_cluster = motif
    elif m_or_c == "Cluster":
        G = "cluster"
        motif_or_cluster = cluster

    fig = plt.figure(figsize=(16, 9))
    ax = sns.kdeplot(data=result.resid, fill=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, result.resid.mean(), result.resid.std())
    ax.plot(x, p, "k", linewidth=2)
    ax.set(
        xlabel="Residuals",
        title=f"KDE Plot of Model Residuals (Blue) and Normal Distribution (Black), {G}: {motif_or_cluster}",
    )
    # Add Shapiro-Wilk test results to the plot
    ax.text(
        0.01,
        0.95,
        f"Shapiro-Wilk Test:\nStatistic = {norm_res[0]:.3f}\np-value = {norm_res[1]:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )
    fig.savefig(
        f"/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/{name}_{G}_{motif_or_cluster}_kde_plot.png"
    )


def make_qq_plot(name, result, norm_res, m_or_c="M", motif=None, cluster=None):
    G = None
    motif_or_cluster = None
    if m_or_c == "Motif":
        G = "motif"
        motif_or_cluster = motif
    elif m_or_c == "Cluster":
        G = "cluster"
        motif_or_cluster = cluster

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    sm.qqplot(data=result.resid, dist=scipy.stats.norm, line="s", ax=ax)
    ax.set_title(f"Q-Q Plot, {G}: {motif_or_cluster}")
    # Add Shapiro-Wilk test results to the plot
    ax.text(
        0.01,
        0.95,
        f"Shapiro-Wilk Test:\nStatistic = {norm_res[0]:.3f}\np-value = {norm_res[1]:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )
    fig.savefig(
        f"/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/{name}_{G}_{motif_or_cluster}_qq_plot.png"
    )

import plotly.graph_objects as go


def plot_outliers_vs_inliers(model_df):
    y_val = model_df["Value"]
    x_val = model_df["Time_Point"]
    grouping = model_df["Treatment_State"]

    color = model_df["threshold_score"]

    fig = go.Figure()

    # Add scatter plot for inliers
    fig.add_trace(go.Scatter(
        x=x_val[model_df["is_inlier"] != -1],
        y=y_val[model_df["is_inlier"] != -1],
        mode='markers',
        marker=dict(
            color=color[model_df["is_inlier"] != -1],  # use threshold_score for color
            colorscale="Viridis",
            colorbar=dict(title="Threshold Score"),  # add a colorbar with title
            showscale=True  # show the color scale
        ),
        name='Inliers'
    ))

    # Add scatter plot for outliers with red circle marker
    fig.add_trace(go.Scatter(
        x=x_val[model_df["is_inlier"] == -1],
        y=y_val[model_df["is_inlier"] == -1],
        mode='markers',
        marker=dict(color='red', size=10, symbol='circle'),
        name='Outliers'
    ))

    fig.update_layout(title="Outliers vs. Inliers", xaxis_title="Time Point", yaxis_title="Value")
    fig.show()
    
def make_residuals_distribution_boxplot(name, result, het_white_res, m_or_c="M", motif=None, cluster=None):
    G = None
    motif_or_cluster = None
    if m_or_c == "M":
        G = "motif"
        motif_or_cluster = motif
    elif m_or_c == "C":
        G = "cluster"
        motif_or_cluster = cluster

    fig = plt.figure(figsize=(16, 9))
    ax = sns.boxplot(x=result.model.groups, y=result.resid)
    ax.set_title(f"Distribution of Residuals for Animal ID, {G}: {motif_or_cluster}")
    ax.set_ylabel("Residuals")
    ax.set_xlabel("Animal ID")
    ax.text(
        0.01,
        0.95,
        f"White's Test:\nLM Statistic = {het_white_res[0]:.3f}\nLM-Test p-value = {het_white_res[1]:.3f}\nF-Statistic = {het_white_res[2]:.3f}\nF-Test p-value = {het_white_res[3]:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )
    fig.savefig(
        f"/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/{name}_{G}_{motif_or_cluster}_residuals_boxplot.png"
    )

def remove_outliers(model_df):
    console.print("Removing Outliers, make sure you have filtered the datafram for a single motif/cluster", style = "bold red")
    random_state = np.random.RandomState(42)
    model_iF: ExtendedIsolationForest = eif(
                n_estimators=100,
                max_samples="auto",
                contamination="auto",
                verbose=0,
                random_state=random_state,
                n_jobs=-1,
            ).fit(model_df[["Value"]])
            
            
    model_df["scores"] = model_iF.decision_function(
                model_df[["Value"]]
            )  # anomaly score. The lower, the more abnormal, negative scores are outlines, positive are inliers
            
    model_df["threshold_score"] = model_iF.score_samples(
                model_df[["Value"]]
            )  # anomaly score, the lower, the more abnormal
            
    model_df["is_inlier"] = model_iF.predict(model_df[["Value"]])

    outlier_count = model_df[model_df["is_inlier"] == -1].shape[0]
    console.print(f"Number of outliers: {outlier_count}", style="red")
    # Count the number of inliers
    inlier_count = model_df[model_df["is_inlier"] != -1].shape[0]
    console.print(f"Number of inliers: {inlier_count}", style="green")
    # print the percentage of rows that are to be removed
    percentage = (outlier_count / (outlier_count + inlier_count)) * 100
    console.print(f"Percentage of rows to be removed: {percentage}%", style="bold red")

    model_df = model_df[model_df["is_inlier"] != -1]
    return model_df

def filter_by_phase(df, phase):
    if phase == "pre":
        df = df[df["Time_Point"] != "Week_11"]
        df = df[df["Time_Point"] != "Week_13"]
        df = df[df["Time_Point"] != "Week_15"]
    elif phase == "post":
        df = df[df["Time_Point"] != "Week_00"]
        df = df[df["Time_Point"] != "Week_02"]
        df = df[df["Time_Point"] != "Week_04"]
        df = df[df["Time_Point"] != "Week_06"]
        df = df[df["Time_Point"] != "Week_08"]
    return df

if __name__ == "__main__":
    filter_dfs = False
    check_normality_cond = False
    make_new_plots = False
    add_p_vals = False
    use_mlmixed_model = True
    make_model_plots = False
    m_or_c = "Desc_Behavior"
    phase = "pre"
    find_fit = False
    report_data_path = "/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/"
    df = pd.read_csv(
        "/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/baseline_summed_df_hmm.csv"
    )
    df.drop(columns=["Unnamed: 0"], inplace=True)
    
    df = df[df["Time_Point"] != "Drug_Trt"]
    #df = filter_by_phase(df, phase)
        
    #states = ["Sham", "Injured"]
    phases = ["pre", "post"]

    df_log = AlSt.log_transform_values(df)
    df_log = df_log.reset_index(drop=True)

    # Drop the "Value" column in df_log
    df_log.drop(columns=["Value"], inplace=True)
    df_log.rename(columns={"log_value": "Value"}, inplace=True)

    df_norm = AlSt.normalize_to_baseline_sham(df, handle_miss_baseline_values=False)
    df_log_norm = AlSt.normalize_to_baseline_sham(
        df_log, handle_miss_baseline_values=False
    )

    # Drop the "Value" column
    df_normSham = df_norm.drop(columns=["Value"])
    df_normSham.rename(columns={"Sham_Normalized_Value": "Value"}, inplace=True)

    df_log_normSham = df_log_norm.drop(columns=["Value"])
    df_log_normSham.rename(columns={"Sham_Normalized_Value": "Value"}, inplace=True)

    dataframes = {
        "dataframe": ("df", df),
        "dataframe_log": ("df_log", df_log),
        "dataframe_normSham": ("df_normSham", df_normSham),
        "dataframe_log_normSham": ("df_log_normSham", df_log_normSham),
    }
    if filter_dfs:
        for name, (key, dataframe) in dataframes.items():
            motif_df_new = []
            for value in dataframe[m_or_c].unique():

                model_df = dataframe[dataframe[m_or_c] == value]

                model_df.loc[model_df["Value"].isna(), "Value"] = 0
                
                # Loop over each unique Treatment_State and Time_Point
                for treatment in model_df['Treatment_State'].unique():
                    for time_point in model_df['Time_Point'].unique():
                        # Subset the DataFrame for the current Treatment_State and Time_Point
                        subset_df = model_df[(model_df['Treatment_State'] == treatment) & (model_df['Time_Point'] == time_point)]
                        if len(subset_df) == 0:
                            continue
                        subset_df = remove_outliers(subset_df)
                        
                        motif_df_new.append(subset_df)

            
                # Turn motif_df_new into a dataframe
            model_df = pd.concat(motif_df_new)
            dataframes[name] = (key, model_df)
        


    if check_normality_cond:
        norm_results = {}
        fit_results = {}

        for name, dataframe in [("df", df), ("df_normSham", df_normSham), ("df_log", df_log), ("df_log_normSham", df_log_normSham)]:
            norm_results[f"{name}_normality{m_or_c}"] = check_normality_tps_checks(dataframe, check=m_or_c)
            if find_fit:
                fit_results[f"{name}_fit{m_or_c}"] = find_data_bestFit(dataframe, check=m_or_c)
                with open("fit_results.json", "w") as f:
                    json.dump(fit_results, f, cls=MyEncoder)
            
            states = dataframe["Treatment_State"].unique()
            tps = dataframe["Time_Point"].unique()
            tp_order = ["Week_00", "Week_02", "Week_04", "Week_06", "Week_08", "Week_11", "Week_13", "Week_15"]

            dataframe.rename(columns={"Time_Point_Int": "Week"}, inplace=True)
            
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))

            sns.kdeplot(data=dataframe, x="Value", hue="Week", fill=True, ax=axs[0])
            axs[0].set_title(f"Density Plot of {name} by Time Point", y=1.035)
            axs[0].text(x=0.5, y=1.01, s="by Time Point", fontsize=10, ha="center", transform=axs[0].transAxes)
            axs[0].set_xlabel('')  
            sns.kdeplot(data=dataframe, x="Value", hue="Treatment_State", fill=True, ax=axs[1])
            axs[1].set_title(f"Density Plot of {name} by Treatment State", y=1.035)
            axs[1].text(x=0.5, y=1.01, s="by Treatment State", fontsize=10, ha="center", transform=axs[1].transAxes)
            axs[1].set_xlabel('')  #

            sns.kdeplot(data=dataframe, x="Value", hue="Cluster", fill=True, ax=axs[2])
            axs[2].set_title(f"Density Plot of {name} by Cluster", y=1.035)
            axs[2].text(x=0.5, y=1.01, s="by Cluster", fontsize=10, ha="center", transform=axs[2].transAxes)
            axs[2].set_xlabel('Frequency') 

            plt.tight_layout()
            plt.savefig(os.path.join(report_data_path, f"Density_plots_{name}.png"), dpi=900)
            plt.close()
            
            import matplotlib.pyplot as plt
            from scipy.stats import probplot

            fig, axs = plt.subplots(2, 4, figsize=(20, 15))
            wks = dataframe["Week"].unique()
            wks.sort()
            count = 0

            for i in range(2):
                for j in range(4):
                    if count < len(wks):
                        wk = wks[count]
                        probplot(dataframe["Value"][dataframe["Week"] == wk], plot=axs[i, j])
                        axs[i, j].set_title(f"Week {wk}", y=1.0)
                        count += 1

            fig.suptitle(f'Q-Q Plots of {name}, by Week', fontsize=16, y=0.99)
            plt.tight_layout()
            plt.savefig(os.path.join(report_data_path, f"qq_plots_byWeek_{name}.png"), dpi=900)
            plt.close()
            
            trts = dataframe["Treatment_State"].unique()
            trts.sort()
            fig, axs = plt.subplots(len(trts), 1, figsize=(10, 5 * len(trts)))
            for i, t in enumerate(trts):
                probplot(dataframe["Value"][dataframe["Treatment_State"] == t], plot=axs[i])
                axs[i].set_title(t, y=1.0)
                if i == len(trts) - 1:
                    axs[i].set_xlabel('Theoretical Quantiles')
                else:
                    axs[i].set_xlabel('')

            fig.suptitle(f'Q-Q Plots of {name}, by Treatment State', fontsize=16, y=0.999)
            plt.tight_layout()
            plt.savefig(os.path.join(report_data_path, f"qq_plots_byTreatmentState_{name}.png"), dpi=900)
            plt.close()
            
            count = 0
            clsts = dataframe["Cluster"].unique()
            clsts.sort()
            fig, axs = plt.subplots(3, 2, figsize=(10, 5 * len(clsts)))
            fig.suptitle(f'Q-Q Plots of {name}, by Cluster', fontsize=16, y=0.995)
            for i in range(3):
                for j in range(2):
                    if count < len(clsts):
                        c = clsts[count]
                        probplot(dataframe['Value'][dataframe['Cluster'] == c], plot=axs[i, j])
                        axs[i, j].set_title(c, y=1.0)
                        if i == len(trts) - 1:
                            axs[i, j].set_xlabel('Theoretical Quantiles')
                        else:
                            axs[i, j].set_xlabel('')
                        count += 1
                        
            plt.tight_layout()
            fig.subplots_adjust(bottom=0.05, top=0.95, hspace=0.25)
            plt.savefig(os.path.join(report_data_path, f"qq_plots_byCluster_{name}.png"), dpi=900)
            plt.close()

        with open("norm_results.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(norm_results.keys())
            writer.writerows(zip(*norm_results.values()))

        for key, value in norm_results.items():
            df = pd.DataFrame(value)
            full_path = os.path.join("/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report", f"q1_{key}.csv")
            df.to_csv(full_path, index=False)

    if make_new_plots:
        summed_baseline_df = df
        for i in range(summed_baseline_df["Motif"].nunique()):
            fig = AlSt.plot_motif_data(
                summed_baseline_df,
                motif=i,
                cluster=None,
                plot_type="boxplot",
                y_label="Motif Frequency",
                title="Usage Over Time for Motif " + str(i),
            )
            fig_name = "q1_hmm_Motif_Usage_Boxplot_" + str(i) + ".html"
            fig_path = "/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/motif_usage_plots_hmm"
            fig.write_html(fig_path + "/" + fig_name)

    # ---------------------------------------------------#
    # Adding p values to dataframe                      #
    # ---------------------------------------------------#
    if add_p_vals:
        pVals_df = pd.DataFrame()
        for f_tuple in dataframes.values():
            f = f_tuple[0]  # Extract the DataFrame from the tuple

            pVals_motif_df = calculate_and_adjust_p_values(f, "Motif")
            pVale_cluster_df = calculate_and_adjust_p_values(f, "Cluster")

            merged_df_motif = add_state_means_to_df(f, "Motif")
            merged_df_cluster = add_state_means_to_df(f, "Cluster")

            pVals_motif_df = pd.merge(
                merged_df_motif, pVals_motif_df, on=["Time_Point", "Motif"], how="outer"
            )

            pVals_cluster_df = pd.merge(
                merged_df_cluster,
                pVale_cluster_df,
                on=["Time_Point", "Cluster"],
                how="outer",
            )

            pVals_motif_df["Comparison"] = (
                pVals_motif_df["Treatment_State"] + " vs. Sham"
            )
            pVals_cluster_df["Comparison"] = (
                pVals_cluster_df["Treatment_State"] + " vs. Sham"
            )
            # Save the dataframes
            pVals_motif_df.to_csv(
                "/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/pVals_motif_df.csv"
            )
            
            pVals_cluster_df.to_csv(
                "/Users/adanredwine/Library/CloudStorage/OneDrive-UniversityofNebraska-Lincoln/ONE Lab - New/Personnel/Adan/Experimental Plans & Data/2022_Rat Antiobiotic + LBP Study/Outlining_Results/Data_used_in_report/pVals_cluster_df.csv"
            )

            # console.print(f"pVals_motif_df: {pVals_motif_df}")
            # console.print(f"pVals_cluster_df: {pVals_cluster_df}")
    # ---------------------------------------------------#
    # Generalized Linear Mixed  Model                    #
    # ---------------------------------------------------#  
    """
    import patsy
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM as BGLM
    from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM as PGLM
    fixed_effects = "C(Treatment_State, Treatment('Sham')) + C(Time_Point)"
    interaction = "C(Treatment_State, Treatment('Sham')):C(Time_Point)"
    dependent_var = "Value"
    temp_df = df.copy()
    
    #for motif in temp_df["Motif"].unique():
        
        #temp_df_motif = temp_df[temp_df["Motif"] == motif]
        #temp_df_motif = remove_outliers(temp_df_motif)

    temp_df.dropna(subset=["Animal_ID"], inplace=True)
    temp_df.reset_index(drop=True, inplace=True)
    formula = f"{dependent_var} ~ {fixed_effects} + {interaction}"
    
    y, X = patsy.dmatrices(formula, data = temp_df, return_type='dataframe') # type: ignore
    
    animal_ids = temp_df['Animal_ID'].unique()
    num_animals = len(animal_ids)
    animal_id_map = {animal_id: idx for idx, animal_id in enumerate(animal_ids)}
    exog_vc = np.zeros((temp_df.shape[0], num_animals))
    for i, row in temp_df.iterrows():
        animal_idx = animal_id_map[row['Animal_ID']]
        if i < exog_vc.shape[0] and animal_idx < exog_vc.shape[1]:
            exog_vc[i, animal_idx] = 1
        else:
            print(f"Warning: index {i} or {animal_idx} is out of bounds for exog_vc with shape {exog_vc.shape}")

    ident = np.arange(num_animals)
    
    fep_names = X.columns.tolist()
    vcp_names = ['VarComp_Animal_' + str(animal_id) for animal_id in animal_ids]
    vc_names = ['RE_Animal_' + str(animal_id) for animal_id in animal_ids]
    y = y.squeeze()
    # Initialize and fit the Poisson Bayesian GLMM
    model = PGLM(endog=y, exog=X, exog_vc=exog_vc, ident=ident, vcp_p=2, fe_p=2, fep_names=fep_names, vcp_names=vcp_names, vc_names=vc_names)
    #model = BGLM(endog=y, exog=X, exog_vc=exog_vc, ident=ident, vcp_p=2, fe_p=2, fep_names=fep_names, vcp_names=vcp_names, vc_names=vc_names)
    result = model.fit_map()  # Using the Laplace approximation
    console.print(result.summary())
    # Calculate Wald statistics

    
    ic(result.fe_mean) # Posterior mean of the fixed effects coefficients.
    ic(result.fe_sd) # Posterior standard deviation of the fixed effects coefficients.
    ic(result.vcp_mean) # Posterior mean of the logged variance component standard deviations.
    ic(result.vcp_sd) # Posterior standard deviation of the logged variance component standard deviations.
    ic(result.vc_mean) # Posterior mean of the random coefficients.
    ic(result.vc_sd) # Posterior standard deviation of the random coefficients.
    ic(result.random_effects()) # Data frame of posterior means and posterior standard deviations of random effects.
    

    fitted_params = result.params
    ic(model.logposterior_grad(fitted_params))

    predicted_values=result.predict(X) # .predict() returns one-dimensional array of fitted values.
    ic(X)
    ic(predicted_values)
    ic(predicted_values.shape)
    ic(y)
    ic(y.shape)

    residuals = y-predicted_values
    sns.scatterplot(x=predicted_values, y=residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

    """
    # ---------------------------------------------------#
    # Linear Mixed Effects Model                         #
    # ---------------------------------------------------#
    import statsmodels.formula.api as smf

    #fixed_effects = "C(Treatment_State, Treatment('Sham')) + C(Time_Point)"
    #interaction = "C(Treatment_State, Treatment('Sham')):C(Time_Point)"
    dependent_var = "Value"
    random_effect = (
        "Animal_ID" 
    )
    
    #fixed_effects = "Treatment_State + Time_Point + Motif"
    
    fixed_effects = "is_injured + Time_Point"
    interaction = "is_injured:Time_Point"
    #random_effects = ["is_treated", "is_abx", "Animal_ID"]
    
    re_f = "~Time_Point" # Allows for the slope of random effect(s) to change with time
    
    vc = {"is_trt": "0 + C(is_treated)", "is_abx": "0 + C(is_abx)"} # specifies variance components for random effects
    
    # Construct the formula with 'Sham' as the reference level for 'Treatment_State'
    formula = f"{dependent_var} ~ {fixed_effects} + {interaction}"
    console.print(formula, style="bold red")  # To verify the formula
    mixedlm_models = {}
    from numpy.linalg import matrix_rank
    if use_mlmixed_model: #! Try running at EACH time point and looking for interaction between 
        #! Try running across all time points for each treatment state (while including all motifs/clusters?)
        
        logger.add(
            f"logs/mixedlm.log",
            colorize=True,
            format="{time} {level} {message}",
            level="DEBUG",
        )
        
        for name, (key, dataframe) in dataframes.items():
            dataframe = dataframe.copy()

            dataframe["Treatment_State"] = pd.Categorical(
                dataframe["Treatment_State"], categories=["Sham", "Injured", "Treated", "ABX"], ordered=False
            )
            
            custom_order = ["Week_00", "Week_02", "Week_04", "Week_06", "Week_08", "Week_11", "Week_13", "Week_15"]

            time_ps_unsorted = dataframe["Time_Point"].unique()
            time_ps = sorted(time_ps_unsorted, key=lambda x: custom_order.index(x))
            
            dataframe["Time_Point"] = pd.Categorical(
                dataframe["Time_Point"], categories=time_ps, ordered=True
            )

            for value in dataframe[m_or_c].unique():
                model_df = dataframe[dataframe[m_or_c] == value]
                model_df.loc[model_df["Value"].isna(), "Value"] = 0
                if len(model_df) > 1: 

                    model = smf.mixedlm(formula, data=model_df, groups=model_df[random_effect], re_formula=re_f, vc_formula=vc)

                    result = None
                    
                    try:
                        result = model.fit(maxiter=10000)
                        logger.info(f"formula: {formula}, re_formula: {re_f}, vc_formula: {vc}, random_effect: {random_effect}")
                                            # Check the rank of the design matrix
                        rank = matrix_rank(result.model.exog)

                        # Check the degrees of freedom of the model
                        df_model = result.df_model
                        if rank != df_model + 1:
                            print("The design matrix has linearly dependent columns.")

                    except Exception as e:
                        logger.debug(f"Failed to fit model for {m_or_c} {value}: {e}")
                    
                    logger.debug(f"Done fitting model for {m_or_c}: " + str(value))
                    
                    if result is not None:
                        logger.debug(result.summary())
                        summary = result.summary().tables[1]
                        console.print(result.summary())
                        summary_str = str(summary)

                        lines = summary_str.split("\n")
                        columns = re.split(r"\s{2,}", lines[0])
                        columns = [
                            "",
                            "Coef.",
                            "Std.Err.",
                            "z",
                            "P>|z|",
                            "[0.025",
                            "0.975]",
                        ]
                        data = [re.split(r"\s{2,}", line) for line in lines[1:]]
                        res_df = pd.DataFrame(data, columns=columns)

                        # Convert numeric columns to appropriate data types
                        for col in ["Coef.", "Std.Err.", "z", "P>|z|", "[0.025", "0.975]"]:
                            if col in res_df.columns:
                                res_df[col] = pd.to_numeric(res_df[col], errors="coerce")
                        
                        # Add a column to res_df with the motif
                        res_df[m_or_c] = value
                        res_df["DataFrame"] = name
                        
                        ## Normality Assumption
                        labels = ["Statistic", "p-value"]
                        norm_res = scipy.stats.shapiro(result.resid)
                        ## Homoscedasticity of Variance
                        het_white_res = het_white(result.resid, result.model.exog)
                        
                        res_df["Norm_Stat"] = norm_res.statistic
                        res_df["Norm_pval"] = norm_res.pvalue
                        
                        labels = [
                            "LM Statistic",
                            "LM-Test p-value",
                            "F-Statistic",
                            "F-Test p-value",
                        ]

                        res_df["Homo_LM_Stat"] = het_white_res[0]
                        res_df["Homo_LM_pval"] = het_white_res[1]
                        res_df["Homo_F_Stat"] = het_white_res[2]
                        res_df["Homo_F_pval"] = het_white_res[3]    
                        
                        console.print(res_df)
                        
                        if make_model_plots:
                            motif = None
                            cluster = None
                            # * Density Plot
                            ic("Density Plot")
                            
                            if m_or_c == "Motif":
                                motif = value
                                cluster = None
                            elif m_or_c == "Cluster":
                                motif = None
                                cluster = value
                                
                            make_density_plot(name, result, norm_res, m_or_c, motif, cluster)

                            # * QQ Plot
                            ic("QQ Plot")
                            make_qq_plot(name, result, norm_res, m_or_c, motif, cluster)

                            # * Residuals Scatter plot
                            ic("Residuals Scatter plot")
                            make_rvf_plot(
                                name, result, het_white_res, m_or_c, motif, cluster
                            )  
                            
                            # * Group Distributions of Residuals
                            ic("Group Distributions of Residuals")
                            make_residuals_distribution_boxplot(
                                name, result, het_white_res, m_or_c, motif, cluster
                            )  
                else:
                    print(f"Not enough data to fit model for {m_or_c}: {value}")

    # ---------------------------------------------------#
    # Plotting Histograms                               #
    # ---------------------------------------------------#
    """
    # Now you can call the function for each DataFrame
    hist_df = plot_histogram_and_shapiro(df)
    hist_df_log = plot_histogram_and_shapiro(df_log)
    hist_df_normSham = plot_histogram_and_shapiro(df_normSham)
    hist_df_log_normSham = plot_histogram_and_shapiro(df_log_normSham)

    hist_df.show()
    hist_df_log.show()
    hist_df_normSham.show()
    hist_df_log_normSham.show()
    """
    plot_motifs = False
    if plot_motifs:
        fig = AlSt.plot_motif_data(
            df_log_normSham,
            motif=1,
            cluster=None,
            plot_type="boxplot",
            y_label="Motif Frequency",
            title="Usage Over Time for Motif ",
        )
    # fig.show()
    # Add the P_Value column from pVals to df based on the "Motif", "Time_Point", and "Treatment_State" columns
    # df_log_normSham = df_log_normSham.merge(pVals_motif_df_log_normSham, on=["Motif", "Time_Point", "Treatment_State"], how="left")
    # Rename "P_Value" to "P_Value_State"
    # df_log_normSham.rename(columns={"P_Value": "P_Value_State"}, inplace=True)
    # AlPf.plot_motifs_all_time_points(df_log_normSham, comparison="Treatment_State", value_col="Value")
    # d.open_browser()


"""
Citations



"""
