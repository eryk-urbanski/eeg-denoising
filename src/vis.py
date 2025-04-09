# src/vis.py

import matplotlib.pyplot as plt
import numpy as np

def plot_eeg(df, n_channels_to_plot=None):
    fig, axs = plt.subplots(len(df.columns) if n_channels_to_plot is None else n_channels_to_plot, 1, figsize=(15, 10), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(df.iloc[:,i], color="black")
        ax.set_ylabel(df.columns[i], rotation=0)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)

def plot_eeg_with_time(df, times, n_channels_to_plot=None):
    n_channels = len(df.columns) if n_channels_to_plot is None else n_channels_to_plot
    fig, axs = plt.subplots(n_channels, 1, figsize=(15, 2 * n_channels), sharex=True)

    # Make sure axs is iterable
    if n_channels == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        channel_name = df.columns[i]
        ax.plot(times, df.iloc[:, i], color="black")
        ax.set_ylabel(channel_name, rotation=0, labelpad=30, fontsize=10)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)

    axs[-1].set_xticks(np.linspace(times[0], times[-1], num=6))
    axs[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()