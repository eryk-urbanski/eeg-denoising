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

def plot_psd_comparison(f, psd_original, psd_denoised, channel_idx=0, log_scale=True, title=None):
    """
    Plot PSD comparison between original and denoised signal for a given channel.

    Parameters:
        f (ndarray): Frequencies array from Welch's method
        psd_original (ndarray): PSD of original signal [channels, freqs]
        psd_denoised (ndarray): PSD of denoised signal [channels, freqs]
        channel_idx (int): Index of the channel to plot
        log_scale (bool): Whether to use log-log scale
        title (str): Optional title for the plot
    """
    plt.figure(figsize=(10, 5))
    if log_scale:
        plt.semilogy(f, psd_original[channel_idx], label='Original', alpha=0.7)
        plt.semilogy(f, psd_denoised[channel_idx], label='Denoised', alpha=0.7)
    else:
        plt.plot(f, psd_original[channel_idx], label='Original', alpha=0.7)
        plt.plot(f, psd_denoised[channel_idx], label='Denoised', alpha=0.7)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V²/Hz)')
    plt.title(title or f"PSD Comparison (Channel {channel_idx})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_psd_multiple_channels(f, psd_original, psd_denoised, channel_idxs=None, log_scale=True, title=None, n_cols=3):
    """
    Plot PSD comparison between original and denoised signals for multiple channels.

    Parameters:
        f (ndarray): Frequencies array from Welch's method
        psd_original (ndarray): PSD of original signal [channels, freqs]
        psd_denoised (ndarray): PSD of denoised signal [channels, freqs]
        channel_idxs (list or None): List of channel indices to plot; if None, plots first 6 channels
        log_scale (bool): Whether to use log scale
        title (str): Optional global title
        n_cols (int): Number of columns in the subplot grid
    """
    if channel_idxs is None:
        channel_idxs = list(range(min(6, psd_original.shape[0])))

    n_channels = len(channel_idxs)
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), squeeze=False)
    fig.suptitle(title or "PSD Comparison Across Channels", fontsize=14)

    for i, ch in enumerate(channel_idxs):
        ax = axes[i // n_cols][i % n_cols]
        if log_scale:
            ax.semilogy(f, psd_original[ch], label='Original', alpha=0.7)
            ax.semilogy(f, psd_denoised[ch], label='Denoised', alpha=0.7)
        else:
            ax.plot(f, psd_original[ch], label='Original', alpha=0.7)
            ax.plot(f, psd_denoised[ch], label='Denoised', alpha=0.7)

        ax.set_title(f"Channel {ch}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (V²/Hz)")
        ax.grid(True)
        if i == 0:
            ax.legend()

    # Hide unused subplots if any
    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()