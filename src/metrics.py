# src/metrics.py

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import welch

def snr(reference, denoised):
    # Compute the signal power (ground truth signal)
    signal_power = np.sum(reference**2)
    
    # Compute the noise power (difference between reference and denoised signal)
    noise_power = np.sum((reference - denoised)**2)
    
    # Compute SNR
    return 10 * np.log10(signal_power / noise_power)

def rmse_value(reference, denoised):
    # Compute the RMSE between reference and denoised signals
    return np.sqrt(np.mean((reference - denoised)**2))

def pearson_corr(reference, denoised):
    r = pearsonr(reference, denoised)
    return r.statistic

def compute_psd(signal, sr, nperseg=256):
    """
    Compute Power Spectral Density (PSD) using Welch's method.

    Parameters:
        signal (ndarray): 1D array or 2D (channels, samples)
        sr (float): Sampling rate
        nperseg (int): Length of each segment (FFT window size)

    Returns:
        f (ndarray): Array of sample frequencies
        psd (ndarray): PSD values (shape: [n_channels, len(f)] or [len(f)])
    """
    if signal.ndim == 1:
        f, psd = welch(signal, fs=sr, nperseg=nperseg)
    else:
        psd_list = [welch(chan, fs=sr, nperseg=nperseg)[1] for chan in signal]
        f, _ = welch(signal[0], fs=sr, nperseg=nperseg)
        psd = np.array(psd_list)
    return f, psd