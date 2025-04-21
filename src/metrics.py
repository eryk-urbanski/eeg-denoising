# src/metrics.py

import numpy as np

def snr(reference, denoised):
    # Compute the signal power (denoised signal)
    signal_power = np.sum(denoised**2)
    
    # Compute the noise power (difference between reference and denoised signal)
    noise_power = np.sum((reference - denoised)**2)
    
    # Compute SNR
    return signal_power / noise_power

def rmse_value(reference, denoised):
    # Compute the RMSE between reference and denoised signals
    return np.sqrt(np.mean((reference - denoised)**2))

def pearson_correlation(reference, denoised):
    # Compute the Pearson correlation coefficient between reference and denoised signals
    return np.corrcoef(reference, denoised)[0, 1]