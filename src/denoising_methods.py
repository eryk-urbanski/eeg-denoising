# src/denoising_methods.py

import pywt
import numpy as np
from sklearn.decomposition import FastICA

# WT thresholding

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wt_denoise(x, wavelet='haar', level=1, threshold_mode='hard'):
    coeffs = pywt.wavedec(x, wavelet, mode="per", level=level)
    sigma = (1/0.6745) * maddest(coeffs[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode=threshold_mode) for i in coeffs[1:])

    ret = pywt.waverec(coeffs, wavelet, mode='per')
    
    return ret

def wt_denoise_multichannel(X, wavelet='haar', level=1, threshold_mode='hard'):
    denoised = np.array([
        wt_denoise(row, wavelet=wavelet, level=level, threshold_mode=threshold_mode)
        for row in X
    ])
    return denoised

# ICA

def ica_denoise_multichannel(X, n_components=None, random_state=14):
    """
    Apply ICA to multichannel EEG array and remove components manually.

    Parameters:
        X (ndarray): Shape (n_channels, n_samples)
        n_components (int or None): Number of ICA components
        random_state (int): Seed for reproducibility

    Returns:
        X_denoised (ndarray): Denoised signal with same shape as input
    """
    ica = FastICA(n_components=n_components, random_state=random_state)
    sources = ica.fit_transform(X.T)  # shape: (n_samples, n_components)

    # Manual strategy to remove bad components could be applied here
    # For now we keep all components
    X_denoised = ica.inverse_transform(sources).T  # Back to (n_channels, n_samples)
    return X_denoised