# src/denoising_methods.py

import pywt
import numpy as np

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1, threshold_mode='hard'):
    coeffs = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeffs[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode=threshold_mode) for i in coeffs[1:])

    ret = pywt.waverec(coeffs, wavelet, mode='per')
    
    return ret

def denoise_multichannel(X, wavelet='haar', level=1, threshold_mode='hard'):
    denoised = np.array([
        denoise(row, wavelet=wavelet, level=level, threshold_mode=threshold_mode)
        for row in X
    ])
    return denoised
