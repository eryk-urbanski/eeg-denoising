# src/denoising_methods.py

import pywt
import numpy as np

def denoise_wt(eeg_signal, wavelet='haar', level=3):
    
    coeffs = pywt.wavedec(eeg_signal, wavelet, mode="per")
    coeffs[1:] = [pywt.threshold(c, np.std(c), mode='hard') for c in coeffs[1:]]
    denoised_eeg = pywt.waverec(coeffs, wavelet)
    
    return denoised_eeg

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):
    coeffs = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeffs[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:])

    ret = pywt.waverec(coeffs, wavelet, mode='per')
    
    return ret