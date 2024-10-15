import torch
from scipy.signal import  get_window
from torchaudio.transforms import MFCC
import matplotlib.pyplot as plt
import numpy as np

class Window:
    def __init__(self,name,frame_size):
        self.name = name
        self.frame_size = frame_size
    def __call__(self,x):
        return x*torch.Tensor(get_window(self.name,self.frame_size))
    
def get_MFCC(frame_size, hop, n_mels, n_fft,n_mfcc, samplerate,window_type="hamming"):
    mfcc = MFCC(
    sample_rate = samplerate,
    n_mfcc = n_mfcc,
    melkwargs={
    "n_fft":n_fft,
    "win_length":frame_size,
    "hop_length":hop,
    "f_min" : 0.0,
    "f_max" : samplerate/2.0,
    "pad" : 0,
    "norm":"slaney",
    "n_mels": n_mels,
    "window_fn": Window(window_type,frame_size),
    "mel_scale":"htk"}
    )
    return mfcc