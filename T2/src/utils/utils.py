import torch
from scipy.signal import  get_window
from torchaudio.transforms import MFCC, MelSpectrogram
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd

class Window:
    def __init__(self,name,frame_size):
        self.name = name
        self.frame_size = frame_size
    def __call__(self,x):
        return x*torch.Tensor(get_window(self.name,self.frame_size))
    
def get_mel_spectrogram(frame_size, hop, n_mels, n_fft, samplerate,window_type="hamming"):
    mel = MelSpectrogram(
        sample_rate=samplerate,
        n_fft = n_fft,
        win_length = frame_size,
        hop_length = hop,
        f_min = 0.0,
        f_max = samplerate/2.0,
        pad = 0,
        norm="slaney",
        n_mels = n_mels,
        window_fn = Window(window_type,frame_size),
        mel_scale="htk")
    return mel
    
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

def augment_audio(signal,sr,augment_type='speed'):
    
    # Aumentación: Cambio de velocidad
    if augment_type == 'speed':
        y_aug = librosa.effects.time_stretch(y=signal, rate=np.random.uniform(0.9, 1.1))
    
    # Aumentación: Cambio de tono
    elif augment_type == 'pitch':
        y_aug = librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=np.random.randint(-2, 3))
    # Aumentación: distorción
    elif augment_type == 'distortion':
        y_aug = signal + 0.3 * np.sin(np.linspace(0, np.pi * 2, len(signal)))
    
    # Retorna el audio aumentado
    return y_aug

def plt_losses(path,title):
    df = pd.read_csv(path)
    plt.plot(np.arange(len(df["train"])),df["train"],label="Losses de entrenamiento")
    plt.plot(np.arange(len(df["val"])),df["val"],label="Losses de validacion")
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    