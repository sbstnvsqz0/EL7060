import torch
from scipy.signal import  get_window
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import MFCC
import matplotlib.pyplot as plt
import numpy as np
import joblib

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

#def padding(waveform,max_large):
 #   if len(waveform) < max_large:
  #      zero_number = max_large-len(waveform)
   #     if zero_number % 2 ==0: 
    #        n_pad_izq,n_pad_der = zero_number//2,zero_number//2
     #   else: 
      #      
   # else:
    #    return waveform
    #return torch.concat((torch.zeros(n_pad_izq),torch.Tensor(waveform),torch.zeros(n_pad_der)))
def padding(waveform,max_large):
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)
    if len(waveform)< max_large:#si el audio es mas corto se hace padding 
        zero_number = max_large-len(waveform)
        if zero_number % 2 ==0: 
            n_pad_izq,n_pad_der = zero_number//2,zero_number//2
        else:
            n_pad_izq, n_pad_der = zero_number//2, zero_number//2+1
        
    elif len(waveform) > max_large:  # Si el audio es más largo, se corta
        waveform = waveform[ :max_large]  # Truncar el audio al largo máximo
        return waveform
    return torch.concat((torch.zeros(n_pad_izq),torch.Tensor(waveform),torch.zeros(n_pad_der)))
    
        
       
def scale_vector(X):
    scaler = joblib.load("scaler.gz")
    return torch.Tensor(scaler.transform(X)).squeeze(0)


def compare_losses(losses_list, param_list, name_param):
    """
    Compara las losses de distintos modelos entrenados variando un parámetro
    """
    colors = [(1,0,0),(0,1,0),(0,0,1),(1,0,1),(0,1,1),(0.3,0.3,0.3),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5)]
    fig, ax = plt.subplots(1,1,figsize=(20,14))
    ax.set_title(f"Losses para entrenamientos con distintos {name_param}")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Loss")
    for i in range(len(losses_list)):
        ax.plot(np.arange(len(losses_list[i][0])),losses_list[i][0],'--',color=colors[i],label=f"train con {name_param}: {param_list[i]}")
        ax.plot(np.arange(len(losses_list[i][0])),losses_list[i][1], color=colors[i],label=f"validación con {name_param}: {param_list[i]}")

    ax.set_xlim([0,30])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()