import torch
import librosa
from src.utils.utils import get_MFCC, get_mel_spectrogram
import pandas as pd

class Preprocessing:
    """
    output: torch.Tensor(n_windows,n_MFCC) if not pad torch.Tensor(500,n_MFCC) if pad
    """
    def __init__(self, frame_size, hop, n_mels, n_fft, n_mfcc, samplerate):
        self.samplerate=samplerate
        self.frame_size = frame_size
        self.hop = hop
        self.n_mels = n_mels
        self.n_fft = n_fft
        
        self.melspec = get_mel_spectrogram(frame_size,
                                            hop,
                                            n_mels,
                                            n_fft,
                                            samplerate,
                                            window_type="hamming")

        self.mfcc = get_MFCC(frame_size, 
                     hop, 
                     n_mels, 
                     n_fft,
                     n_mfcc, 
                     samplerate)
        
        df = pd.read_csv("normalize_settings.csv")

        self.mean = torch.Tensor(df["mean"])
        self.min = torch.Tensor(df["min"])
        self.max = torch.Tensor(df["max"])
    
    def transform(self,waveform,pad=True):
        signal = torch.Tensor(waveform)
        mfcc = self.mfcc(signal).T
        logmelspec = torch.Tensor(librosa.power_to_db(self.melspec(signal))).T
        zfc = torch.Tensor(librosa.feature.zero_crossing_rate(y=waveform,frame_length=self.frame_size, hop_length=self.hop)).T
        rms = torch.Tensor(librosa.feature.rms(y=waveform,frame_length=self.frame_size, hop_length=self.hop)).T
        features = torch.cat((mfcc,logmelspec,zfc,rms),dim=1)
        features = (features-self.mean)/(self.max-self.min)

        # Padding (no se procesa)
        if pad:
            zeros = torch.zeros(500-features.size()[0],features.size()[1])
                   
            features = torch.cat((features,zeros),dim = 0)  
        return features
    
    
    