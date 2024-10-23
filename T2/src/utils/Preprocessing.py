import torch
import librosa
from src.utils.utils import get_MFCC, get_mel_spectrogram
import pandas as pd

class Preprocessing:
    """
    output: torch.Tensor(n_windows,n_MFCC) if not pad torch.Tensor(200,n_MFCC) if pad
    """
    def __init__(self, frame_size, hop, n_mels, n_fft, n_mfcc, samplerate,normalize=True):
        self.samplerate=samplerate
        self.frame_size = frame_size
        self.hop = hop
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.normalize = normalize
        
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

        if self.normalize:
            df = pd.read_csv("normalize_settings.csv")

            self.mean = torch.Tensor(df["mean"])
            self.std = torch.Tensor(df["std"])
    
    def transform(self,waveform,pad=True):
        signal = torch.Tensor(waveform)
        mfcc = self.mfcc(signal).T
        logmelspec = torch.Tensor(librosa.power_to_db(self.melspec(signal))).T
        chromagram = torch.Tensor(librosa.feature.chroma_stft(y=waveform,
                                                 sr=self.samplerate,
                                                 n_fft=self.n_fft,
                                                 hop_length=self.hop,
                                                 win_length=self.frame_size,
                                                 window="hamming",
                                                 n_chroma=12)).T
        zfc = torch.Tensor(librosa.feature.zero_crossing_rate(y=waveform,frame_length=self.frame_size, hop_length=self.hop)).T
        rms = torch.Tensor(librosa.feature.rms(y=waveform,frame_length=self.frame_size, hop_length=self.hop)).T
    
        features = torch.cat((mfcc,zfc,rms,chromagram,logmelspec),dim=1)
        if self.normalize:
            features = (features-self.mean)/(self.std)

        # Padding (no se procesa)
        if pad:
            zeros = torch.zeros(500-features.size()[0],features.size()[1])
                   
            features = torch.cat((features,zeros),dim = 0)  
        return features
    
    
    