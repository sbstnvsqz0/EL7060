import torchaudio.functional as F
import torch
from src.utils.utils import get_MFCC


class Preprocessing:
    """
    Salen (n_windows,n_MFCC)
    """
    def __init__(self, frame_size, hop, n_mels, n_fft, n_mfcc, samplerate):
        self.samplerate=samplerate
        self.frame_size = frame_size
        self.hop = hop
        self.n_mels = n_mels
        self.n_fft = n_fft

        self.mfcc = get_MFCC(frame_size, 
                     hop, 
                     n_mels, 
                     n_fft,
                     n_mfcc, 
                     samplerate)
    
    def transform(self,waveform):
        signal = torch.Tensor(waveform)
        features = self.mfcc(signal).T
        features = torch.cat((features,torch.zeros((features.size[0],500-features.size[1]))),dim = 0)

        
        return features
    
    