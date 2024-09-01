import torchaudio.functional as F
from scipy.signal import get_window
from torchaudio.transforms import ComputeDeltas
import torch
from src.utils.utils import get_mel_spectrogram, padding, get_MFCC

class Preprocessing:
    def __init__(self, frame_size, hop, n_mels, n_fft, n_mfcc, samplerate):
        self.samplerate=samplerate
        self.frame_size = frame_size
        self.hop = hop
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.mel_spectrogram = get_mel_spectrogram(
                                    frame_size, 
                                    hop, 
                                    n_mels, 
                                    n_fft, 
                                    samplerate,
                                    window_type="hamming"
                                )
        self.mfcc = get_MFCC(frame_size, 
                     hop, 
                     n_mels, 
                     n_fft,
                     n_mfcc, 
                     samplerate)
    
    def transform(self,waveform,max_large,compute_deltas=False):
        out = padding(waveform,max_large)
        out = self.mfcc(out)
        a = torch.std(out,dim=1)
        out = torch.mean(out,dim=1)
        out = torch.concat((a,out))
        
        #TODO: computar deltas
        return out
    
    