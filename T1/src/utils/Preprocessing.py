import torchaudio.functional as F
from scipy.signal import get_window
from torchvision.transforms import Normalize
import torch
from src.utils.utils import get_mel_spectrogram, padding, get_MFCC, scale_vector


class Preprocessing:
    def __init__(self, frame_size, hop, n_mels, n_fft, n_mfcc, samplerate, max_large):
        self.samplerate=samplerate
        self.frame_size = frame_size
        self.hop = hop
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.max_large = max_large
        #self.mel_spectrogram = get_mel_spectrogram(
        #                            frame_size, 
        #                            hop, 
        #                            n_mels, 
        #                            n_fft, 
        #                            samplerate,
        #                            window_type="hamming"
        #                        )
        self.mfcc = get_MFCC(frame_size, 
                     hop, 
                     n_mels, 
                     n_fft,
                     n_mfcc, 
                     samplerate)
    
    def transform(self,waveform):
        signal = padding(waveform,self.max_large)
        #signal = torch.Tensor(waveform)
        mfcc_features = self.mfcc(signal)
        
        deltas = F.compute_deltas(specgram = mfcc_features,
                             win_length=3)
        deltasdeltas = F.compute_deltas(specgram = deltas,
                                            win_length=3)
            
        ft1 = torch.mean(mfcc_features,dim=1)
        ft2 = torch.mean(deltas,dim=1)
        ft3 = torch.mean(deltasdeltas,dim=1)
        #ft4 = mfcc_features.flatten()
        #features = torch.cat((ft1,ft2,ft3,ft4))

        features = torch.cat((ft1,ft2,ft3))
        features_scaled = scale_vector(features.unsqueeze(0)).squeeze(0)
        
        return features_scaled
    
    
