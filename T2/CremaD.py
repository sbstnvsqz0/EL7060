import pandas as pd
import os
import librosa
import torch
from config import DATA_FOLDER, SAMPLERATE
from src.utils.utils import augment_audio

class CremaDDataset(torch.utils.data.Dataset):
    def __init__(self, dir, preprocessing=None,augmentation=None):
        assert dir in ["train","validation","test"], "Conjunto invalido"
        assert augmentation in [None, "speed","pitch", "distortion"], "Augmentation Invalida"
        self.dir = dir
        self.preprocessing = preprocessing 
        self.augmentation = augmentation
        df = (pd.read_csv(os.path.join(DATA_FOLDER, 'labels.csv'))[lambda x: x['partition'] == f"{self.dir}"])
        self.df = df
        
        
    def __getitem__(self,idx):
        signal_path = os.path.join(DATA_FOLDER,self.df.iloc[idx]["path"])
        waveform,_ = librosa.load(signal_path,sr=SAMPLERATE)
        label =  self.df.iloc[idx]["class"]

        if self.augmentation is not None:
            waveform = augment_audio(waveform,SAMPLERATE,augment_type=self.augmentation)
            
            
        if self.preprocessing is not None:
            not_padded_features = self.preprocessing.transform(waveform,pad=False)
            features = self.preprocessing.transform(waveform)
        else: 
            features = waveform
        
        return features,label, not_padded_features.size(0)
    
    def __len__(self):
        return len(self.df)