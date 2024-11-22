import pandas as pd
import os
import librosa
import torch
from config import DATA_FOLDER, SAMPLERATE

class PodcastDataset(torch.utils.data.Dataset):
    def __init__(self, dir, preprocessing=None):
        assert dir in ["train","validation","test"], "Conjunto invalido"

        self.dir = dir
        self.preprocessing = preprocessing 
        df = (pd.read_csv(os.path.join(DATA_FOLDER, 'labels.csv'))[lambda x: x['partition'] == f"{self.dir}"])
        self.df = df
        #Means y stds sacadas desde dataset de entrenamiento para cada variable objetivo.
        self.means = {"activation":4.4851, "dominance":4.5854,"valence":3.9211}
        self.stds = {"activation":0.9946 , "dominance":0.9531,"valence":1.0403}
        
    def __getitem__(self,idx):
        signal_path = os.path.join(DATA_FOLDER,self.df.iloc[idx]["path"])
        waveform,_ = librosa.load(signal_path,sr=SAMPLERATE)
        activation =  (self.df.iloc[idx]["activation"]-self.means["activation"])/self.stds["activation"]
        valence = (self.df.iloc[idx]["valence"]-self.means["valence"])/self.stds["valence"]
        dominance = (self.df.iloc[idx]["dominance"]-self.means["dominance"])/self.stds["dominance"]
        
                       
        if self.preprocessing is not None:
            not_padded_features = self.preprocessing.transform(waveform,pad=False)
            features = self.preprocessing.transform(waveform)
        else: 
            features = waveform
        
        return features,activation, valence, dominance, not_padded_features.size(0)
    
    def __len__(self):
        return len(self.df)