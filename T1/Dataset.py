import torch
import os
import pandas as pd
import librosa
from config import DATA_FOLDER, SAMPLERATE, MAX_SIZE

class SignalDataset(torch.utils.data.Dataset):
    """ 
    input: 
    Recibe tres posibles dir: "train", "validation","test"
    """
    def __init__(self,dir,preprocessing=None):
        assert dir in ["train","validation","test"], "Conjunto invalido"
        self.dir = dir
        self.df = (pd.read_csv(os.path.join(DATA_FOLDER, 'labels.csv'))[lambda x: x['partition'] == f"{self.dir}"])
        self.preprocessing = preprocessing

    def __getitem__(self,idx):
        signal_path = os.path.join(DATA_FOLDER,self.df.iloc[idx]["path"])
        waveform,_ = librosa.load(signal_path,sr=SAMPLERATE)
        label =  self.df.iloc[idx]["class"]

        if self.preprocessing is not None:
            features = self.preprocessing.transform(waveform)
        else: 
            features = waveform
        return features,label    
    
    def __len__(self):
        return len(self.df) # Retorna el número de muestras en el conjunto de datos