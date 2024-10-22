import torch
import librosa
import os
import pandas as pd
from config import DATA_FOLDER, SAMPLERATE, FRAME_SIZE, HOP, N_MELS, N_MFCC
from src.utils.Preprocessing import Preprocessing

class NormDataset(torch.utils.data.Dataset):
    def __init__(self, dir, preprocessing=None):
        assert dir in ["train","validation","test"], "Conjunto invalido"
        self.dir = dir
        self.df = (pd.read_csv(os.path.join(DATA_FOLDER, 'labels.csv'))[lambda x: x['partition'] == f"{self.dir}"])
        self.preprocessing = preprocessing 
        
    def __getitem__(self,idx):
        signal_path = os.path.join(DATA_FOLDER,self.df.iloc[idx]["path"])
        waveform,_ = librosa.load(signal_path,sr=SAMPLERATE)
        label =  self.df.iloc[idx]["class"]
        if self.preprocessing is not None:
            features = self.preprocessing.transform(waveform,pad=False)
        else: 
            features = waveform
        
        return features,label
    
    def __len__(self):
        return len(self.df)
    
def get_normalize_settings():

    p = Preprocessing(frame_size=FRAME_SIZE, 
                      hop=HOP, 
                      n_mels=N_MELS, 
                      n_fft=FRAME_SIZE, 
                      n_mfcc=N_MFCC, 
                      samplerate=SAMPLERATE,
                      normalize=False)
    
    dataset = NormDataset(dir="train",preprocessing=p)
    data_list = []
    for data, _ in dataset:
        data_list.append(data)

    all_data = torch.cat(data_list,axis=0)
    df=pd.DataFrame(torch.stack((all_data.mean(axis=0),all_data.std(axis=0)),dim=1),columns=["mean","std"])
    df.to_csv("normalize_settings.csv",index=False)


if __name__ == "__main__":
    get_normalize_settings()