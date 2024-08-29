import torch
import os
import pandas as pd

DATA_FOLDER = 'TiDigits'
class SignalDataset(torch.utils.data.Dataset):
    """ 
    input: 
    Recibe tres posibles dir: "train", "validation","test"
    """
    def __init__(self,dir):
        assert dir in ["train","validation","test"], "Conjunto invalido"
        self.dir = dir
        self.df = (pd.read_csv(os.path.join(DATA_FOLDER, 'labels.csv'))[lambda x: x['partition'] == f"{self.dir}"])
        
    def __getitem__(self,idx):
        signal_path = os.path.join(DATA_FOLDER,self.df.iloc[idx]["path"])
        label =  self.df.iloc[idx]["class"]
        #TODO: preprocesamiento, editar dtypes si es necesario

        return signal_path,label    # De momento retorna path, label
    
    def __len__(self):
        return len(self.df) # Retorna el número de muestras en el conjunto de datos