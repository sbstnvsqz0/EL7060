import torch.nn as nn
import torch
from config import SEED
torch.manual_seed(SEED)


class HiddenLayer(nn.Module):
    """
    inputs:
    - hidden_dim: dimensión de entrada y salida de la capa.
    - dropout: probabilidad de dropout

    output: modulo con capas Linear, ReLU y Dropout.
    """
    def __init__(self, hidden_dim, dropout):
        super(HiddenLayer, self).__init__()
        self.layer = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x0 = self.layer(x)
        x1 = self.activation(x0)
        out = self.dropout(x1)
        return out

class MLP(nn.Module):
    """
    inputs:
    - input_dim: dimensión del input (número de features)
    - hidden_dim: dimensión de las capas ocultas (número de neuronas)
    - output_dim: dimensión del output (número de clases)
    - n_layers: número de capas ocultas
    - dropout: probabilidad de dropout
    """
    def __init__(self,input_dim,hidden_dim,output_dim,n_layers,dropout):
        super(MLP,self).__init__()
        self.first = nn.Linear(input_dim,hidden_dim)
        self.hidden = nn.ModuleList([HiddenLayer(hidden_dim,dropout) for _ in range(n_layers)]) #Cadena de capas ocultas
        self.out = nn.Linear(hidden_dim,output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        x = self.first(x)
        for layer in self.hidden:
            x = layer(x)
        x = self.out(x)
        out = self.softmax(x)

        return out