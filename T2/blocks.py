import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch

class HiddenLayer(nn.Module):
    """
    inputs:
    - input_dim: dimensión de entrada de la capa.
    - output_dim: dimensión de salida de la capa.
    - dropout: probabilidad de dropout

    output: modulo con capas Linear, Tanh y Dropout.
    """
    def __init__(self, input_dim, output_dim, dropout):
        super(HiddenLayer, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x0 = self.layer(x)
        x1 = self.activation(x0)
        out = self.dropout(x1)
        return out
    
class CustomRNN(nn.Module):
    """
    inputs: 
    - input_size: dimensión de datos que entran a RNN.
    - hidden_size: dimensión de las capas ocultas de RNN (entrada a MLP)
    - num_lstm_layers: cantidad de capas RNN.
    - num_mlp_layers: cantidad de capas de la MLP.
    - output_size: dimensión de los datos de salida de la MLP
    - dropout: dropout

    output: LSTM con num_lstm_layers -> CustomMLP.
    
    """
    def __init__(self, input_size,hidden_size, num_lstm_layers,num_mlp_layers, output_size, dropout):
        super().__init__()
        assert num_mlp_layers in [1,2], "Número de capas de mlp solo puede ser 1 o 2"

        self.num_mlp_layers = num_mlp_layers
        self.input_size = input_size
        # capa recurrente
        self.lstm = torch.nn.GRU(input_size=self.input_size, 
                                  hidden_size=hidden_size, 
                                  num_layers=num_lstm_layers, 
                                  batch_first=True,
                                  dropout=dropout,
                                  bidirectional=True)

        if self.num_mlp_layers==1:
            self.fc = nn.Linear(2*hidden_size,output_size)
        elif self.num_mlp_layers==2:
            self.fc = HiddenLayer(2*hidden_size,2*hidden_size//2,dropout=dropout)
            self.fc2 = nn.Linear(2*hidden_size//2,output_size)

        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, lengths):
        #Se empaquetan inputs, para que en cada batch se procesen solo los datos útiles.
        packed_input = rnn_utils.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        packed_output, h_n= self.lstm(packed_input)
        #Se desempaquetan los outputs del batch 
        output, _ = rnn_utils.pad_packed_sequence(packed_output,batch_first=True)
        #Se saca el último output válido para cada dato.
        valid_output = self.get_last_valid_output(output,lengths)
        #valid_output = h_n[-1]
        if self.num_mlp_layers==1:
            mlp_output = self.fc(valid_output)
        elif self.num_mlp_layers==2:
            mlp_output = self.fc(valid_output)
            mlp_output = self.fc2(mlp_output)
        return (mlp_output)
    
    def get_last_valid_output(self,output,lengths):
        batch_size = output.size(0)
        last_valid_output = []
        for i in range(batch_size):
            l = lengths[i]
            
            last_valid_output.append(output[i,l-1,:])
        return torch.stack(last_valid_output)
    