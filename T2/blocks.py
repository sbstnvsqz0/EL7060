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
    
class CustomMLP(nn.Module):
    """
    inputs:
    - input_dim: dimensión de entrada de MLP
    - output_dim: dimensión de salida de MLP
    - n_layer: número de capas ocultas
    - dropout: probabilidad de dropout

    output: MLP tal que las neuronas de las hidden layers se van reduciendo a la mitad:
    input_dim -> input_dim//2 -> input_dim//4 -> ... -> input_dim//2**(n_layers-1) -> output_dim.
    Finaliza con un softmax.
    """
    def __init__(self,input_dim,output_dim,n_layers,dropout):
        super().__init__()
        self.n_layers = n_layers
        hidden_dims = [input_dim//(2**k) for k in range(1,n_layers+2)]
        self.first = nn.Linear(input_dim,hidden_dims[0])
        if self.n_layers>0:
            self.hidden = nn.ModuleList([HiddenLayer(hidden_dims[k],hidden_dims[k+1],dropout) for k in range(len(hidden_dims)-1)]) #Cadena de capas ocultas
        self.out = nn.Linear(hidden_dims[-1],output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.first(x)
        if self.n_layers>0:
            for layer in self.hidden:
                x = layer(x)
        x = self.out(x)
        x = self.softmax(x)

        return x


class CustomLSTM(nn.Module):
    """
    inputs: 
    - input_size: dimensión de datos que entran a LSTM.
    - hidden_size: dimensión de las capas ocultas de LSTM (entrada a MLP)
    - num_lstm_layers: cantidad de capas LSTM.
    - num_mlp_layers: cantidad de capas ocultas de la MLP.
    - output_size: dimensión de los datos de salida de la MLP
    - dropout: dropout

    output: LSTM con num_lstm_layers -> CustomMLP.
    
    """
    def __init__(self, input_size,hidden_size, num_lstm_layers,num_mlp_layers, output_size, dropout):
        super().__init__()

        # capa recurrente
        self.input_size = input_size
        self.lstm = torch.nn.GRU(input_size=self.input_size, 
                                  hidden_size=hidden_size, 
                                  num_layers=num_lstm_layers, 
                                  batch_first=True,
                                  dropout=dropout)

        # capa fully conected
        #self.fc = CustomMLP(input_dim = hidden_size,
                            #output_dim = output_size,
                            #n_layers = num_mlp_layers,
                            #dropout = dropout)
        self.fc = nn.Linear(hidden_size,output_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, lengths):
        #Se enpaquetan inputs, para que en cada batch se procesen solo los datos útiles.
        packed_input = rnn_utils.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        packed_output, _= self.lstm(packed_input)
        #Se desempaquetan los outputs del batch 
        output, _ = rnn_utils.pad_packed_sequence(packed_output,batch_first=True)
        #Se saca el último output válido para cada dato.
        valid_output = self.get_last_valid_output(output,lengths)
        mlp_output = self.fc(valid_output)
        return self.softmax(mlp_output)
    
    def get_last_valid_output(self,output,lengths):
        batch_size = output.size(0)
        last_valid_output = []
        for i in range(batch_size):
            l = lengths[i]
            last_valid_output.append(output[i,l-1,:])
        return torch.stack(last_valid_output)
    