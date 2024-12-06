import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch
from config import DEVICE
    
class CustomRNN(nn.Module):
    """
    inputs: 
    - input_size: dimensión de datos que entran a RNN.
    - hidden_size: dimensión de las capas ocultas de RNN (entrada a MLP)
    - num_rnn_layers: cantidad de capas RNN.
    - num_mlp_layers: cantidad de capas de la MLP.
    - output_size: dimensión de los datos de salida de la MLP
    - dropout: dropout

    output: RNN con num_rnn_layers -> CustomMLP.
    
    """
    def __init__(self, input_size,hidden_size, num_rnn_layers,num_mlp_layers, output_size, dropout):
        super().__init__()
        assert num_mlp_layers in [1,2], "Número de capas de mlp solo puede ser 1 o 2"

        self.num_mlp_layers = num_mlp_layers
        self.input_size = input_size
        # capa recurrente
        self.rnn = torch.nn.GRU(input_size=self.input_size, 
                                  hidden_size=hidden_size, 
                                  num_layers=num_rnn_layers, 
                                  batch_first=True,
                                  dropout=dropout,
                                  bidirectional=True)



        if self.num_mlp_layers==1:
            self.linear_activation = nn.Linear(2*hidden_size,1)
            self.linear_dominance = nn.Linear(2*hidden_size,1)
            self.linear_valence = nn.Linear(2*hidden_size,1)
        elif self.num_mlp_layers==2:
            self.linear_activation = nn.Sequential(nn.Linear(2*hidden_size, hidden_size),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_size,1))
            self.linear_dominance = nn.Sequential(nn.Linear(2*hidden_size, hidden_size),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_size,1))
            self.linear_valence = nn.Sequential(nn.Linear(2*hidden_size, hidden_size),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_size,1))

        self.activation = nn.Sigmoid()
            
    def forward(self, x, lengths):
        #Se empaquetan inputs, para que en cada batch se procesen solo los datos útiles.
        packed_input = rnn_utils.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        packed_output, h_n= self.rnn(packed_input)
        #Se desempaquetan los outputs del batch 
        rnn_output, _ = rnn_utils.pad_packed_sequence(packed_output,batch_first=True)
        #Se saca el último output válido para cada dato.}
        valid_output = self.get_last_valid_output(rnn_output,lengths)

        activation = self.linear_activation(valid_output)

        dominance = self.linear_dominance(valid_output)


        valence = self.linear_valence(valid_output)

        output = torch.cat([activation,dominance,valence],dim=-1)

        output = self.activation(output)
        
        return (output)
    
    def get_last_valid_output(self,output,lengths):
        batch_size = output.size(0)
        last_valid_output = []
        for i in range(batch_size):
            l = lengths[i]
            
            last_valid_output.append(output[i,l-1,:])
        return torch.stack(last_valid_output)
    