import argparse
import logging
from config import FRAME_SIZE, HOP, N_MELS, SAMPLERATE, N_MFCC ,DEVICE, OUT_DIM
from src.utils.Preprocessing import Preprocessing
from src.engine.RNN import EngineRNN

def get_args():
    parser = argparse.ArgumentParser('Entrenar la red')
    parser.add_argument("--epochs","-e",metavar="E",type=int,default=100,help="Número de épocas que se entrena la red")
    parser.add_argument("--batch_size","-b",metavar="B",type=int,default=32,help="Batch size")
    parser.add_argument("--hidden_size","-hs",type=int,default=64,help="Neuronas en capa oculta de LSTM")
    parser.add_argument("--num_lstm_layers","-nlstm",type=int,default=1,help="Número capas LSTM")
    parser.add_argument("--num_mlp_layers","-nmlp",type=int,default=2,help="Número capas ocultas de MLP")
    parser.add_argument("--learning_rate","-lr",type=float,default=0.01,help="Learning rate del optimizer")
    parser.add_argument("--dropout","-d",type=float,default=0.2,help="Dropout")
   
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    preprocessing = Preprocessing(frame_size=FRAME_SIZE,
                  hop = HOP, 
                  n_mels =N_MELS, 
                  n_fft = FRAME_SIZE, 
                  n_mfcc=N_MFCC,
                  samplerate=SAMPLERATE)
    device = DEVICE

    logging.info(f'Using device {device}')

    trainer = EngineRNN(input_size = N_MFCC+2+12,
                         hidden_size = args.hidden_size,
                         num_lstm_layers = args.num_lstm_layers,
                         num_mlp_layers = args.num_mlp_layers,
                         output_size = OUT_DIM, 
                         dropout = args.dropout,
                         batch_size = args.batch_size,
                         learning_rate = args.learning_rate,
                         preprocessing = preprocessing
                         )
    
    trainer.train(epochs = args.epochs,
                  patience=40,
                  delta = 0.01,
                  augmentation=False)