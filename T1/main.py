import argparse
import logging
from config import DEVICE, OUT_DIM
from src.engine.MLP import EngineMLP

def get_args():
    parser = argparse.ArgumentParser('Entrenar la red')
    parser.add_argument("--epochs","-e",metavar="E",type=int,default=100,help="Número de épocas que se entrena la red")
    parser.add_argument("--batch_size","-b",metavar="B",type=int,default=32,help="Batch size")
    parser.add_argument("--hidden_dim","-hd",type=int,default=64,help="Número de neuronas en capas ocultas")
    parser.add_argument("--n_layers","-lay",type=int,default=5,help="Número de capas ocultas de la MLP")
    parser.add_argument("--learning_rate","-lr",type=float,default=0.01,help="Learning rate del optimizer")
    parser.add_argument("--dropout","-d",type=float,default=0.2,help="Dropout")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = DEVICE
    logging.info(f'Using device {device}')

    trainer = EngineMLP(input_dim = 24,
                        hidden_dim = args.hidden_dim,
                        output_dim = OUT_DIM,
                        n_layers = args.n_layers,
                        batch_size = args.batch_size,
                        learning_rate = args.learning_rate,
                        dropout = args.dropout)
    
    trainer.train(epochs = args.epochs)