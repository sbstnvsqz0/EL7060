import torch

SEED=70
DATA_FOLDER = 'TiDigits'
MAX_SIZE = 26000
FRAME_SIZE = 1024
HOP = 512
N_MELS = 40
SAMPLERATE = 20000
N_MFCC = 13
OUT_DIM = 11

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")