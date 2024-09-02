import torch

SEED=70
DATA_FOLDER = 'TiDigits'
MAX_SIZE = 60000
FRAME_SIZE = 2048
HOP = 256
N_MELS = 24
SAMPLERATE = 20000
N_MFCC = 12
OUT_DIM = 11

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")