import torch

DATA_FOLDER = "Podcast"
SAMPLERATE = 16000
FRAME_SIZE = 2048
HOP = 1024
N_MELS = 80
N_MFCC = 13
OUT_DIM = 3
SEED = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")