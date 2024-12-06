import torch

DATA_FOLDER = "Podcast"
SAMPLERATE = 16000
FRAME_SIZE = 2048
HOP = 1024
N_MELS = 80
N_MFCC = 13
OUT_DIM = 3
SEED = 3
MEANS = {"activation":4.4851, "dominance":4.5854,"valence":3.9211}
STDS = {"activation":0.9946 , "dominance":0.9531,"valence":1.0403}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")