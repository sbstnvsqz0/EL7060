
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD 
from blocks import MLP
from Dataset import SignalDataset 
from src.utils.Preprocessing import Preprocessing
from config import SEED, FRAME_SIZE, HOP, N_MELS, SAMPLERATE, N_MFCC, DEVICE

preprocessing = Preprocessing(frame_size=FRAME_SIZE,
                  hop = HOP, 
                  n_mels =N_MELS, 
                  n_fft = FRAME_SIZE, 
                  n_mfcc=N_MFCC,
                  samplerate=SAMPLERATE)

#TODO: gpu
class EngineMLP:
    def __init__(self,input_dim,hidden_dim,output_dim,n_layers,batch_size,learning_rate,dropout):
        torch.manual_seed(SEED)
        self.model = MLP(input_dim,hidden_dim,output_dim,n_layers,dropout)
        self.batch_size = batch_size
        self.optimizer = SGD(self.model.parameters(),lr=learning_rate) 
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        
    def train(self,epochs):
        dataloader = DataLoader(SignalDataset("train",preprocessing),batch_size=self.batch_size,shuffle=False)
        dataloader_eval = DataLoader(SignalDataset("validation",preprocessing),shuffle=False)
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            val_loss = 0
            acc = 0 
            with tqdm(total=len(dataloader),desc=f'Epoca {epoch}/{epochs}',unit="batch") as pbar:
                for batch in dataloader:
                    self.optimizer.zero_grad()
                    input,label = batch[0].float(),batch[1].type(torch.uint8)
                    pred = self.model(input)
                    label_pred = torch.argmax(pred,dim=1)

                    
                    t_loss = self.criterion(pred,label)
                    train_loss +=t_loss.item()  #Acumulo losses por época
                    t_loss.backward()   #Propago loss en cada batch

                    self.optimizer.step()   #Step en cada batch
                    pbar.update(1)
            train_loss = train_loss/len(dataloader)
            self.train_losses.append(train_loss)

            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader_eval, total=len(dataloader_eval),desc="Validación"):
                    input,label = batch[0].float(),batch[1].type(torch.uint8)
                    pred = self.model(input)
                    label_pred = torch.argmax(pred,dim=1)

                    val_loss += self.criterion(pred,label).item()
                    acc += (label_pred==label).sum()
                
                val_loss = val_loss/len(dataloader_eval)
                acc = acc/len(dataloader_eval)

            self.val_losses.append(val_loss)
            print("Epoca: {}, \tTrain_loss: {:.4f}, \tVal loss: {:.4f}, \tAcc: {:.4f}".format(epoch,train_loss,val_loss,acc))
            if acc >0.93: 
                print("El entrenamiento ha concluido, ya que se llegó a la accuracy pedida")
                break
        print("El entrenamiento ha concluido las épocas sin llegar al accuracy pedido")

    def return_losses(self):
        return self.train_losses, self.val_losses