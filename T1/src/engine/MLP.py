import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CyclicLR
from blocks import MLP
from Dataset import SignalDataset 
from config import SEED, DEVICE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class EngineMLP:
    def __init__(self,input_dim,hidden_dim,output_dim,n_layers,batch_size,learning_rate,dropout,preprocessing,):
        torch.manual_seed(SEED)
        self.model = MLP(input_dim,hidden_dim,output_dim,n_layers,dropout).to(DEVICE)
        self.batch_size = batch_size
        #self.optimizer = SGD(self.model.parameters(),lr=learning_rate,momentum=0.9) 
        self.preprocessing = preprocessing
        self.optimizer = Adam(self.model.parameters(),lr=learning_rate)
        self.scheduler = CyclicLR(self.optimizer,base_lr=0.1*learning_rate,max_lr=learning_rate,step_size_up=1000)
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = np.inf
        self.acc = 0
        self.best_acc = 0
        
    def train(self,epochs,patience,delta,save_model=True):
        dataloader = DataLoader(SignalDataset("train",self.preprocessing),batch_size=self.batch_size,shuffle=False)
        dataloader_eval = DataLoader(SignalDataset("validation",self.preprocessing),shuffle=False)
        p = 0
        status = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            val_loss = 0
            acc = 0 
            with tqdm(total=len(dataloader),desc=f'Epoca {epoch}/{epochs}',unit="batch") as pbar:
                for batch in dataloader:
                    self.optimizer.zero_grad()
                    input,label = batch[0].float().to(DEVICE),batch[1].type(torch.uint8).to(DEVICE)
                    pred = self.model(input)
                    label_pred = torch.argmax(pred,dim=1)

                    
                    t_loss = self.criterion(pred,label)
                    train_loss +=t_loss.item()  #Acumulo losses por época
                    t_loss.backward()   #Propago loss en cada batch

                    self.optimizer.step()   #Step en cada batch
                    self.scheduler.step()
                    pbar.update(1)
            train_loss = train_loss/len(dataloader)
            self.train_losses.append(train_loss)

            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader_eval, total=len(dataloader_eval),desc="Validación"):
                    input,label = batch[0].float().to(DEVICE),batch[1].type(torch.uint8).to(DEVICE)
                    pred = self.model(input)
                    label_pred = torch.argmax(pred,dim=1)

                    val_loss += self.criterion(pred,label).item()
                    acc += (label_pred==label).sum()
                
                val_loss = val_loss/len(dataloader_eval)
                acc = acc/len(dataloader_eval)
                self.acc = acc
            
            
            self.val_losses.append(val_loss)

            print("Epoca: {}, \tTrain_loss: {:.4f}, \tVal loss: {:.4f}, \tAcc: {:.4f}".format(epoch,train_loss,val_loss,acc))
            if acc > self.best_acc:
                self.best_acc = acc
                if save_model:
                    self.save_model()
            if val_loss+delta<self.best_val_loss:
                self.best_val_loss = val_loss
                self.acc = acc
                p = 0
            else:
                p+=1
            
            if p==patience:
                self.train_losses = self.train_losses[:-patience]
                self.val_losses = self.val_losses[:-patience]
                status = 1
                print("Patience alcanzada, terminando entrenamiento")
                break
        if status == 0:
            print("El entrenamiento ha concluido dado que se llegó a las épocas")

    def evaluate(self,dataloader):
        self.model.eval()
        acc = 0
        real_labels = []
        pred_labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader),desc="Evaluación"):
                input,label = batch[0].float().to(DEVICE),batch[1].type(torch.uint8).to(DEVICE)
                
                pred = self.model(input)
                label_pred = torch.argmax(pred,dim=1)
                real_labels.append(label.cpu())
                pred_labels.append(label_pred.cpu())
                acc += (label_pred==label).sum()
            acc = acc/len(dataloader)
            conf_matrix = confusion_matrix(y_true = real_labels,
                                             y_pred = pred_labels,
                                             normalize='true')
            sns.heatmap(conf_matrix,annot=True,cmap="summer")
            plt.xlabel("Valores predichos")
            plt.ylabel("Valores reales")
            plt.show()
            print(f"Accuracy: {acc}")
        
    def save_model(self):
        torch.save(self.model.state_dict(), "model.pth")

    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))

    def return_losses(self):
        return [self.train_losses, self.val_losses]
    
    def return_acc(self):
        return self.acc
