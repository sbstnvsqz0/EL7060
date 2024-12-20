import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
from blocks import CustomRNN
from config import SEED, DEVICE, SAMPLERATE
from PodcastDataset import PodcastDataset
import os
from sklearn.metrics import mean_squared_error as mse

class EngineRNN:
    def __init__(self,input_size,hidden_size,num_rnn_layers,num_mlp_layers, output_size, dropout,batch_size,learning_rate,preprocessing):
        torch.manual_seed(SEED)
        self.model = CustomRNN(input_size = input_size,
                                hidden_size = hidden_size,
                                num_rnn_layers = num_rnn_layers,
                                num_mlp_layers = num_mlp_layers,
                                output_size = output_size, 
                                dropout=dropout).to(DEVICE)
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.optimizer = Adam(self.model.parameters(),lr=learning_rate) 
        self.scheduler = StepLR(self.optimizer,step_size=10,gamma=0.1)
        self.criterion = nn.MSELoss()
        #self.criterion = CCCLoss()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = np.inf


        try:
            os.makedirs("losses")
        except FileExistsError:
            # directory already exists
            pass

        
    def train(self,epochs,patience,delta,save_model=True,name="model"):

        ds = PodcastDataset("train",self.preprocessing)

        dataloader = DataLoader(ds,batch_size=self.batch_size,shuffle=True)
        dataloader_eval = DataLoader(PodcastDataset("validation",self.preprocessing),shuffle=False)
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
                    input,activation, valence, dominance,lengths = batch[0].float().to(DEVICE),batch[1].to(DEVICE),batch[2].to(DEVICE),batch[3].to(DEVICE), batch[4]
                    
                    pred = self.model(input,lengths)
                    real_values = torch.stack([activation,valence,dominance],dim=1).float()
                    

                    t_loss = self.criterion(pred,real_values)
                    train_loss +=t_loss.item()  #Acumulo losses por época
                    t_loss.backward()   #Propago loss en cada batch

                    self.optimizer.step()   #Step en cada batch
                    pbar.update(1)
            train_loss = train_loss/len(dataloader)
            self.train_losses.append(train_loss)

            activation_loss = 0
            valence_loss = 0
            dominance_loss = 0

            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader_eval, total=len(dataloader_eval),desc="Validación"):
                    input,activation, valence, dominance,lengths = batch[0].float().to(DEVICE),batch[1].to(DEVICE),batch[2].to(DEVICE),batch[3].to(DEVICE), batch[4]
                    pred = self.model(input,lengths).cpu()
                    real_values = torch.stack([activation,valence,dominance],dim=1).cpu()

                    val_loss += self.criterion(pred,real_values).item()

                    activation_loss += mse([pred[0][0].item()*6+1],[real_values[0][0].item()*6+1])
                    valence_loss += mse([pred[0][1].item()*6+1],[real_values[0][1].item()*6+1])
                    dominance_loss += mse([pred[0][2].item()*6+1],[real_values[0][2].item()*6+1])

                val_loss = val_loss/len(dataloader_eval)
                activation_loss = activation_loss/len(dataloader_eval)
                valence_loss = valence_loss/len(dataloader_eval)
                dominance_loss = dominance_loss/len(dataloader_eval)

            if epochs<11:
                self.scheduler.step()   #Step en cada época
            
            self.val_losses.append(val_loss)

            print("Epoca: {}, \tTrain_loss: {:.4f}, \tVal loss: {:.4f} \tAct_MSE: {:.4f} \tDom_MSE: {:.4f} \tVal_MSE: {:.4f}".format(epoch,train_loss,val_loss,activation_loss, dominance_loss, valence_loss))
            if val_loss < self.best_val_loss:
                self.save_model(name=name)
            if val_loss+delta<self.best_val_loss:
                self.best_val_loss = val_loss
                p = 0
            else:
                p+=1
            
            if p==patience:
                self.train_losses = self.train_losses
                self.val_losses = self.val_losses
                status = 1
                print("Patience alcanzada, terminando entrenamiento")
                break
        
        self.save_losses(name)
        if status == 0:
            print("El entrenamiento ha concluido dado que se llegó a las épocas")

    def evaluate(self,dataloader_eval,title="",return_outputs=False):
        self.model.eval()
        activation_loss = 0
        valence_loss = 0
        dominance_loss = 0
        real_output, predicted_output = {"activation":[],"dominance":[],"valence":[]},{"activation":[],"dominance":[],"valence":[]}
        with torch.no_grad():
            for batch in tqdm(dataloader_eval, total=len(dataloader_eval),desc="Evaluación"):
                input,activation, valence, dominance,lengths = batch[0].float().to(DEVICE),batch[1].to(DEVICE),batch[2].to(DEVICE),batch[3].to(DEVICE), batch[4]
                pred = self.model(input,lengths)
                real_values = torch.stack([activation,valence,dominance],dim=1)
                real_output["activation"].append(activation.item()*6+1); real_output["dominance"].append(dominance.item()*6+1); real_output["valence"].append(valence.item()*6+1);
                predicted_output["activation"].append(pred[0][0].item()*6+1); predicted_output["dominance"].append(pred[0][2].item()*6+1); predicted_output["valence"].append(pred[0][1].item()*6+1);#Hice cambios aqui
                activation_loss += self.criterion(pred[0][0]*6+1,real_values[0][0]*6+1)
                valence_loss += self.criterion(pred[0][1]*6+1,real_values[0][1]*6+1)
                dominance_loss += self.criterion(pred[0][2]*6+1,real_values[0][2]*6+1)

            activation_loss = activation_loss.cpu()/len(dataloader_eval)
            valence_loss = valence_loss.cpu()/len(dataloader_eval)
            dominance_loss = dominance_loss.cpu()/len(dataloader_eval)
            x = ["Activation","Dominance","Valence","Mean"]
            y = [activation_loss,dominance_loss,valence_loss,np.mean([activation_loss,dominance_loss,valence_loss])]
            plt.bar(x=x,height=y)
            for i in range(len(x)):
                plt.text(i, y[i].item()/2, np.round(y[i].item(),4), ha = 'center')
            plt.title(title)
            plt.xlabel("Variable Objetivo")
            plt.ylabel("MSE")
            plt.show()
            print(f"MSE Activation: {activation_loss}, MSE dominance: {dominance_loss}, MSE valence: {valence_loss}, MSE mean: {np.mean([activation_loss,dominance_loss,valence_loss])}")
        if return_outputs:
            return real_output, predicted_output
        
    def save_model(self,name):
        torch.save(self.model.state_dict(), name+".pth")

    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))

    def save_losses(self,name):
        try:
            os.makedirs("losses")
        except FileExistsError:
            # directory already exists
            pass
        df = pd.DataFrame(np.array([self.train_losses,self.val_losses]).T,columns=["train","val"])
        df.to_csv("losses/"+name+".csv",index=False)

