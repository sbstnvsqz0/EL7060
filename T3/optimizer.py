import optuna
from optuna.samplers import TPESampler
from src.engine.RNN import EngineRNN
from config import FRAME_SIZE, HOP, N_MELS, SAMPLERATE, N_MFCC, OUT_DIM
from src.utils.Preprocessing import Preprocessing

def objective_function(trial):
    params_model ={"batch_size":trial.suggest_categorical("batch_size",[16,32,64,80,96]),
                   "hidden_size":trial.suggest_categorical("hidden_size",[32,48,64,80,96,112,128]),
                   "num_rnn_layers":trial.suggest_int("num_rnn_layers",1,6),
                   "num_mlp_layers":trial.suggest_int("num_mlp_layers",1,2),
                   "learning_rate":trial.suggest_float("learning_rate",1e-5,1e-2,log=True),
                   "dropout":trial.suggest_categorical("dropout",[0.1,0.2,0.3])
                   }
    
    preprocessing = Preprocessing(frame_size=FRAME_SIZE,
                  hop = HOP, 
                  n_mels =N_MELS, 
                  n_fft = FRAME_SIZE, 
                  n_mfcc=N_MFCC,
                  samplerate=SAMPLERATE)
    
    input_dim = N_MFCC+2+N_MELS+12

    model = EngineRNN(input_size = input_dim,
                         output_size = OUT_DIM, 
                         preprocessing = preprocessing,
                         **params_model
                         )
    model.train(epochs=10,patience=5,delta=0.01,save_model=False)
    return model.best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize",sampler=TPESampler())
    study.optimize(objective_function,n_trials=50)

    df = study.trials_dataframe()
    df.to_csv("optuna_study.csv",index=False)

