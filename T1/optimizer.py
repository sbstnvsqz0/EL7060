import optuna
from optuna.samplers import TPESampler
from src.engine.MLP import EngineMLP
from config import FRAME_SIZE, HOP, N_MELS, SAMPLERATE, N_MFCC, MAX_SIZE, OUT_DIM
from src.utils.Preprocessing import Preprocessing

def objective_function(trial):
    params_model ={"batch_size":trial.suggest_int("batch_size",4,64),
                   "hidden_dim":trial.suggest_int("hidden_dim",40,70),
                   "n_layers":trial.suggest_int("n_layers",1,3),
                   "learning_rate":trial.suggest_float("learning_rate",1e-4,1e-1,log=True),
                   "dropout":trial.suggest_categorical("dropout",[0.1,0.2,0.3,0.4,0.5])
                   }

    preprocessing = Preprocessing(frame_size=FRAME_SIZE,
                  hop = HOP, 
                  n_mels =N_MELS, 
                  n_fft = FRAME_SIZE, 
                  n_mfcc=N_MFCC,
                  samplerate=SAMPLERATE,
                  max_large=MAX_SIZE,
                  include_deltas=True)
    
    input_dim = N_MFCC*6 

    model = EngineMLP(input_dim = input_dim,
                      output_dim = OUT_DIM,
                      preprocessing = preprocessing,
                      **params_model)
    
    model.train(20,patience=10,
                  delta = 0.01)

    return model.return_acc()

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",sampler=TPESampler())
    study.optimize(objective_function,n_trials=100)

    df = study.trials_dataframe()
    df.to_csv("optuna_study.csv",index=False)



    