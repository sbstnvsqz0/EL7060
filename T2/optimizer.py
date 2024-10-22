import optuna
from optuna.samplers import TPESampler
from src.engine.LSTM import EngineLSTM
from config import FRAME_SIZE, HOP, N_MELS, SAMPLERATE, N_MFCC, OUT_DIM
from src.utils.Preprocessing import Preprocessing

def objective_function(trial):
    params_model ={"batch_size":trial.suggest_int("batch_size",4,64),
                   "hidden_size":trial.suggest_int("hidden_size",32,128),
                   "num_lstm_layers":trial.suggest_int("num_lstm_layers",1,6),
                   "num_mlp_layers":trial.suggest_int("num_mlp_layers",1,3),
                   "learning_rate":trial.suggest_float("learning_rate",1e-5,1e-2,log=True),
                   "dropout":trial.suggest_categorical("dropout",[0.1,0.2,0.3])
                   }
    
    preprocessing = Preprocessing(frame_size=FRAME_SIZE,
                  hop = HOP, 
                  n_mels =N_MELS, 
                  n_fft = FRAME_SIZE, 
                  n_mfcc=N_MFCC,
                  samplerate=SAMPLERATE)
    
    input_dim = N_MFCC+N_MELS+2

    model = EngineLSTM(input_size = input_dim,
                         output_size = OUT_DIM, 
                         preprocessing = preprocessing,
                         **params_model
                         )
    model.train(epochs=10,patience=5,delta=0.01,save_model=False,augmentation=False)
    return model.best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize",sampler=TPESampler())
    study.optimize(objective_function,n_trials=200)

    df = study.trials_dataframe()
    df.to_csv("optuna_study.csv",index=False)

