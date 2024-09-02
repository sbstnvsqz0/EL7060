import optuna
from optuna.samplers import TPESampler
from src.engine.MLP import EngineMLP
import pandas as pd

def objective_function(trial):
    params_model ={"batch_size":trial.suggest_categorical("batch_size",[16,32,64,128]),
                   "hidden_dim":trial.suggest_int("hidden_dim",10,50),
                   "n_layers":trial.suggest_int("n_layers",1,5),
                   "learning_rate":trial.suggest_float("learning_rate",1e-5,1e-2,log=True),
                   "dropout":trial.suggest_categorical("dropout",[0.1,0.2,0.3,0.4,0.5])}

    model = EngineMLP(input_dim = 36,
                      output_dim = 11,
                      **params_model)
    
    model.train(5)

    return model.return_acc()

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",sampler=TPESampler())
    study.optimize(objective_function,n_trials=2)

    df = study.trials_dataframe()
    df.to_csv("optuna_study.csv",index=False)

    #with open(f"results.csv","a") as f:
    #   f.write("{best_value},{best_params}".format(best_value=study.best_value,best_params=study.best_params))
    #  f.write("\n")
    # f.close()

    