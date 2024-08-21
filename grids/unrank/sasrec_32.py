import optuna
from grids.core import generate_base_config
from grids.core.unrank import sasrec 

RANK = 32

def generate_config(trial: optuna.Trial) -> dict:
    trial_params = sasrec.trial_params
    fixed_params = {**sasrec.fixed_params, 'hidden_units': RANK}
    suggest = trial.suggest_categorical
    config = generate_base_config(trial_params, fixed_params, suggest)
    return config
