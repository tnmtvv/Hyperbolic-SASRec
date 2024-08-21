import os
import re
import optuna
from grids.core import generate_base_config
from grids.core.unrank import hypsasrec

# Get the file name without extension
file_name = os.path.basename(__file__).split('.')[0]
# Extract digits from the file name
RANK = int(re.search(r'\d+$', file_name).group())

def generate_config(trial: optuna.Trial) -> dict:
    trial_params = hypsasrec.trial_params
    fixed_params = {
        **hypsasrec.fixed_params,
        'hidden_units': RANK,
        'c': 0.01204150730389511
    }
    suggest = trial.suggest_categorical
    config = generate_base_config(trial_params, fixed_params, suggest)
    return config