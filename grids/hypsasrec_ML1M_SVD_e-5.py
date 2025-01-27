import optuna
from grids.core import hypsasrec, generate_base_config


def generate_config(trial: optuna.Trial) -> dict:
    trial_params = hypsasrec.trial_params
    fixed_params = {
        **hypsasrec.fixed_params,
        # overwrite c values if set;
        # based on https://canyon-indigo-809.notion.site/b1a23bafebc846a4b2c87d73c7a4b943
        # 'c': 0.08226903980753697, # obtained from SVD embeddings at 1e^{-5} tolerance
        'c': 1.0
    }
    suggest = trial.suggest_categorical
    config = generate_base_config(trial_params, fixed_params, suggest)
    return config