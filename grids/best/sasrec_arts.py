import optuna
from grids.core import generate_best_config


def generate_config(trial: optuna.Trial) -> dict:

    fixed_params = dict(
        batch_size=512,
        learning_rate=0.005,
        hidden_units=64,
        num_blocks=2,
        dropout_rate=0.6,
        num_heads=1,
        l2_emb=0.0,
        maxlen=50,
        batch_quota=None,
        seed=0,
        sampler_seed=789,
        device="cuda",
        max_epochs=80,
        geom="ball",
        bias=True,
        # c=0.02890294854523254,  # obtained from SVD embeddings at 1e^{-12} tolerance
        c=1.0,
        model_save=False,
        pretrained=False,
        train_curv=True
    )
    config = generate_best_config(fixed_params)
    return config
