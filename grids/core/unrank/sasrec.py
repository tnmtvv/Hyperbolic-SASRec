trial_params = dict(
    batch_size = [128, 256, 512],
    learning_rate = [0.005, 0.001, 0.0005],
    num_blocks = [1, 2, 3],
    dropout_rate = [0.2, 0.4, 0.6],
    # с = [0.01204150730389511, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
)

#config for grocery
fixed_params = dict(
    num_heads = 1,
    l2_emb = 0.0,
    maxlen = None,
    batch_quota = None,
    seed = 0,
    sampler_seed = 789,
    device = 'cuda',
    max_epochs = 400,
    # step_replacement = 'max_epochs', # during tests, max value will be replaced with an optimal one
)