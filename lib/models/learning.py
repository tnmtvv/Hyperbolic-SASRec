import importlib
from collections.abc import Callable
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

from lib import defaults
from .base import RecommenderModel
from lib.evaluation import Evaluator
from lib.optunatools import EarlyStopping

import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_expmap(tensor, save_dir, epoch, step_size=100, indices=None, seq_to_plot=None):
    """
    Visualize the exponential map distribution for every `step_size` sequence.

    Args:
        tensor: Tensor of shape (num_seq, seq_length, emb_dim=2).
        save_dir: Directory to save the plots.
        epoch: Current epoch number.
        step_size: Interval for selecting sequences (e.g., every 100th sequence).
    """
    # Ensure embedding dimension is 2
    num_seq, seq_length, emb_dim = tensor.shape
    # assert emb_dim == 2, "Visualization is only supported for emb_dim=2."

    # Create directory for saving plots
    os.makedirs(save_dir, exist_ok=True)

    # Convert tensor to numpy for plotting
    tensor_np = tensor.detach().cpu().numpy()  # Shape: (num_seq, seq_length, emb_dim)

    # Create a new plot
    plt.figure(figsize=(15, 15))

    if indices is not None:
        selected_indices = indices
        num_selected_sequences = len(selected_indices)        
        colors = plt.cm.rainbow(np.linspace(0, 1, num_selected_sequences))  # Assign distinct colors

    # Plot every `step_size`-th sequence
    if seq_to_plot is not None:
        sequence = tensor_np[seq_to_plot]

        plt.scatter(sequence[:, 0], sequence[:, 1], marker='o', color=colors[0], alpha=0.3)
        
    else:
        for idx, seq_idx in enumerate(range(0, num_seq, step_size)):
            sequence = tensor_np[seq_idx]  # Extract the sequence (shape: [seq_length, emb_dim])
            sequence_reshaped = sequence.reshape(-1, emb_dim)  # Flatten sequences into shape (num_seq * seq_length, emb_dim)
            U, S, _ = np.linalg.svd(sequence_reshaped, full_matrices=False)

            # Keep only the top 2 singular values/vectors for low-rank representation
            low_rank_sequence = np.dot(U[:, :2], np.diag(S[:2]))

            # Plot the sequence points with a unique color
            plt.scatter(low_rank_sequence[:, 0], low_rank_sequence[:, 1], marker='o', color=colors[idx], alpha=0.3)

    # Add labels and legend
    plt.title(f"Exponential Map Distribution at Epoch {epoch}", fontsize = 40)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    
    # Save the plot to a file
    save_path = os.path.join(save_dir, f"expmap_distribution_epoch_{str(epoch).zfill(3)}.png")
    plt.savefig(save_path)
    
    plt.close()


    
def trainer(model: RecommenderModel, evaluator: Evaluator) -> None:
    '''
    A standard boilerplate for training a neural network.
    ====================================================

    Supports pruning/reporting based on callbacks. Callback
    must be available in the `evaluator`.
    '''
    max_epochs = model.config['max_epochs']
    validation_interval = defaults.validation_interval
    assert validation_interval <= max_epochs, 'Number of epochs is too small. Won\'t validate.'
    # TODO warn if max_epochs % validation_interval != 0
    for epoch in range(max_epochs):
        model._model.epoch = epoch
        #loss = model.train_epoch(epoch)
        loss = model.train_epoch()
        hyp_embs = torch.cat(model._model.hyp_embs, dim=0)
        seq_to_plot = 10
        visualize_expmap(hyp_embs, "./expmap0_SVD_32_one_color_constant_curv", epoch, 100)
        model._model.hyp_embs = []
        if (epoch + 1) % validation_interval == 0:
            try:
                evaluator.submit(model, step=epoch, args=(loss,))
            except EarlyStopping:
                break


def import_model_routine(model_name: str, name: str) -> Callable[..., None]:
    try:
        module_name = f'lib.models.{model_name.lower()}'
    except KeyError:    
        raise ValueError(f'Unrecognized model name {model_name}.')
    module = importlib.import_module(module_name)
    return getattr(module, name)


def train_data_format(model_name):
    sparse = ['svd', 'random', 'mostpop']
    sequential_packed = ['sasrec', 'sasrecb', 'hypsasrec', 'hypsasrecb']
    sequential = []
    sequential_typed = []
    if model_name in sparse:
        return 'sparse'
    if model_name in sequential:
        return 'sequential' # pandas Series
    if model_name in sequential_packed:
        return 'sequential_packed' # csr-like format
    if model_name in sequential_typed:
        return 'sequential_typed' # numba dict
    return 'default'

def test_data_format(args):
    if args.next_item_only: # e.g., lorentzfm is not suitable for sequential prediction
        return ('interactions', dict(stepwise=True, max_steps=1))
    return 'sequential' # 'sequential' will enable vectorized evaluation, if a model supports it