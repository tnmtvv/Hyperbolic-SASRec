from functools import partial

import numpy as np
import torch

from lib.models.sasrec.sampler import packed_sequence_batch_sampler
from lib.utils import fix_torch_seed, get_torch_device
from lib.models.base import RecommenderModel
from lib.models.learning import trainer
from lib.evaluation import Evaluator
from .source import SASRecCE


def train_validate(config: dict, evaluator: Evaluator, model_save=False) -> None:
    dataset = evaluator.dataset
    n_items = len(dataset.item_index)
    fix_torch_seed(config.get('seed', None))
    model = SASRecCEModel(config, n_items)
    model.fit(dataset.train, evaluator)
    num_items = model._model.item_emb.weight.cpu().detach().numpy().shape[0]
    model_save = config.get('model_save', False)
    dataset_name = config.get('dataset_name', False)
    print(f"model_save: {model_save}")
    if model_save:
        torch.save(model._model.state_dict(), f'./data/results/models/{dataset_name}_best_sasrecb_model_state_dict_{num_items}.pt')


class SASRecCEModel(RecommenderModel):
    def __init__(self, config: dict, n_items: int):
        self.n_items = n_items
        self.config = config
        self.device = get_torch_device(self.config.pop('device', None))
        self._model = SASRecCE(self.config, self.n_items).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self._model.pad_token).to(self.device)
        self.optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.config['learning_rate'], betas=(0.9, 0.98)
        )
        self.sampler = None
        self.n_batches = None
        self.batch_size = None
        self.gradient_accumulation_steps = None
        self.set_batch_size()

    def set_batch_size(self):
        quota = self.config.get('batch_quota', None)
        batch_size = self.config['batch_size']
        if (not quota) or (quota >= batch_size):
            self.batch_size = batch_size
            self.gradient_accumulation_steps = 1
        elif batch_size % quota == 0:
            self.batch_size = quota
            self.gradient_accumulation_steps = batch_size // quota
        else:
            raise NotImplementedError

    @property
    def model(self):
        return self._model

    def fit(self, data: tuple, evaluator: Evaluator):
        indices, sizes = data
        self.sampler = packed_sequence_batch_sampler(
            indices, sizes, self.n_items,
            batch_size = self.batch_size,
            maxlen = self.config['maxlen'],
            seed = self.config['sampler_seed'],
        )

        self.n_batches = (len(sizes) - 1) // self.batch_size
        trainer(self, evaluator)


    def predict(self, seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        with torch.no_grad():
            log_seqs = torch.full([maxlen], model.pad_token, dtype=torch.int64, device=device)
            log_seqs[-len(seq):] = torch.as_tensor(seq[-maxlen:], device=device)
            log_feats = model.log2feats(log_seqs)
            logits = model.head(log_feats[:, -1, :])
        return logits.cpu().numpy().squeeze()

    def predict_sequential(self, target_seq, seen_seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        n_seen = len(seen_seq)
        n_targets = len(target_seq)
        seq = np.concatenate([seen_seq, target_seq])

        with torch.no_grad():
            pad_seq = torch.as_tensor(
                np.pad(
                    seq, (max(0, maxlen-n_seen), 0),
                    mode = 'constant',
                    constant_values = model.pad_token
                ),
                dtype = torch.int64,
                device = device
            )
            log_seqs = torch.as_strided(pad_seq[-n_targets-maxlen:], (n_targets+1, maxlen), (1, 1))
            log_feats = model.log2feats(log_seqs)
            logits = model.head(log_feats[:, -1, :])
        return logits.cpu().numpy()
        
