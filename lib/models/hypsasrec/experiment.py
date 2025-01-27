from functools import partial
import numpy as np
import torch
import torch.nn as nn

from lib.utils import fix_torch_seed, get_torch_device
from lib.models.base import RecommenderModel
from lib.models.learning import trainer
from lib.evaluation import Evaluator
from lib.models.sasrec.sampler import packed_sequence_batch_sampler
from geoopt import PoincareBallExact, optim

from .source import HypSASRec


def train_validate(config: dict, evaluator: Evaluator, model_save=False) -> None:
    dataset = evaluator.dataset
    n_items = len(dataset.item_index)
    fix_torch_seed(config.get('seed', None))
    dataset_name=config.get('dataset_name', None)
    model = HypSASRecModel(config, n_items)
    # print(f"one more curv: {model._model.c}")
    model.fit(dataset.train, evaluator)
    model_save = config.get('model_save', False)
    # num_items = model._model.scaler.cpu().detach().numpy().shape[0]
    num_items = model._model.item_emb.weight.cpu().detach().numpy().shape[0]
    print(num_items)
    if model_save:
        torch.save(model._model.state_dict(), f'/workspace/data/results/models/{dataset_name}_hypsasrec_model_state_dict_{num_items}.pt')

class HypSASRecModel(RecommenderModel):
    def __init__(self, config: dict, n_items: int):
        self.n_items = n_items
        self.config = config
        dataset_name = config.get('dataset_name', None)
        self.device = get_torch_device(self.config.pop('device', None))
        self._model = HypSASRec(self.config, self.n_items)
        self._model.epoch = 0

        # self._model = HypSASRec(self.config, self.n_items).to(self.device)
        # print(f"_model curv: {self._model.c}")
        # num_items = self._model.scaler.cpu().detach().numpy().shape[0]
        num_items = self._model.item_emb.weight.cpu().detach().numpy().shape[0]

        take_from_pretrained = self.config.pop('pretrained', False)
        print(f"pretrained: {take_from_pretrained}")
        if take_from_pretrained:
            pretrained_model_path= f"./data/results/models/dig_best_sasrec_model_state_dict_{num_items}.pt"
            # pretrained_model_path = f"./data/results/models/{dataset_name}_hypsasrec_model_state_dict_{num_items}.pt"
            pretrained_state_dict = torch.load(pretrained_model_path)

            self._model.load_state_dict(pretrained_state_dict, strict=False)

            for param in self._model.parameters():
                param.requires_grad = False

            ball = PoincareBallExact(c=torch.tensor(1.0), learnable=True)
            # print(f"pbe redefined{ball.k}")
            self._model.manifold = ball

        self._model = self._model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self._model.pad_token).to(self.device)

        # for name, param in self._model.named_parameters():
        #     print(name, param.requires_grad)

        # self.optimizer = torch.optim.Adam(
        #     self._model.parameters(), lr=self.config['learning_rate'], betas=(0.9, 0.98)
        # )
        self.optimizer = optim.RiemannianAdam(self._model.parameters(), lr=self.config['learning_rate'])
        # self.optimizer = optim.RiemannianAdam(filter(lambda p: p.requires_grad, self._model.parameters()), lr=self.config['learning_rate'])
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
        n_users = len(sizes) - 1
        self.users_to_track = np.sort(np.random.randint(n_users, size=100))
        self.sampler = packed_sequence_batch_sampler(
            indices, sizes, self.n_items,
            batch_size = self.batch_size,
            maxlen = self.config['maxlen'],
            seed = self.config['sampler_seed'],
        )

        self.n_batches = (len(sizes) - 1) // self.batch_size
        trainer(self, evaluator)

    def track_layer_norms_to_file(self, num_epoch, filename="layer_norms.txt"):
        with open(filename, "a") as f:  # Open the file in write mode
            for name, param in self.model.named_parameters():
                if param.grad is not None:  # Ensure gradients exist
                    grad_norm = torch.norm(param.grad).item()
                    weight_norm = torch.norm(param).item()
                    # Write the norms to the file
                    
                    f.write(f"Layer: {name}, Grad_Norm: {grad_norm}, Weight_Norm: {weight_norm},  Num_Epoch: {num_epoch}\n")
            f.write("\n")
    

    def train_epoch(self, epoch=None):
        model = self.model
        criterion, optimizer, sampler, device, n_batches = [
            getattr(self, a) for a in ['criterion', 'optimizer', 'sampler', 'device', 'n_batches']
        ]
        l2_emb = self.config['l2_emb']
        file_name = self.config.get('file_name', None)
        as_tensor = partial(torch.as_tensor, device=device)
        
        loss = 0
        cur_users = []
        model.train()
        for index in range(n_batches):
            usr, inputs, target, _ = next(sampler)
            cur_users.append(usr)
            # convert batch data into torch tensors
            inputs = as_tensor(inputs, dtype=torch.int32) # batch x seq.len
            target = as_tensor(target, dtype=torch.long)  # batch x seq.len, CrossEntropy requires `long` ints
            # need to permute output to comply with CrossEntropy inputs shape requirement
            logits = model(inputs).permute(0, 2, 1)  # batch x num.items x seq.len
            batch_loss = criterion(logits, target)
            batch_loss = batch_loss / self.gradient_accumulation_steps
            if l2_emb != 0:
                for param in model.item_emb.parameters():
                    batch_loss += l2_emb * torch.norm(param)**2
            batch_loss.retain_grad()
            batch_loss.backward()
            if (index + 1) == n_batches:
                if file_name:
                    self.track_layer_norms_to_file(epoch, file_name)
            if (index + 1) % self.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                loss += batch_loss.item()
        if model.geom == "ball":
            with open('./data/results/dig_curv_test.txt', 'a') as f:
                f.write(f"{model.manifold.c} \n")
        user_dict = dict(zip(np.hstack(cur_users), range(len(cur_users))))
        self.indices = [user_dict[user] for user in self.users_to_track]

        model.eval()
        return loss


    def predict(self, seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        with torch.no_grad():
            log_seqs = torch.full([maxlen], model.pad_token, dtype=torch.int64, device=device)
            log_seqs[-len(seq):] = torch.as_tensor(seq[-maxlen:], device=device)
            log_feats = model.log2feats(log_seqs)
            hyp_feats = model.manifold.expmap0(log_feats[:, -1, :]) # only use the final sequence state            
            logits = model.poincare_hyperplane(hyp_feats)
        return logits.cpu().numpy()

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
            log_seqs = torch.as_strided(pad_seq[-n_targets-maxlen:], (n_targets+1, maxlen), (1, 1)) # двигаем макслен 
            log_feats = model.log2feats(log_seqs)
            hyp_feats = model.manifold.expmap0(log_feats[:, -1, :]) # only use the final sequence state            
            logits = model.poincare_hyperplane(hyp_feats)
        return logits.cpu().numpy()
