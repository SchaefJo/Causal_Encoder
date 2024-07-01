import argparse
from collections import OrderedDict

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from experiments.datasets import iTHORDataset, VoronoiDataset
from models.biscuit_vae import BISCUITVAE
from models.shared.causal_encoder import CausalEncoder
import torch.utils.data as data
import sys
from datetime import datetime
import os
from models.biscuit_nf import BISCUITNF
import seaborn as sns

from models.shared.causal_mlp import CausalMLP


class RunnerCausalMLP():
    def __init__(self, num_train_epochs=100, train_prop=0.5, log_interval=10, dataset='ithor',
                 checkpoint_path='pretrained_models/causal_encoder/', disentangled=True):
        super().__init__()
        self.dataset = dataset
        self.num_train_epochs = num_train_epochs
        self.train_prop = train_prop
        self.log_interval = log_interval
        self.disentangled = disentangled
        current_time = datetime.now()
        self.date_time_str = current_time.strftime('%d.%m._%H:%M:%S')

        self.checkpoint_path = checkpoint_path

        self.cluster = False
        self.log_postfix = ''

    def train_network(self, pl_module, train_dataset, name):
        device = pl_module.device

        # We use one, sufficiently large network that predicts for any input all causal variables
        # To iterate over the different sets, we use a mask which is an extra input to the model
        # This is more efficient than using N networks and showed same results with large hidden size
        mlp = CausalMLP(hidden_dim=128,
                        lr=4e-3,
                        output_dim=18)
        optimizer = mlp.configure_optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]

        train_loader = data.DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=512)
        mlp.to(device)
        mlp.train()
        range_iter = range(self.num_train_epochs)
        if not self.cluster:
            range_iter = tqdm(range_iter, leave=False, desc=f'Training correlation encoder {self.log_postfix}')
        for epoch_idx in range_iter:
            avg_loss = 0.0
            for inps, latents in train_loader:
                inps = inps.to(device)
                latents = latents.to(device)
                #inps, latents = self._prepare_input(inps, target_assignment, latents)
                loss = mlp._get_loss(inps, latents)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            if epoch_idx % self.log_interval == 0:
                avg_loss /= len(train_loader)
                print(f'Epoch [{epoch_idx}/{self.num_train_epochs}], Loss: {avg_loss:.4f}')
        model_path = os.path.join(self.checkpoint_path, f'model_{name}_{self.date_time_str}.pth')
        torch.save(mlp.state_dict(), model_path)
        print(f'Model checkpoint saved at {self.checkpoint_path}')
        return mlp

    def test_model(self, pl_module, dataset, name):
        # Encode whole dataset with pl_module
        is_training = pl_module.training
        pl_module = pl_module.eval()
        loader = data.DataLoader(dataset, batch_size=256, drop_last=False, shuffle=False)
        all_encs, all_latents = [], []
        for batch in loader:
            inps, *_, latents = batch

            if self.dataset == 'ithor':
                inps = pl_module.autoencoder.encoder(inps)
            if self.disentangled:
                print('Latent space disentangled')
                encs = pl_module.encode(inps.to(pl_module.device)).cpu()
            else:
                print('Latent space entangled')
                if self.dataset == 'voronoi':
                    raise NotImplementedError('this is not implemented yet!')
                encs = inps
            all_encs.append(encs)
            all_latents.append(latents)
        all_encs = torch.cat(all_encs, dim=0)
        all_latents = torch.cat(all_latents, dim=0)
        # Normalize latents for stable gradient signals
        all_encs = (all_encs - all_encs.mean(dim=0, keepdim=True)) / all_encs.std(dim=0, keepdim=True).clamp(min=1e-2)
        # Create new tensor dataset for training (50%) and testing (50%)
        full_dataset = data.TensorDataset(all_encs, all_latents)
        train_size = int(self.train_prop * all_encs.shape[0])
        print(f'Train Size {train_size}')
        test_size = all_encs.shape[0] - train_size
        train_dataset, test_dataset = data.random_split(full_dataset,
                                                        lengths=[train_size, test_size],
                                                        generator=torch.Generator().manual_seed(42))

        encoder = self.train_network(pl_module, train_dataset, name)
        encoder.eval()

        # Record predictions of model on test and calculate distances
        test_inps, test_labels = all_encs[test_dataset.indices], all_latents[test_dataset.indices]
        #test_exp_inps, test_exp_labels = self._prepare_input(test_inps, target_assignment.cpu(), test_labels,
        #                                                     flatten_inp=False)
        pred = encoder.forward(test_inps.to(pl_module.device))

        loss = F.mse_loss(pred, test_labels)
        print(f'MSE loss: {loss.item()}')

        return loss


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'ithor':
        dataset = iTHORDataset(args.data_dir, split=args.split, single_image=True, return_targets=False,
                               return_latents=True)
        model = BISCUITNF.load_from_checkpoint(args.biscuit_checkpoint)
    else:
        dataset = VoronoiDataset(args.data_dir, split=args.split, single_image=True, return_targets=False,
                                 return_latents=True)
        model = BISCUITVAE.load_from_checkpoint(args.biscuit_checkpoint)
    model.to(device)
    model.freeze()
    _ = model.eval()

    causal_encode_runner = RunnerCausalMLP(num_train_epochs=args.max_epochs, log_interval=args.log_interval,
                                                      checkpoint_path=args.causal_encoder_output,
                                                      train_prop=args.train_prop, dataset=args.dataset,
                                                      disentangled=args.disentangled)

    loss = causal_encode_runner.test_model(model, dataset, 'first_R2')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/ithor/val_small/")
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--dataset', choices=['voronoi', 'ithor'], default='ithor')
    parser.add_argument('--biscuit_checkpoint', type=str,
                        default="../data/ithor/models/BISCUITNF_40l_64hid.ckpt")
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--causal_encoder_output', type=str,
                        default='output_causal_encoder/')
    parser.add_argument('--disentangled', type=bool, default=False)
    args = parser.parse_args()

    args.causal_encoder_output = os.path.join(args.causal_encoder_output,
                                              f'{args.dataset}_{args.split}_{args.train_prop}_{args.max_epochs}/')
    os.makedirs(os.path.dirname(args.causal_encoder_output), exist_ok=True)

    main(args)
