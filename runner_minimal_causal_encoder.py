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

class RunnerMinimalCausalEncoder():
    def __init__(self, num_train_epochs=100, train_prop=0.5, log_interval=10, dataset='ithor',
                 checkpoint_path='pretrained_models/causal_encoder/'):
        super().__init__()
        self.dataset = dataset
        self.num_train_epochs = num_train_epochs
        self.train_prop = train_prop
        self.log_interval = log_interval
        current_time = datetime.now()
        self.date_time_str = current_time.strftime('%d.%m._%H:%M:%S')

        self.checkpoint_path = checkpoint_path

        self.cluster = False
        self.log_postfix = ''

    def train_network(self, pl_module, train_dataset, target_assignment, name):
        device = pl_module.device
        if hasattr(pl_module, 'causal_encoder') and pl_module.causal_encoder is not None:
            causal_var_info = pl_module.causal_encoder.hparams.causal_var_info
        else:
            causal_var_info = pl_module.hparams.causal_var_info
        # We use one, sufficiently large network that predicts for any input all causal variables
        # To iterate over the different sets, we use a mask which is an extra input to the model
        # This is more efficient than using N networks and showed same results with large hidden size
        encoder = CausalEncoder(c_hid=128,
                                lr=4e-3,
                                causal_var_info=causal_var_info,
                                single_linear=True,
                                c_in=pl_module.hparams.num_latents * 2,
                                warmup=0)
        optimizer, _ = encoder.configure_optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]

        train_loader = data.DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=512)
        target_assignment = target_assignment.to(device)
        encoder.to(device)
        encoder.train()
        range_iter = range(self.num_train_epochs)
        if not self.cluster:
            range_iter = tqdm(range_iter, leave=False, desc=f'Training correlation encoder {self.log_postfix}')
        for epoch_idx in range_iter:
            avg_loss = 0.0
            for inps, latents in train_loader:
                inps = inps.to(device)
                latents = latents.to(device)
                inps, latents = self._prepare_input(inps, target_assignment, latents)
                loss = encoder._get_loss([inps, latents], mode=None)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            if epoch_idx % self.log_interval == 0:
                avg_loss /= len(train_loader)
                print(f'Epoch [{epoch_idx}/{self.num_train_epochs}], Loss: {avg_loss:.4f}')
        model_path = os.path.join(self.checkpoint_path, f'model_{name}_{self.date_time_str}.pth')
        torch.save(encoder.state_dict(), model_path)
        print(f'Model checkpoint saved at {self.checkpoint_path}')
        return encoder

    def test_model(self, pl_module, dataset, name):
        # Encode whole dataset with pl_module
        is_training = pl_module.training
        pl_module = pl_module.eval()
        loader = data.DataLoader(dataset, batch_size=256, drop_last=False, shuffle=False)
        all_encs, all_latents = [], []
        for batch in loader:
            inps, *_, latents = batch
            # if multiple images per batch, then train only on last image
            # if len(inps.shape) == 5:
            #    inps = inps[:, -1, :, :, :]
            if self.dataset == 'ithor':
                inps = pl_module.autoencoder.encoder(inps)
            encs = pl_module.encode(inps.to(pl_module.device)).cpu()
            all_encs.append(encs)
            all_latents.append(latents)
        all_encs = torch.cat(all_encs, dim=0)
        all_latents = torch.cat(all_latents, dim=0)
        # Normalize latents for stable gradient signals
        all_encs = (all_encs - all_encs.mean(dim=0, keepdim=True)) / all_encs.std(dim=0, keepdim=True).clamp(min=1e-2)
        # Create new tensor dataset for training (50%) and testing (50%)
        full_dataset = data.TensorDataset(all_encs, all_latents)
        train_size = int(self.train_prop * all_encs.shape[0])
        test_size = all_encs.shape[0] - train_size
        train_dataset, test_dataset = data.random_split(full_dataset,
                                                        lengths=[train_size, test_size],
                                                        generator=torch.Generator().manual_seed(42))
        # Train network to predict causal factors from latent variables
        if hasattr(pl_module, 'target_assignment') and pl_module.target_assignment is not None:
            target_assignment = pl_module.target_assignment.clone()
        else:
            target_assignment = torch.eye(all_encs.shape[-1])
        encoder = self.train_network(pl_module, train_dataset, target_assignment, name)
        encoder.eval()
        # Record predictions of model on test and calculate distances
        test_inps, test_labels = all_encs[test_dataset.indices], all_latents[test_dataset.indices]
        test_exp_inps, test_exp_labels = self._prepare_input(test_inps, target_assignment.cpu(), test_labels,
                                                             flatten_inp=False)
        pred_dict = encoder.forward(test_exp_inps.to(pl_module.device))
        for key in pred_dict:
            pred_dict[key] = pred_dict[key].cpu()
        _, dists, norm_dists = encoder.calculate_loss_distance(pred_dict, test_exp_labels)
        # Calculate statistics (R^2, pearson, etc.)
        avg_norm_dists, r2_matrix = self.log_R2_statistic(encoder, test_labels, norm_dists, pl_module, name)
        # self.log_Spearman_statistics(trainer, encoder, pred_dict, test_labels, pl_module=pl_module)
        if is_training:
            pl_module = pl_module.train()
        return r2_matrix

    def log_R2_statistic(self, encoder, test_labels, norm_dists, pl_module, name):
        avg_pred_dict = OrderedDict()
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            gt_vals = test_labels[..., i]
            if var_info.startswith('continuous'):
                avg_pred_dict[var_key] = gt_vals.mean(dim=0, keepdim=True).expand(gt_vals.shape[0], )
            elif var_info.startswith('angle'):
                avg_angle = torch.atan2(torch.sin(gt_vals).mean(dim=0, keepdim=True),
                                        torch.cos(gt_vals).mean(dim=0, keepdim=True)).expand(gt_vals.shape[0], )
                avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2 * np.pi, avg_angle)
                avg_pred_dict[var_key] = torch.stack([torch.sin(avg_angle), torch.cos(avg_angle)], dim=-1)
            elif var_info.startswith('categ'):
                gt_vals = gt_vals.long()
                mode = torch.mode(gt_vals, dim=0, keepdim=True).values
                avg_pred_dict[var_key] = F.one_hot(mode, int(var_info.split('_')[-1])).float().expand(gt_vals.shape[0],
                                                                                                      -1)
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in R2 statistics.'
        _, _, avg_norm_dists = encoder.calculate_loss_distance(avg_pred_dict, test_labels, keep_sign=True)

        r2_matrix = []

        for var_key in encoder.hparams.causal_var_info:
            ss_res = (norm_dists[var_key] ** 2).mean(dim=0)
            ss_tot = (avg_norm_dists[var_key] ** 2).mean(dim=0, keepdim=True)
            ss_tot = torch.where(ss_tot == 0.0, torch.tensor(1.0, device=ss_tot.device), ss_tot)
            r2 = 1 - ss_res / ss_tot
            r2_matrix.append(r2)
        r2_matrix = [r2.detach() for r2 in r2_matrix]
        r2_matrix = torch.stack(r2_matrix, dim=-1).cpu().numpy()
        self.log_matrix(r2_matrix, name)
        self._log_heatmap(values=r2_matrix,
                         tag='r2_matrix',
                         title='R^2 Matrix',
                         xticks=[key for key in encoder.hparams.causal_var_info],
                         pl_module=pl_module, name=name)
        return avg_norm_dists, r2_matrix

    def _log_heatmap(self, values, tag, title=None, xticks=None, yticks=None, xlabel=None, ylabel=None, pl_module=None,
                     name=''):
        if ylabel is None:
            ylabel = 'Target dimension'
        if xlabel is None:
            xlabel = 'True causal variable'
        if yticks is None:
            yticks = [f'Dim {i+1}' for i in range(values.shape[0])]
        if xticks is None:
            xticks = self.dataset.target_names()
        fig = plt.figure(figsize=(min(6, max(4, values.shape[1]/1.25)),
                                  min(6, max(4, values.shape[0]/1.25))),
                         dpi=150)
        sns.heatmap(values, annot=min(values.shape) < 10,
                    yticklabels=yticks,
                    xticklabels=xticks,
                    vmin=0.0,
                    vmax=1.0,
                    fmt='3.2f')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        fig.tight_layout()
        heatmap_path = os.path.join(self.checkpoint_path, f'heatmap_{name}_{self.date_time_str}.png')
        plt.savefig(heatmap_path)
        plt.close(fig)

        # if values.shape[0] == values.shape[1] + 1:
        #     values = values[:-1]
        #
        # if values.shape[0] == values.shape[1]:
        #     avg_diag = np.diag(values).mean()
        #     max_off_diag = (values - np.eye(values.shape[0]) * 10).max(axis=-1).mean()
        #     if pl_module is None:
        #         trainer.logger.experiment.add_scalar(f'corr_callback_{tag}_diag{self.log_postfix}', avg_diag, global_step=trainer.global_step)
        #         trainer.logger.experiment.add_scalar(f'corr_callback_{tag}_max_off_diag{self.log_postfix}', max_off_diag, global_step=trainer.global_step)
        #     else:
        #         pl_module.log(f'corr_callback_{tag}_diag{self.log_postfix}', avg_diag)
        #         pl_module.log(f'corr_callback_{tag}_max_off_diag{self.log_postfix}', max_off_diag)

    def _prepare_input(self, inps, target_assignment, latents, flatten_inp=True):
        ta = target_assignment.detach()[None, :, :].expand(inps.shape[0], -1, -1)
        inps = torch.cat([inps[:, :, None] * ta, ta], dim=-2).permute(0, 2, 1)
        latents = latents[:, None].expand(-1, inps.shape[1], -1)
        if flatten_inp:
            inps = inps.flatten(0, 1)
            latents = latents.flatten(0, 1)
        return inps, latents

    def log_matrix(self, matrix, name):
        """ Saves a numpy array to the logging directory """
        filename = os.path.join(self.checkpoint_path, f'{name}_{self.date_time_str}.pth')

        new_epoch = np.array([self.num_train_epochs])
        new_val = matrix[None]
        if os.path.isfile(filename):
            prev_data = np.load(filename)
            epochs, values = prev_data['epochs'], prev_data['values']
            for i in [1, 2]:
                if values.shape[i] > new_val.shape[i]:
                    pad_shape = list(values.shape)
                    pad_shape[i] -= new_val.shape[i]
                    new_val = np.concatenate([new_val, np.zeros(pad_shape, dtype=new_val.dtype)], axis=i)
                elif values.shape[i] < new_val.shape[i]:
                    pad_shape = list(new_val.shape)
                    pad_shape[i] -= values.shape[i]
                    values = np.concatenate([values, np.zeros(pad_shape, dtype=values.dtype)], axis=i)
            epochs = np.concatenate([epochs, new_epoch], axis=0)
            values = np.concatenate([values, new_val], axis=0)
        else:
            epochs = new_epoch
            values = new_val
        np.savez_compressed(filename, epochs=epochs, values=values)


    def first_r2_matrix(self, model, dataset, name):
        r2 = self.test_model(model, dataset, name)
        np.set_printoptions(precision=6, suppress=True)
        r2_matrix = torch.from_numpy(r2)
        # Assign each latent to the causal variable with the highest (relative) correlation
        r2_matrix = r2_matrix / r2_matrix.abs().max(dim=0, keepdims=True).values.clamp(min=0.1)
        max_r2 = r2_matrix.argmax(dim=-1)
        ta = F.one_hot(max_r2, num_classes=r2_matrix.shape[-1]).float()
        # Group multi-dimensional causal variables together
        if isinstance(dataset, iTHORDataset):
            ta = torch.cat([ta[:, :1],
                            ta[:, 1:7].sum(dim=-1, keepdims=True),
                            ta[:, 7:9],
                            ta[:, 9:13].sum(dim=-1, keepdims=True),
                            ta[:, 13:]], dim=-1)

        model.target_assignment = ta
        model.last_target_assignment.data = ta


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

    causal_encode_runner = RunnerMinimalCausalEncoder(num_train_epochs=args.max_epochs, log_interval=args.log_interval,
                                                      checkpoint_path=args.causal_encoder_output,
                                                      train_prop=args.train_prop, dataset=args.dataset)

    r2_start = causal_encode_runner.first_r2_matrix(model, dataset, 'first_R2')
    r2_end = causal_encode_runner.test_model(model, dataset, 'second_R2')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/ithor/val_small/")
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--dataset', choices=['voroni', 'ithor'], default='ithor')
    parser.add_argument('--biscuit_checkpoint', type=str,
                        default="../data/ithor/models/BISCUITNF_40l_64hid.ckpt")
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--causal_encoder_output', type=str,
                        default='output_causal_encoder/')
    args = parser.parse_args()

    args.causal_encoder_output = os.path.join(args.causal_encoder_output,
                                              f'{args.dataset}_{args.split}_{args.train_prop}_{args.max_epochs}/')
    os.makedirs(os.path.dirname(args.causal_encoder_output), exist_ok=True)

    main(args)
