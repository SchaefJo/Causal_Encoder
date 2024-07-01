import argparse
from typing import List, Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Patch
from tqdm.auto import tqdm
from glob import glob
import torch
import pytorch_lightning as pl
import torch.utils.data as data
import sys
import os
from collections import OrderedDict
import torch.nn.functional as F
from scipy.stats import spearmanr
# from scipy.optimize import linear_sum_assignment
import seaborn as sns

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
print(parent_dir)

from models.biscuit_nf import BISCUITNF
from models.biscuit_vae import BISCUITVAE
from experiments.datasets import VoronoiDataset, CausalWorldDataset, iTHORDataset
from models.shared.causal_encoder import CausalEncoder


# from experiments.utils import encode_dataset


class LabelEvaluation():
    """ This is used for Causal Relationship and evaluation. """

    def __init__(self, dataset, num_train_epochs=100):
        super().__init__()
        assert dataset is not None, "Dataset for correlation metrics cannot be None."
        self.dataset = dataset
        self.val_dataset = dataset
        # self.test_dataset = test_dataset
        # self.every_n_epochs = every_n_epochs
        self.num_train_epochs = num_train_epochs
        # self.cluster = cluster
        self.log_postfix = ''
        self.extra_postfix = ''

    @torch.enable_grad()
    @torch.inference_mode(False)
    def train_network(self, pl_module, train_dataset, target_assignment):
        device = pl_module.device
        if hasattr(pl_module, 'causal_encoder') and pl_module.causal_encoder is not None:
            causal_var_info = pl_module.causal_encoder.hparams.causal_var_info
        else:
            causal_var_info = pl_module.hparams.causal_var_info
        # We use one, sufficiently large network that predicts for any input all causal variables
        # To iterate over the different sets, we use a mask which is an extra input to the model
        # This is more efficient than using N networks and showed same results with large hidden size
        # Use CausalEncoder instead of MLP cause the input could ebe continuous, categorical or angle.
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
        range_iter = tqdm(range_iter, leave=False, desc=f'Training correlation encoder {self.log_postfix}')
        for epoch_idx in range_iter:
            avg_loss = 0.0
            for inps, latents in train_loader:
                inps = inps.to(device)
                latents = latents.to(device)
                # print(f"inps, latents in train_network: {inps.shape}, {latents.shape}")
                # Add mask to the input
                inps, latents = self._prepare_input(inps, target_assignment, latents)
                loss = encoder._get_loss([inps, latents], mode=None)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
        return encoder

    # P: what latents for training step 2
    # Mask input to use the same network
    def _prepare_input(self, inps, target_assignment, latents, flatten_inp=True):
        ta = target_assignment.detach()[None, :, :].expand(inps.shape[0], -1, -1)
        # concatenate the masked input with the mask.
        inps = torch.cat([inps[:, :, None] * ta, ta], dim=-2).permute(0, 2, 1)
        latents = latents[:, None].expand(-1, inps.shape[1], -1)
        if flatten_inp:
            inps = inps.flatten(0, 1)
            latents = latents.flatten(0, 1)
        return inps, latents

    def log_R2_statistic(self, encoder, pred_dict, test_labels, norm_dists, point, pl_module=None, prop=0.5):
        avg_pred_dict = OrderedDict()
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            # Why calculate the distance between ground truth and the normalized ground truth?
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
            ss_tot = np.where(ss_tot == 0.0, 1.0, ss_tot)
            r2 = 1 - ss_res / ss_tot
            r2_matrix.append(r2)
        r2_matrix = torch.stack(r2_matrix, dim=-1).cpu().numpy()
        # log_matrix(r2_matrix, trainer, 'r2_matrix' + self.log_postfix)
        print(f"R2 matrix shape: {r2_matrix.shape}")
        self._log_heatmap(values=r2_matrix,
                          tag='r2_matrix',
                          title='R^2 Matrix',
                          xticks=[key for key in encoder.hparams.causal_var_info],
                          pl_module=pl_module,
                          point=point,
                          prop=prop)
        return avg_norm_dists, r2_matrix

    def log_Spearman_statistics(self, trainer, encoder, pred_dict, test_labels, pl_module=None):
        spearman_matrix = []
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            gt_vals = test_labels[..., i]
            pred_val = pred_dict[var_key]
            if var_info.startswith('continuous'):
                spearman_preds = pred_val.squeeze(dim=-1)  # Nothing needs to be adjusted
            elif var_info.startswith('angle'):
                spearman_preds = F.normalize(pred_val, p=2.0, dim=-1)
                gt_vals = torch.stack([torch.sin(gt_vals), torch.cos(gt_vals)], dim=-1)
                # angles = torch.atan(pred_val[...,0] / pred_val[...,1])
                # angles = torch.where(angles < 0.0, angles + 2*np.pi, angles)
                # spearman_preds = angles
            elif var_info.startswith('categ'):
                spearman_preds = pred_val.argmax(dim=-1).float()
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in Spearman statistics.'

            gt_vals = gt_vals.cpu().numpy()
            spearman_preds = spearman_preds.cpu().numpy()
            results = torch.zeros(spearman_preds.shape[1], )
            for j in range(spearman_preds.shape[1]):
                if len(spearman_preds.shape) == 2:
                    if np.unique(spearman_preds[:, j]).shape[0] == 1:
                        results[j] = 0.0
                    else:
                        results[j] = spearmanr(spearman_preds[:, j], gt_vals).correlation
                elif len(spearman_preds.shape) == 3:
                    num_dims = spearman_preds.shape[-1]
                    for k in range(num_dims):
                        if np.unique(spearman_preds[:, j, k]).shape[0] == 1:
                            results[j] = 0.0
                        else:
                            results[j] += spearmanr(spearman_preds[:, j, k], gt_vals[..., k]).correlation
                    results[j] /= num_dims

            spearman_matrix.append(results)

        spearman_matrix = torch.stack(spearman_matrix, dim=-1).cpu().numpy()
        #log_matrix(spearman_matrix, trainer, 'spearman_matrix' + self.log_postfix)
        self._log_heatmap(trainer=trainer,
                          values=spearman_matrix,
                          tag='spearman_matrix',
                          title='Spearman\'s Rank Correlation Matrix',
                          xticks=[key for key in encoder.hparams.causal_var_info],
                          pl_module=pl_module)

    @torch.no_grad()
    def test_model(self, pl_module, point, prop=0.5):
        # Encode whole dataset with pl_module
        is_training = pl_module.training
        pl_module = pl_module.eval()
        loader = data.DataLoader(self.dataset, batch_size=256, drop_last=False, shuffle=False)
        all_encs, all_latents = [], []
        for batch in loader:
            # What are the entries in the batch?
            # inps are the images.
            inps, *_, latents = batch
            # There is no encode function in AE?
            # The output of NF encode is the causals of the BISCUIT
            if isinstance(self.dataset, iTHORDataset):
                encs = pl_module.autoencoder.encoder(inps.to(pl_module.device))
            else:
                encs = inps.to(pl_module.device)
            # TODO test entangled latents
            # encs = pl_module.encode(encs).cpu()
            encs = encs.cpu()
            all_encs.append(encs)
            all_latents.append(latents)
        all_encs = torch.cat(all_encs, dim=0)
        all_latents = torch.cat(all_latents, dim=0)
        # shape of all_encs and all latents in test_model: torch.Size([400, 40]), torch.Size([400, 0, 2])
        # print(f"shape of all_encs and all latents in test_model: {all_encs.shape}, {all_latents.shape}")
        # Normalize latents for stable gradient signals
        all_encs = (all_encs - all_encs.mean(dim=0, keepdim=True)) / all_encs.std(dim=0, keepdim=True).clamp(min=1e-2)
        # Create new tensor dataset for training (50%) and testing (50%)
        full_dataset = data.TensorDataset(all_encs, all_latents)
        # TODO P: Half for training first step
        print(f"Dataset size: {all_encs.shape[0]}")
        train_size = int(prop * all_encs.shape[0])
        test_size = all_encs.shape[0] - train_size
        train_dataset, test_dataset = data.random_split(full_dataset,
                                                        lengths=[train_size, test_size],
                                                        generator=torch.Generator().manual_seed(42))
        # Train network to predict causal factors from latent variables
        if hasattr(pl_module, 'target_assignment') and pl_module.target_assignment is not None:
            target_assignment = pl_module.target_assignment.clone()
            print(f"Cloned target assignment in {point}: {target_assignment.shape}")
        else:
            # For the step 1, the ta is the unit matrix so the masked input stays unchanged.
            target_assignment = torch.eye(all_encs.shape[-1])
            print(f"Target assignment in {point}: {target_assignment.shape}")
        encoder = self.train_network(pl_module, train_dataset, target_assignment)
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
        # Why use test_label instead of test_exp_labels?
        avg_norm_dists, r2_matrix = self.log_R2_statistic(encoder, pred_dict, test_labels, norm_dists, point,
                                                          pl_module=pl_module, prop=prop)
        # self.log_Spearman_statistics(trainer, encoder, pred_dict, test_labels, pl_module=pl_module)
        if is_training:
            pl_module = pl_module.train()
        return r2_matrix

    def _log_heatmap(self, values, tag, title=None, xticks=None, yticks=None, xlabel=None, ylabel=None, pl_module=None,
                     point=None, prop=0.5):
        print(f"value shape in log heatmap: {values.shape}")
        if ylabel is None:
            ylabel = 'Target dimension'
        if xlabel is None:
            xlabel = 'True causal variable'
        if yticks is None:
            yticks = [f'Dim {i + 1}' for i in range(values.shape[0])]
        if xticks is None:
            xticks = self.dataset.target_names()
        fig = plt.figure(figsize=(min(6, max(4, values.shape[1] / 1.25)),
                                  min(6, max(4, values.shape[0] / 1.25))),
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

        # pl_module.logger.experiment.add_figure(tag + pl_module.log_postfix, fig, global_step=global_step)
        save_directory = "images"
        filename = f"{point}_{prop}.png"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        file_path = os.path.join(save_directory, filename)
        fig.savefig(file_path)
        plt.close(fig)

        if values.shape[0] == values.shape[1] + 1:
            values = values[:-1]

        if values.shape[0] == values.shape[1]:
            avg_diag = np.diag(values).mean()
            print(f"avg_diag, max_off_diag: {avg_diag}")
            max_off_diag = (values - np.eye(values.shape[0]) * 10).max(axis=-1).mean()
            # pl_module.log(f'corr_callback_{tag}_diag{pl_module.log_postfix}', avg_diag, global_step=global_step)
            # pl_module.log(f'corr_callback_{tag}_max_off_diag{pl_module.log_postfix}', max_off_diag, global_step=global_step)
            print(f"avg_diag, max_off_diag: {max_off_diag}")

            return avg_diag, max_off_diag

    # At first, map all latents to causals. Group the latents.
    @torch.no_grad()
    def on_validation_epoch_start(self, pl_module, is_test=False, prop=0.5):
        self.log_postfix = '_all_latents' + ('_test' if is_test else '')
        self.extra_postfix = '_test' if is_test else ''
        pl_module.target_assignment = None
        r2_matrix = self.test_model(pl_module, "R2_start", prop)
        r2_matrix = torch.from_numpy(r2_matrix)
        # Assign each latent to the causal variable with the highest (relative) correlation
        r2_matrix = r2_matrix / r2_matrix.abs().max(dim=0, keepdims=True).values.clamp(min=0.1)
        max_r2 = r2_matrix.argmax(dim=-1)
        ta = F.one_hot(max_r2, num_classes=r2_matrix.shape[-1]).float()
        print(f"Target assignment in on_validation_epoch_start: {ta.shape}")
        # Group multi-dimensional causal variables together
        # Why the correponding relationship is continuous?
        if isinstance(self.dataset, iTHORDataset):
            ta = torch.cat([ta[:, :1],
                            ta[:, 1:7].sum(dim=-1, keepdims=True),  # Combine the latents into the causals
                            ta[:, 7:9],
                            ta[:, 9:13].sum(dim=-1, keepdims=True),
                            ta[:, 13:]], dim=-1)
        elif isinstance(self.dataset, CausalWorldDataset):
            ta = torch.cat([ta[:, :6],
                            ta[:, 6:].sum(dim=-1, keepdims=True)], dim=-1)
        # Voronoi is fully entangled
        # Why? TODO
        # if trainer.current_epoch == 0:
        #     ta[:,0] = 1
        #     ta[:,1:] = 0
        pl_module.target_assignment = ta
        pl_module.last_target_assignment.data = ta

        return r2_matrix

    @torch.no_grad()
    # The second stage for mapping latents to causals
    #
    def on_validation_epoch_end(self, pl_module, is_test=False, prop=0.5):
        self.log_postfix = '_grouped_latents' + ('_test' if is_test else '')
        r2_matrix = self.test_model(pl_module, "R2_end", prop)

        return r2_matrix

        # if not is_test:
        #     results = trainer._results
        #     if 'validation_step.val_loss' in results:
        #         val_comb_loss = results['validation_step.val_loss'].value / results['validation_step.val_loss'].cumulated_batch_size
        #         new_val_dict = {'val_loss': val_comb_loss}
        #         for key in ['on_validation_epoch_end.corr_callback_r2_matrix_diag_grouped_latents',
        #                     'on_validation_epoch_end.corr_callback_spearman_matrix_diag_grouped_latents',
        #                     'on_validation_epoch_end.corr_callback_r2_matrix_max_off_diag_grouped_latents',
        #                     'on_validation_epoch_end.corr_callback_spearman_matrix_max_off_diag_grouped_latents']:
        #             if key in results:
        #                 val = results[key].value
        #                 new_val_dict[key.split('_',5)[-1]] = val
        #         new_val_dict = {key: (val.item() if isinstance(val, torch.Tensor) else val) for key, val in new_val_dict.items()}
        #         if self.cluster:
        #             s = f'[Epoch {trainer.current_epoch}] ' + ', '.join([f'{key}: {new_val_dict[key]:5.3f}' for key in sorted(list(new_val_dict.keys()))])
        #             print(s)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    parser.add_argument('--disentangled', type=bool, default=True)
    args = parser.parse_args()

    args.causal_encoder_output = os.path.join(args.causal_encoder_output,
                                              f'{args.dataset}_{args.split}_{args.train_prop}_{args.max_epochs}/')
    os.makedirs(os.path.dirname(args.causal_encoder_output), exist_ok=True)

    #CHECKPOINT_FILE_AE = '/home/mguo/BISCUIT/pretrained_model/pretrained_models/AE_iTHOR/AE_40l_64hid.ckpt'

    # CHECKPOINT_FILE_AE = '/home/mguo/BISCUIT/pretrained_model/pretrained_models/AE_iTHOR/AE_40l_64hid.ckpt'


    # iTHOR
    model = BISCUITNF.load_from_checkpoint(args.biscuit_checkpoint)#autoencoder_checkpoint=CHECKPOINT_FILE_AE
    model.to(device)
    model.eval()

    DataClass = iTHORDataset

    Evaluate_Label(model, DataClass, args.data_dir, args.train_prop)


def Evaluate_Label(model, DataClass, DATA_FOLDER, train_prop):
    batch_size = 64

    val_dataset = DataClass(
        data_folder=DATA_FOLDER, split='val', single_image=True, return_targets=False, return_latents=True)
    LE = LabelEvaluation(val_dataset, num_train_epochs=100)


    r2_diag = []
    r2_sep = []
    r2_matrix_start = LE.on_validation_epoch_start(model, is_test=False, prop=train_prop)
    # print(f"r2_matrix_start: {r2_matrix_start.shape}")

    r2_matrix_end = LE.on_validation_epoch_end(model, is_test=False, prop=train_prop)
    # print(f"r2_matrix_end: {r2_matrix_end.shape}, {r2_matrix_end.dtype}")
    # print(isinstance(r2_matrix_end, torch.Tensor))

    r2_matrix_end = torch.from_numpy(r2_matrix_end)
    if isinstance(val_dataset, iTHORDataset):
        r2_matrix_end = torch.cat([r2_matrix_end[:, :1],
                                   r2_matrix_end[:, 1:7].mean(dim=-1, keepdims=True),
                                   # Combine the latents into the causals
                                   r2_matrix_end[:, 7:9],
                                   r2_matrix_end[:, 9:13].mean(dim=-1, keepdims=True),
                                   r2_matrix_end[:, 13:]], dim=-1)
    elif isinstance(val_dataset, CausalWorldDataset):
        r2_matrix_end = torch.cat([r2_matrix_end[:, :6],
                                   r2_matrix_end[:, 6:].mean(dim=-1, keepdims=True)], dim=-1)
    avg_diag, max_off_diag = LE._log_heatmap(values=r2_matrix_end.numpy(),
                                             tag='r2_matrix',
                                             title='R^2 Matrix',
                                             xticks=[f'Dim {i + 1}' for i in range(r2_matrix_end.shape[0])],
                                             pl_module=model,
                                             point="R2_final")
    r2_diag.append(avg_diag)
    r2_sep.append(max_off_diag)

    # plt.figure(figsize=(10, 5))
    # plt.plot(props, r2_diag, label='Average Diagonal', marker='o')
    # plt.legend()
    # plt.title('R2 Average Diagonal Metric')
    # plt.xlabel('Prop Value')
    # plt.ylabel('Average Diagonal Metric Value')
    # plt.grid(True)
    # #plt.savefig('/home/mguo/BISCUIT/run/images/R2_diag.png')
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.plot(props, r2_sep, label='Max Off-Diagonal', marker='s')
    # plt.legend()
    # plt.title('R2 Max Off-Diagonal Metric')
    # plt.xlabel('Prop Value')
    # plt.ylabel('Max Off-Diagonal Metric Value')
    # plt.grid(True)
    # #plt.savefig('/home/mguo/BISCUIT/run/images/R2_off_diag.png')
    # plt.show()

    # model.freeze()
    # model.eval()







if __name__ == '__main__':
    main()
