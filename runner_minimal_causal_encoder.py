from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from experiments.datasets import iTHORDataset
from models.shared.causal_encoder import CausalEncoder
import torch.utils.data as data
import sys
from models.biscuit_nf import BISCUITNF

class RunnerMinimalCausalEncoder():
    def __init__(self, num_train_epochs=100, train_prop=0.5):
        super().__init__()
        self.num_train_epochs = num_train_epochs
        self.train_prop = train_prop
        self.cluster = False
        self.log_postfix = ''

    def train_network(self, pl_module, train_dataset, target_assignment):
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
        return encoder

    def test_model(self, pl_module, dataset):
        # Encode whole dataset with pl_module
        is_training = pl_module.training
        pl_module = pl_module.eval()
        loader = data.DataLoader(dataset, batch_size=256, drop_last=False, shuffle=False)
        all_encs, all_latents = [], []
        for batch in loader:
            inps, *_, latents = batch
            ae_output = pl_module.autoencoder.encoder(inps)
            encs = pl_module.encode(ae_output.to(pl_module.device)).cpu()
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
        encoder = self.train_network(pl_module, train_dataset, target_assignment)
        encoder.eval()
        # Record predictions of model on test and calculate distances
        test_inps, test_labels = all_encs[test_dataset.indices], all_latents[test_dataset.indices]
        test_exp_inps, test_exp_labels = self._prepare_input(test_inps, target_assignment.cpu(), test_labels, flatten_inp=False)
        pred_dict = encoder.forward(test_exp_inps.to(pl_module.device))
        for key in pred_dict:
            pred_dict[key] = pred_dict[key].cpu()
        _, dists, norm_dists = encoder.calculate_loss_distance(pred_dict, test_exp_labels)
        # Calculate statistics (R^2, pearson, etc.)
        avg_norm_dists, r2_matrix = self.log_R2_statistic(encoder, test_labels, norm_dists, )
        # self.log_Spearman_statistics(trainer, encoder, pred_dict, test_labels, pl_module=pl_module)
        if is_training:
            pl_module = pl_module.train()
        return r2_matrix

    def log_R2_statistic(self, encoder, test_labels, norm_dists):
        avg_pred_dict = OrderedDict()
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            gt_vals = test_labels[...,i]
            if var_info.startswith('continuous'):
                avg_pred_dict[var_key] = gt_vals.mean(dim=0, keepdim=True).expand(gt_vals.shape[0],)
            elif var_info.startswith('angle'):
                avg_angle = torch.atan2(torch.sin(gt_vals).mean(dim=0, keepdim=True),
                                        torch.cos(gt_vals).mean(dim=0, keepdim=True)).expand(gt_vals.shape[0],)
                avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2*np.pi, avg_angle)
                avg_pred_dict[var_key] = torch.stack([torch.sin(avg_angle), torch.cos(avg_angle)], dim=-1)
            elif var_info.startswith('categ'):
                gt_vals = gt_vals.long()
                mode = torch.mode(gt_vals, dim=0, keepdim=True).values
                avg_pred_dict[var_key] = F.one_hot(mode, int(var_info.split('_')[-1])).float().expand(gt_vals.shape[0], -1)
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
        #log_matrix(r2_matrix, trainer, 'r2_matrix' + self.log_postfix)
        #self._log_heatmap(trainer=trainer,
        #                  values=r2_matrix,
        #                  tag='r2_matrix',
        #                  title='R^2 Matrix',
        #                  xticks=[key for key in encoder.hparams.causal_var_info],
        #                  pl_module=pl_module)
        return avg_norm_dists, r2_matrix

    def _prepare_input(self, inps, target_assignment, latents, flatten_inp=True):
        ta = target_assignment.detach()[None,:,:].expand(inps.shape[0], -1, -1)
        inps = torch.cat([inps[:,:,None] * ta, ta], dim=-2).permute(0, 2, 1)
        latents = latents[:,None].expand(-1, inps.shape[1], -1)
        if flatten_inp:
            inps = inps.flatten(0, 1)
            latents = latents.flatten(0, 1)
        return inps, latents


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = iTHORDataset("../data/ithor/val_small/", split='val', single_image=True, return_targets=True, return_latents=True)#

    model = BISCUITNF.load_from_checkpoint('../data/ithor/models/BISCUITNF_40l_64hid.ckpt')
    model.to(device)
    model.freeze()
    _ = model.eval()

    causal_encode_runner = RunnerMinimalCausalEncoder(num_train_epochs=5)
    r2 = causal_encode_runner.test_model(model, dataset)
    np.set_printoptions(precision=6, suppress=True)
    print(r2)

if __name__ == '__main__':
    main()
