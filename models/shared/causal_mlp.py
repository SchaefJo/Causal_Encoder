import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score


class CausalMLP(nn.Module):
    def __init__(self, causal_var_info, input_dim=40, hidden_dim=128, lr=4e-3, weight_decay=0.0, results_path='results.json'):
        super(CausalMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.causal_var_info = causal_var_info
        categorical_dim, continuous_dim = self.process_output(causal_var_info)

        self.continuous_layer = nn.Linear(hidden_dim, continuous_dim)
        self.categorical_layer = nn.Linear(hidden_dim, 2 * categorical_dim)

        self.lr = lr
        self.weight_decay = weight_decay

        self.continuous_dim = continuous_dim
        self.categorical_dim = categorical_dim

        self.results_path = results_path

    def process_output(self, output):
        continuous_dim = 0
        categorical_dim = 0
        for idx, val in enumerate(output.values()):
            if val.startswith('categ_'):
                categorical_dim += 1
            elif val.startswith('continuous_'):
                continuous_dim += 1
        return categorical_dim, continuous_dim

    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))

        continuous_output = self.continuous_layer(x)
        categorical_output = self.categorical_layer(x)

        categorical_output = categorical_output.view(-1, self.categorical_dim, 2)

        return continuous_output, categorical_output

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def separate_cat_contin(self, target, var_info):
        # TODO: for now ignore what comes after the _ in values: categ_2 or continous_2.8
        categ_indices = []
        contin_indices = []
        categ_causal = []
        contin_causal = []

        for idx, (causal, val) in enumerate(var_info.items()):
            if val.startswith('categ_'):
                categ_indices.append(idx)
                categ_causal.append(causal)
            elif val.startswith('continuous_'):
                contin_indices.append(idx)
                contin_causal.append(causal)

        categorical_target = target[:, categ_indices]
        continuous_target = target[:, contin_indices]

        return continuous_target, categorical_target, contin_causal, categ_causal


    def _get_loss(self, inps, target):
        continuous_target, categorical_target, _, _ = self.separate_cat_contin(target, self.causal_var_info)
        continuous_pred, categorical_pred = self.forward(inps)

        mse_loss = F.mse_loss(continuous_pred, continuous_target)

        batch_size = categorical_target.size(0)
        categorical_pred = categorical_pred.view(batch_size * self.categorical_dim, 2)
        categorical_target = categorical_target.view(batch_size * self.categorical_dim).long()
        ce_loss = F.cross_entropy(categorical_pred, categorical_target)

        num_targets = self.continuous_dim + self.categorical_dim

        if (len(categorical_target) > 0) & (len(continuous_target) > 0):
            combined_loss = (self.continuous_dim * mse_loss) / num_targets + (self.categorical_dim * ce_loss) / num_targets
        elif len(categorical_target) > 0:
            combined_loss = ce_loss
        else:
            combined_loss = mse_loss

        return combined_loss

    def compute_individual_losses(self, data_loader, var_info, split):
        individual_losses = []

        for batch_idx, (inps, target) in enumerate(data_loader):
            continuous_target, categorical_target, continuous_causal, categorical_causal = self.separate_cat_contin(
                target, var_info)
            continuous_pred, categorical_pred = self.forward(inps)

            for i, causal in enumerate(continuous_causal):
                mse = F.mse_loss(continuous_pred[:, i], continuous_target[:, i])
                individual_losses.append({'causal': causal, 'loss': mse.item(), 'split': split})

            for i, causal in enumerate(categorical_causal):
                categorical_pred_i = categorical_pred[:, i, :]
                categorical_target_i = categorical_target[:, i].long()

                ce = F.cross_entropy(categorical_pred_i, categorical_target_i)

                categorical_target_i_np = categorical_target_i.detach().numpy()
                categorical_pred_i_np = torch.argmax(categorical_pred_i, dim=1).detach().numpy()
                acc = accuracy_score(categorical_target_i_np, categorical_pred_i_np)
                individual_losses.append({'causal': causal, 'split': split, 'loss': ce.item(), 'accuracy': acc})

        self._save_metrics_to_file(individual_losses, self.results_path)

    def _save_metrics_to_file(self, metrics, file_name):
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                existing_metrics = json.load(f)
            existing_metrics.extend(metrics)
            with open(file_name, 'w') as f:
                json.dump(existing_metrics, f, indent=4)
        else:
            with open(file_name, 'w') as f:
                json.dump(metrics, f, indent=4)
