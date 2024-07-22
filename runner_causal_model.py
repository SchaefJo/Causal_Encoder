import argparse
import random

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process.kernels import RBF
from torch.utils.data import RandomSampler
from tqdm import tqdm
import torch.nn.functional as F
from experiments.datasets import iTHORDataset, VoronoiDataset
from models.biscuit_vae import BISCUITVAE
import torch.utils.data as data
from datetime import datetime
import os
from models.biscuit_nf import BISCUITNF
from models.shared.causal_gp_sklearn import CausalGPSklearn
from models.shared.causal_model_sklearn import CausalUncertaintySklearn
from models.shared.causal_mlp import CausalMLP
from sklearn.ensemble import RandomForestClassifier


seed = 42
torch.manual_seed(seed)


class RunnerCausalModel:
    def __init__(self, causal_var_info, model, num_train_epochs=100, train_prop=0.5, log_interval=10, dataset='ithor',
                 checkpoint_path='pretrained_models/causal_encoder/', disentangled=True):
        super().__init__()
        self.active_learning_pool = None
        self.dataset = dataset
        self.num_train_epochs = num_train_epochs
        self.train_prop = train_prop
        self.log_interval = log_interval
        self.disentangled = disentangled
        current_time = datetime.now()
        self.date_time_str = current_time.strftime('%d.%m._%H:%M:%S')

        self.checkpoint_path = checkpoint_path
        self.result_path = os.path.join(checkpoint_path, 'results.json')
        self.causal_var_info = causal_var_info

        # TODO do this?
        self.cluster = False
        self.log_postfix = ''

        self.model_type = model
        if model == 'gp':
            kernel = 1.0 * RBF(length_scale=1.0)
            # kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
            #self.model = CausalGPSklearn(kernel=kernel, causal_var_info=causal_var_info, result_path=self.result_path)
            self.model = CausalUncertaintySklearn(model_name=model, causal_var_info=causal_var_info,
                                                  result_path=self.result_path, kernel=kernel)
        elif model == 'mlp':
            self.model = CausalMLP(hidden_dim=128,
                            lr=4e-3,
                            causal_var_info=causal_var_info,
                            results_path=self.result_path)
        elif model == 'rf':
            self.model = CausalUncertaintySklearn(model_name=model, causal_var_info=causal_var_info, result_path=self.result_path)

    def train_mlp(self, pl_module, train_dataset, causal_var_info):
        device = pl_module.device

        # We use one, sufficiently large network that predicts for any input all causal variables
        # To iterate over the different sets, we use a mask which is an extra input to the model
        # This is more efficient than using N networks and showed same results with large hidden size

        optimizer = self.model.configure_optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]

        train_loader = data.DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=512)
        self.model.to(device)
        self.model.train()
        range_iter = range(self.num_train_epochs)
        if not self.cluster:
            range_iter = tqdm(range_iter, leave=False, desc=f'Training correlation encoder {self.log_postfix}')
        for epoch_idx in range_iter:
            avg_loss = 0.0
            for inps, latents in train_loader:
                inps = inps.to(device)
                latents = latents.to(device)
                #inps, latents = self._prepare_input(inps, target_assignment, latents)
                loss = self.model._get_loss(inps, latents)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            if epoch_idx % self.log_interval == 0:
                avg_loss /= len(train_loader)
                print(f'Epoch [{epoch_idx}/{self.num_train_epochs}], Loss: {avg_loss:.4f}')
        self.model.compute_individual_losses(train_loader, causal_var_info, "train")
        model_path = os.path.join(self.checkpoint_path, f'model_{self.date_time_str}.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f'Model checkpoint saved at {self.checkpoint_path}')
        return self.model

    def preprocess_data(self, dataset, pl_module):
        loader = data.DataLoader(dataset, batch_size=256, drop_last=False, shuffle=False)

        all_encs, all_latents = [], []

        if self.disentangled:
            print('Latent space disentangled')
        else:
            print('Latent space entangled')

        for batch in loader:
            inps, *_, latents = batch

            if self.dataset == 'ithor':
                inps = pl_module.autoencoder.encoder(inps)
            if self.disentangled:
                encs = pl_module.encode(inps.to(pl_module.device)).cpu()
            else:
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
        return all_encs, all_latents

    def test_model(self, pl_module, dataset, dataset_test, iterations):
        pl_module = pl_module.eval()
        causal_var_info = dataset.get_causal_var_info()

        all_encs, all_latents = self.preprocess_data(dataset, pl_module)
        full_dataset = data.TensorDataset(all_encs, all_latents)

        train_size = int(self.train_prop * all_encs.shape[0])
        print(f'Train Size {train_size}')

        test_size = all_encs.shape[0] - train_size

        for i in range(iterations):
            print(f"Iteration {i}")
            if dataset_test is None:
                print('No test dataset given, validation is split in half.')
                train_dataset, test_dataset = data.random_split(full_dataset,
                                                            lengths=[train_size, test_size])

                test_inps, test_labels = all_encs[test_dataset.indices], all_latents[test_dataset.indices]

                self.active_learning_pool = None

                train_inps, train_labels = all_encs[train_dataset.indices], all_latents[train_dataset.indices]
            else:
                print('The given test dataset is used.')
                samples_train = RandomSampler(full_dataset, replacement=True, num_samples=train_size)
                indices_train = list(samples_train)
                self.train_index = indices_train
                train_dataset = data.TensorDataset(all_encs[indices_train], all_latents[indices_train])
                self.train_dataset = train_dataset
                train_inps, train_labels = all_encs[indices_train], all_latents[indices_train]

                #get non training examples from the dataset
                all_indices = set(range(len(all_encs)))
                train_indices_set = set(indices_train)
                remaining_indices = list(all_indices - train_indices_set)
                self.active_learning_pool = data.TensorDataset(all_encs[remaining_indices], all_latents[remaining_indices])

                test_inps, test_labels = self.preprocess_data(dataset_test, pl_module)
                test_dataset = data.TensorDataset(test_inps, test_labels)
                self.test_dataset = test_dataset

            if self.model_type == 'mlp':
                self.train_mlp(pl_module, train_dataset, causal_var_info)
                self.model.eval()
            elif self.model_type == 'gp':
                print("Start training")
                self.model.train(train_inps, train_labels, verbose=False)

            # Record predictions of model on test and calculate distances
            test_inps = test_inps.to(pl_module.device)
            test_labels = test_labels.to(pl_module.device)

            with torch.no_grad():
                comb_loss = self.model._get_loss(test_inps, test_labels)
            print(f'Comb loss: {comb_loss.item()}')

            # TODO change loss to make _get_loss equal to _individual_loss
            if self.model_type == 'mlp':
                test_loader = data.DataLoader(test_dataset, shuffle=False, drop_last=False, batch_size=512)
                self.model.compute_individual_losses(test_loader, causal_var_info, "test")


    def save_data(self, new_data, new_labels, path, max_uncertainty_idx=None):
        new_data_array = new_data.numpy() if hasattr(new_data, 'numpy') else new_data
        new_labels_array = new_labels.numpy() if hasattr(new_labels, 'numpy') else new_labels

        data_df = pd.DataFrame(new_data_array)
        labels_df = pd.DataFrame(new_labels_array)
        merged_df = pd.concat([data_df, labels_df], axis=1)
        if max_uncertainty_idx is not None:
            if isinstance(max_uncertainty_idx, (int, list, np.ndarray, pd.Index)):
                merged_df.index = pd.Index(max_uncertainty_idx)
            else:
                merged_df.index = pd.Index([max_uncertainty_idx])
        merged_df.to_csv(path)


    def active_learning(self, al_iterations, al_strategy, pl_module):
        # TODO compare MLP with gp
        print("Active Learning")
        al_path = os.path.join(self.checkpoint_path, f'active_learning_train_data.csv')
        train_data, train_labels = self.train_dataset.tensors
        self.save_data(self, train_data, train_labels, al_path, self.train_index)
        if self.model_type == 'gp':
            for i in range(al_iterations):
                print(f'Active Learning Iteration: {i}')
                _, uncertainty = self.model.forward(self.active_learning_pool.tensors[0])
                if al_strategy == 'most_uncertain':
                    #TODO does not work
                    max_uncertainty = -1
                    max_uncertainty_idx = -1
                    max_causal = ''
                    for causal, uncertainties in uncertainty.items():
                        cur_idx = np.argmax(uncertainties)
                        cur_val = np.max(uncertainties)

                        if cur_val > max_uncertainty:
                            max_uncertainty_idx = cur_idx
                            max_uncertainty = cur_val
                            max_causal = causal
                    print(f'Most uncertain Causal: {max_causal} with uncertainty: {max_uncertainty}')
                elif al_strategy == 'one_most_uncertain':
                    max_uncertainty_idx = []
                    max_uncertainty_vals = []
                    for causal, uncertainties in uncertainty.items():
                        cur_idx = np.argmax(uncertainties)
                        max_uncertainty_vals.append(np.max(uncertainties))
                        max_uncertainty_idx.append(cur_idx)
                    max_uncertainty_idx = random.choice(max_uncertainty_idx)
                elif al_strategy == 'uncertain_per_causal':
                    max_uncertainty_idx = []
                    max_uncertainty_vals = []
                    for causal, uncertainties in uncertainty.items():
                        cur_idx = np.argmax(uncertainties)
                        max_uncertainty_vals.append(np.max(uncertainties))
                        max_uncertainty_idx.append(cur_idx)
                    max_uncertainty_idx = np.unique(np.array(max_uncertainty_idx, dtype=int))

                elif al_strategy == 'random_per_causal':
                    #max_uncertainty_idx = random.randint(0, len(uncertainty.items()[0]) - 1)

                    num_picks = 17
                    #max_index = len(uncertainty.items()[0])
                    #max_uncertainty_idx = random.sample(range(max_index), num_picks)
                    max_uncertainty_vals = None
                    uncertainty_values = list(uncertainty.items())[0][1]
                    max_index = len(uncertainty_values)

                    max_uncertainty_idx = random.sample(range(max_index), num_picks)
                elif al_strategy == 'random':
                    uncertainty_values = list(uncertainty.items())[0][1]
                    max_index = len(uncertainty_values)
                    max_uncertainty_idx = random.randint(0, max_index - 1)
                elif al_strategy == 'average_uncertainty':
                    raise NotImplementedError('not yet implemented')
                    #TODO here look at which data point would help the most classifiers at once

                else:
                    raise ValueError('This active learning strategy does not exist!')

                # TODO maybe not only picking one but picking the top 3 is better

                new_data, new_labels = self.active_learning_pool.tensors[0][max_uncertainty_idx], \
                    self.active_learning_pool.tensors[1][max_uncertainty_idx]

                al_path = os.path.join(self.checkpoint_path, f'active_learning_data_iter_{i}.csv')
                self.save_data(self, new_data, new_labels, al_path, max_uncertainty_idx)

                train_data, train_labels = self.train_dataset.tensors

                if len(new_data.shape) == 1:
                    new_data = new_data.unsqueeze(0)
                if len(new_labels.shape) == 1:
                    new_labels = new_labels.unsqueeze(0)

                new_train_data = torch.cat((train_data, new_data), dim=0)
                new_train_labels = torch.cat((train_labels, new_labels), dim=0)

                print(f'New dataset shape: {new_train_data.shape}')

                self.train_dataset = data.TensorDataset(new_train_data, new_train_labels)

                # TODO this is for MLP
                # new_train_dataset = data.TensorDataset(new_train_data, new_train_labels)

                self.model.train(new_train_data, new_train_labels, verbose=False, save=False)

                test_inps, test_labels = self.test_dataset.tensors
                # TODO save this per al iteration, look at performance of uncertain examples, still uncertain?
                comb_loss = self.model._get_loss(test_inps, test_labels, save=True, al_iter=i)
                print(f'Comb loss testing: {comb_loss}')

                #print(f'Old uncertainty: {max_uncertainty_vals}')
                #_, uncertainty = self.model.forward(new_data)
                #print(f'New uncertainty: {uncertainty}')

                # TODO remove new trained data from uncertainty calc data

        else:
            raise NotImplementedError('Not implemented uncertainty for mlp')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'ithor':
        dataset = iTHORDataset(args.data_dir, split=args.split, single_image=True, return_targets=False,
                               return_latents=True)
        if args.test_data_split is not None:
            test_dataset = iTHORDataset(args.data_dir, split=args.test_data_split, single_image=True, return_targets=False,
                               return_latents=True)
        else:
            test_dataset = None
        model_biscuit = BISCUITNF.load_from_checkpoint(args.biscuit_checkpoint)
    else:
        dataset = VoronoiDataset(args.data_dir, split=args.split, single_image=True, return_targets=False,
                                 return_latents=True)
        if args.test_data_split is not None:
            test_dataset = VoronoiDataset(args.data_dir, split=args.test_data_split, single_image=True, return_targets=False,
                                 return_latents=True)
        else:
            test_dataset = None
        model_biscuit = BISCUITVAE.load_from_checkpoint(args.biscuit_checkpoint)
    model_biscuit.to(device)
    model_biscuit.freeze()
    _ = model_biscuit.eval()

    causal_encode_runner = RunnerCausalModel(num_train_epochs=args.max_epochs, log_interval=args.log_interval,
                                             checkpoint_path=args.causal_encoder_output, causal_var_info=dataset.get_causal_var_info(),
                                             train_prop=args.train_prop, dataset=args.dataset,
                                             disentangled=args.disentangled, model=args.model)

    causal_encode_runner.test_model(model_biscuit, dataset, test_dataset, args.iterations)

    if args.active_learning:
        causal_encode_runner.active_learning(args.active_learning_iterations, args.active_learning_strategy, model_biscuit)


def create_versioned_subdir(base_dir):
    os.makedirs(base_dir, exist_ok=True)

    subdirs = [d for d in os.listdir(base_dir) if
               os.path.isdir(os.path.join(base_dir, d)) and d.startswith("version_")]

    version_numbers = [int(d.split("_")[1]) for d in subdirs if d.split("_")[1].isdigit()]
    next_version = max(version_numbers, default=-1) + 1

    new_version_dir = os.path.join(base_dir, f"version_{next_version}")
    os.makedirs(new_version_dir)

    return new_version_dir




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/ithor/val_small/")
    parser.add_argument('--test_data_split', type=str)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--dataset', choices=['voronoi', 'ithor'], default='ithor')
    parser.add_argument('--biscuit_checkpoint', type=str,
                        default="../data/ithor/models/BISCUITNF_40l_64hid.ckpt")
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--causal_encoder_output', type=str,
                        default='.')
    parser.add_argument('--entangled', dest='disentangled', action='store_false',
                        help='If set, disables disentanglement. Disentanglement is normally on.')
    parser.add_argument('--model', choices=['mlp', 'encoder', 'gp'], default='gp')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for model fitting')
    parser.add_argument('--active_learning', action='store_true', default=False)
    parser.add_argument('--active_learning_iterations', type=int, default=10)
    parser.add_argument('--active_learning_strategy', default='most_uncertain', choices=['most_uncertain',
                        'uncertain_per_causal', 'average_uncertainty', 'random', 'one_most_uncertain', 'random_per_causal'])
    parser.set_defaults(disentangled=True)
    args = parser.parse_args()

    args.causal_encoder_output = os.path.join(args.causal_encoder_output, f'output_causal_{args.model}',
                                              f'{args.dataset}_{args.split}_{args.train_prop}_{args.max_epochs}_DISENTANGLED{args.disentangled}/')
    args.causal_encoder_output = create_versioned_subdir(args.causal_encoder_output)
    print('args:')
    print(args)

    main(args)
