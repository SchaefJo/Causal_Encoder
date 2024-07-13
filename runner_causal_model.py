import argparse
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
from models.shared.causal_mlp import CausalMLP

seed = 42
torch.manual_seed(seed)


class RunnerCausalModel:
    def __init__(self, causal_var_info, model, num_train_epochs=100, train_prop=0.5, log_interval=10, dataset='ithor',
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
        self.result_path = os.path.join(checkpoint_path, 'results.json')

        # TODO do this?
        self.cluster = False
        self.log_postfix = ''

        self.model_type = model
        if model == 'gp':
            kernel = 1.0 * RBF(length_scale=1.0)
            # kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
            self.model = CausalGPSklearn(kernel=kernel, causal_var_info=causal_var_info, result_path=self.result_path)
        elif model == 'mlp':
            self.model = CausalMLP(hidden_dim=128,
                            lr=4e-3,
                            causal_var_info=causal_var_info,
                            results_path=self.result_path)

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

        for _ in range(iterations):
            if dataset_test is None:
                print('No test dataset given, validation is split in half.')
                train_dataset, test_dataset = data.random_split(full_dataset,
                                                            lengths=[train_size, test_size])

                test_inps, test_labels = all_encs[test_dataset.indices], all_latents[test_dataset.indices]
            else:
                print('The given test dataset is used.')
                samples_train = RandomSampler(full_dataset, replacement=True, num_samples=train_size)
                indices_train = list(samples_train)
                train_dataset = data.TensorDataset(all_encs[indices_train], all_latents[indices_train])

                test_inps, test_labels = self.preprocess_data(dataset_test, pl_module)
                test_dataset = data.TensorDataset(test_inps, test_labels)

            if self.model_type == 'mlp':
                self.train_mlp(pl_module, train_dataset, causal_var_info)
                self.model.eval()
            elif self.model_type == 'gp':
                train_inps, train_labels = all_encs[train_dataset.indices], all_latents[train_dataset.indices]
                print("Start training")
                self.model.train(train_inps, train_labels)

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
    parser.set_defaults(disentangled=True)
    args = parser.parse_args()

    args.causal_encoder_output = os.path.join(args.causal_encoder_output, f'output_causal_{args.model}',
                                              f'{args.dataset}_{args.split}_{args.train_prop}_{args.max_epochs}_DISENTANGLED{args.disentangled}/')
    os.makedirs(os.path.dirname(args.causal_encoder_output), exist_ok=True)

    main(args)
