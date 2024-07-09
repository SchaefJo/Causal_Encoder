import argparse
import torch
from sklearn.gaussian_process.kernels import RBF
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
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor



class RunnerCausalGP:
    def __init__(self, causal_var_info, train_prop=0.5, dataset='ithor',
                 checkpoint_path='pretrained_models/causal_encoder/', disentangled=True):
        super().__init__()
        self.dataset = dataset
        self.train_prop = train_prop
        self.disentangled = disentangled
        current_time = datetime.now()
        self.date_time_str = current_time.strftime('%d.%m._%H:%M:%S')

        self.checkpoint_path = checkpoint_path

        self.cluster = False
        self.log_postfix = ''

        kernel = 1.0 * RBF(length_scale=1.0)
        # kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        self.CGP = CausalGPSklearn(kernel=kernel, causal_var_info=causal_var_info)


    def test_model(self, pl_module, dataset, name):
        pl_module = pl_module.eval()
        loader = data.DataLoader(dataset, batch_size=256, drop_last=False, shuffle=False)
        causal_var_info = dataset.get_causal_var_info()
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
        train_inps, train_labels = all_encs[train_dataset.indices], all_latents[train_dataset.indices]
        print("Start training")
        self.CGP.train(train_inps, train_labels)

        test_inps, test_labels = all_encs[test_dataset.indices], all_latents[test_dataset.indices]

        comb_loss = self.CGP._get_loss(test_inps, test_labels)
        print(f'Comb loss: {comb_loss}')

        return comb_loss


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

    causal_encode_runner = RunnerCausalGP(causal_var_info=dataset.get_causal_var_info(),
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
    parser.add_argument('--entangled', dest='disentangled', action='store_false',
                        help='If set, disables disentanglement. Disentanglement is normally on.')
    parser.set_defaults(disentangled=True)
    args = parser.parse_args()

    args.causal_encoder_output = os.path.join(args.causal_encoder_output,
                                              f'{args.dataset}_{args.split}_{args.train_prop}_{args.max_epochs}_DISENTANGLED{args.disentangled}/')
    os.makedirs(os.path.dirname(args.causal_encoder_output), exist_ok=True)

    main(args)
