"""
Run file to train BISCUIT-VAE
"""

import sys

import torch.utils.data as data

sys.path.append('../')
from models.vae import VAE
from experiments.datasets import VoronoiDataset, iTHORDataset
from experiments.utils import train_model, load_datasets, get_default_parser, print_params


if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--c_hid', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--decoder_num_blocks', type=int, default=1)
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--num_latents', type=int, default=16)
    parser.add_argument('--prior_action_add_prev_state', action='store_true')
    parser.add_argument('--logit_reg_factor', type=float, default=0.0005)

    args = parser.parse_args()
    model_args = vars(args)

    print('Loading datasets...')
    if 'voronoi' in args.data_dir:
        data_name = 'voronoi' + args.data_dir.split('voronoi')[-1].replace('/', '')
        DataClass = VoronoiDataset
    elif 'ithor' in args.data_dir:
        data_name = 'ithor' + args.data_dir.split('ithor')[-1].replace('/', '')
        DataClass = iTHORDataset
    else:
        assert False, 'Unknown dataset'

    train_dataset = DataClass(
        data_folder=args.data_dir, split='train', single_image=True, seq_len=1, cluster=args.cluster)
    val_dataset = DataClass(
        data_folder=args.data_dir, split='val', single_image=True, seq_len=1, cluster=args.cluster)
    test_dataset = DataClass(
        data_folder=args.data_dir, split='test_indep', single_image=True, seq_len=1,
        causal_vars=train_dataset.target_names(), cluster=args.cluster)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)
    print(f'Length training dataset: {len(train_dataset)} / Train loader: {len(train_loader)}')
    print(f'Length val dataset: {len(val_dataset)} / Test loader: {len(val_loader)}')

    model_args = vars(args)
    model_args['img_width'] = train_dataset.get_img_width()
    model_args['max_iters'] = args.max_epochs * len(train_loader)
    if hasattr(train_dataset, 'get_inp_channels'):
        model_args['c_in'] = train_dataset.get_inp_channels()
    print(f'Image size: {model_args["img_width"]}')

    model_args['data_folder'] = [s for s in args.data_dir.split('/') if len(s) > 0][-1]

    model_name = 'VAE'
    model_class = VAE
    logger_name = f'{model_name}_{args.num_latents}l_{train_dataset.num_vars()}b_{args.c_hid}hid_{data_name}'
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name

    print_params(logger_name, model_args)
    
    check_val_every_n_epoch = model_args.pop('check_val_every_n_epoch')
    if check_val_every_n_epoch <= 0:
        check_val_every_n_epoch = 1 if not args.cluster else 5

    train_model(model_class=model_class,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                logger_name=logger_name,
                check_val_every_n_epoch=check_val_every_n_epoch,
                progress_bar_refresh_rate=0 if args.cluster else 1,
                callback_kwargs={'dataset': train_dataset,
                                 'correlation_dataset': val_dataset,
                                 'correlation_test_dataset': test_dataset,
                                 }, #'action_data_loader': data_loaders['action']
                save_last_model=True,
                action_size=train_dataset.action_size(),
                causal_var_info=train_dataset.get_causal_var_info(),
                **model_args)
