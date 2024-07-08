from typing import Any, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from collections import defaultdict

import sys

sys.path.append('../')
from models.shared import CosineWarmupScheduler, get_act_fn, Encoder, Decoder, SimpleEncoder, SimpleDecoder, VAESplit, \
    ImageLogCallback, PermutationCorrelationMetricsLogCallback
from models.shared import AutoregNormalizingFlow, gaussian_log_prob
from models.shared import create_interaction_prior, InteractionVisualizationCallback


def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, correlation_dataset=None,
                  correlation_test_dataset=None, action_data_loader=None, **kwargs):
    callbacks = [LearningRateMonitor('step')]
    if exmp_inputs is not None:
        img_callback = ImageLogCallback(exmp_inputs, dataset, every_n_epochs=10 if not cluster else 50,
                                        cluster=cluster)
        callbacks.append(img_callback)
    if correlation_dataset is not None:
        corr_callback = PermutationCorrelationMetricsLogCallback(correlation_dataset, cluster=cluster,
                                                                 test_dataset=correlation_test_dataset)
        callbacks.append(corr_callback)
    if action_data_loader is not None:
        actionvq_callback = InteractionVisualizationCallback(action_data_loader=action_data_loader)
        callbacks.append(actionvq_callback)
    return callbacks


class VAE(pl.LightningModule):
    """ The main module implementing BISCUIT-VAE """

    def __init__(self, c_hid, num_latents, lr, action_size=-1,
                 warmup=100, max_iters=100000,
                 img_width=64,
                 c_in=3,
                 decoder_num_blocks=1,
                 act_fn='silu',
                 no_encoder_decoder=False,
                 linear_encoder_decoder=False,
                 use_flow_prior=True,
                 decoder_latents=-1,
                 prior_action_add_prev_state=False,
                 **kwargs):
        """
        Parameters
        ----------
        c_hid : int
                Hidden dimensionality to use in the network
        num_latents : int
                      Number of latent variables in the VAE
        lr : float
             Learning rate to use for training
        action_size : int
                      Size of the action space.
        warmup : int
                 Number of learning rate warmup steps
        max_iters : int
                    Number of max. training iterations. Needed for 
                    cosine annealing of the learning rate.
        img_width : int
                    Width of the input image (assumed to be equal to height)
        c_in : int
               Number of input channels (3 for RGB)
        decoder_num_blocks : int
                             Number of residual blocks to use per dimension in the decoder.
        act_fn : str
                 Activation function to use in the encoder and decoder network.
        no_encoder_decoder : bool
                             If True, no encoder or decoder are initialized. Used for BISCUIT-NF
        linear_encoder_decoder : bool
                                 If True, the encoder and decoder are simple MLPs.
        use_flow_prior : bool
                         If True, use a normalizing flow in the prior.
        decoder_latents : int
                          Number of latent variables to use as input to the decoder. If -1, equal to encoder.
                          Can be used when additional variables are added to the decoder, e.g. for the action.
        prior_action_add_prev_state : bool
                                      If True, we consider the interaction variables to potentially depend on
                                      the previous state and add it to the MLPs.
        """
        super().__init__()
        self.save_hyperparameters()
        self.hparams.num_latents = num_latents
        act_fn_func = get_act_fn(self.hparams.act_fn)
        if self.hparams.decoder_latents < 0:
            self.hparams.decoder_latents = self.hparams.num_latents

        # Encoder-Decoder init
        if self.hparams.no_encoder_decoder:
            print('case1')
            self.encoder, self.decoder = nn.Identity(), nn.Identity()
        elif self.hparams.linear_encoder_decoder:
            print('case2')
            nn_hid = max(512, 2 * self.hparams.c_hid)
            self.encoder = nn.Sequential(
                nn.Linear(self.hparams.c_in, nn_hid),
                nn.SiLU(),
                nn.Linear(nn_hid, nn_hid),
                nn.SiLU(),
                nn.Linear(nn_hid, 2 * self.hparams.num_latents),
                VAESplit(self.hparams.num_latents)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.hparams.num_latents, nn_hid),
                nn.SiLU(),
                nn.Linear(nn_hid, nn_hid),
                nn.SiLU(),
                nn.Linear(nn_hid, self.hparams.c_in)
            )
        else:
            if self.hparams.img_width == 32:
                print('case3')
                self.encoder = SimpleEncoder(c_in=self.hparams.c_in,
                                             c_hid=self.hparams.c_hid,
                                             num_latents=self.hparams.num_latents)
                self.decoder = SimpleDecoder(c_in=self.hparams.c_in,
                                             c_hid=self.hparams.c_hid,
                                             num_latents=self.hparams.num_latents)
            else:
                print('case4')
                self.encoder = Encoder(num_latents=self.hparams.num_latents,
                                       c_hid=self.hparams.c_hid,
                                       c_in=self.hparams.c_in,
                                       width=self.hparams.img_width,
                                       act_fn=act_fn_func,
                                       variational=True)
                self.decoder = Decoder(num_latents=self.hparams.decoder_latents,
                                       c_hid=self.hparams.c_hid,
                                       c_out=self.hparams.c_in,
                                       width=self.hparams.img_width,
                                       num_blocks=self.hparams.decoder_num_blocks,
                                       act_fn=act_fn_func)

        # Logging
        self.all_val_dists = defaultdict(list)
        self.output_to_input = None
        self.register_buffer('last_target_assignment', torch.zeros(self.hparams.num_latents, 1))

    def forward(self, x):
        # Full encoding and decoding of samples
        z_mean, z_logstd = self.encoder(x)
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample)
        return x_rec, z_sample, z_mean, z_logstd

    def encode(self, x, random=True):
        # Map input to encoding, e.g. for correlation metrics
        z_mean, z_logstd = self.encoder(x)
        if random:
            z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        else:
            z_sample = z_mean
        return z_sample

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def loss_function(self, x, x_rec, z_mean, z_logstd, mode='train'):
        # Reconstruction loss (e.g., MSE)
        reconstruction_loss = F.mse_loss(x_rec, x, reduction='sum')

        # KL Divergence
        kld = -0.5 * torch.sum(1 + z_logstd - z_mean ** 2 - torch.exp(z_logstd))

        # Logging
        self.log(f'{mode}_kld_t1', kld.mean() / (x.shape[1]))
        self.log(f'{mode}_rec_loss_t1', reconstruction_loss.mean())
        # Total loss
        return reconstruction_loss + kld

    def _get_loss(self, batch, mode='train'):
        if len(batch) == 2:
            imgs, action = batch
            labels = imgs
        else:
            imgs, labels, action = batch

        z_mean, z_logstd = self.encoder(imgs)
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample)
        loss = self.loss_function(labels, x_rec, z_mean, z_logstd)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='train')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation/Testing with correlation matrices done via callbacks
        loss = self._get_loss(batch, mode='val')
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='test')
        self.log('test_loss', loss)
        return loss

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Need to do it explicitly since the size of the last_target_assignment might have changed
        if 'last_target_assignment' in checkpoint['state_dict']:
            self.last_target_assignment.data = checkpoint['state_dict']['last_target_assignment']

    @staticmethod
    def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, correlation_dataset=None,
                      correlation_test_dataset=None, action_data_loader=None, **kwargs):
        callbacks = [LearningRateMonitor('step')]
        if exmp_inputs is not None:
            img_callback = ImageLogCallback(exmp_inputs, dataset, every_n_epochs=10 if not cluster else 50,
                                            cluster=cluster)
            callbacks.append(img_callback)
        if correlation_dataset is not None:
            corr_callback = PermutationCorrelationMetricsLogCallback(correlation_dataset, cluster=cluster,
                                                                     test_dataset=correlation_test_dataset)
            callbacks.append(corr_callback)
        if action_data_loader is not None:
            actionvq_callback = InteractionVisualizationCallback(action_data_loader=action_data_loader)
            callbacks.append(actionvq_callback)
        return callbacks
