import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.shared.modules import CosineWarmupScheduler, TanhScaled


class CausalMLP(nn.Module):
    def __init__(self, input_dim=40, output_dim=10, hidden_dim=128, lr=4e-3, weight_decay=0.0):
        super(CausalMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.lr,
                                weight_decay=self.weight_decay)
        #lr_scheduler = CosineWarmupScheduler(optimizer,
        #                                     warmup=self.warmup,
        #                                     max_iters=self.max_iters)
        return optimizer #, [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, inps, target):
        pred = self.forward(inps)
        return F.mse_loss(pred, target)