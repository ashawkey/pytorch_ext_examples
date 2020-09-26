import torch.nn as nn

import modules.functional as F


class KLLoss(nn.Module):
    def forward(self, x, y):
        return F.kl_loss(x, y)
