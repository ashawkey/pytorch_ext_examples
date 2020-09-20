import torch
from torch.autograd import Function
import torch.nn as nn

import add_cpp


# Function
class _add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return add_cpp.forward(x, y)

    @staticmethod
    def backward(ctx, gradOutput):
        gradX, gradY = add_cpp.backward(gradOutput)
        return gradX, gradY

add = _add.apply


# Module
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return add(x, y)
