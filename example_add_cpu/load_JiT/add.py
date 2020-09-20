import torch
from torch.autograd import Function
import torch.nn as nn

from backend import _backend

'''
Encapsulation of raw methods.
pytorch has functional and Module level encapsulations.
'''

# Function
class _add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return _backend.forward(x, y)

    @staticmethod
    def backward(ctx, gradOutput):
        gradX, gradY = _backend.backward(gradOutput)
        return gradX, gradY

add = _add.apply

# Module
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return add(x, y)
