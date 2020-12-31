from pytorch_nearest_neighbor import nearest_neighbor

import numpy as np

import torch

def nearest_neighbor_naive(A, B):
    dist = torch.cdist(A.unsqueeze(0), B.unsqueeze(0)).squeeze(0)
    idx = torch.argmin(dist, dim=1)
    return idx


A = torch.randn(10000, 3)
B = torch.randn(10000, 3)

from time import time

ts = time()
idx = nearest_neighbor_naive(A, B)
te = time()
print(idx[:100])
print(f'cpu naive {te-ts:.4f} s')

ts = time()
idx, _ = nearest_neighbor(A, B)
te = time()
print(idx[:100])
print(f'cpu openmp {te-ts:.4f} s')

t1 = time()
A = A.cuda()
B = B.cuda()
t2 = time()
idx, _ = nearest_neighbor(A, B)
torch.cuda.synchronize()
t3 = time()
print(idx[:100])
print(f'gpu cuda {t3-t2:.4f} + {t2-t1:.4f} s')
