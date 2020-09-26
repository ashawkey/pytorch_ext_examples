import modules.functional as F
import torch
import numpy as np
import time

import unittest

def batch_pairwise_squared_distances(x, y):
  '''                                                                                              
  Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3         
  Input: x is a bxNxd matrix y is an optional bxMxd matirx                                                             
  Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
  i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2                                                         
  '''                                                                                              
  x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
  y_t = y.permute(0,2,1).contiguous()
  y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])
  dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
  dist[dist != dist] = 0 # replace nan values with 0
  return torch.clamp(dist, 0.0, np.inf)

def pytorch_k_nearest_neighbours(points_coords, center_coords, k):
    dists = batch_pairwise_squared_distances(points_coords.permute(0,2,1), center_coords.permute(0,2,1)) # [B, N, M]
    dists, indices = torch.topk(dists, k, dim=2, largest=False, sorted=True)
    return indices.permute(0,2,1), dists.permute(0,2,1)

class TestKNN(unittest.TestCase):

  def test_correctness(self):

    points_coords = torch.randn(8, 3, 10240)
    center_coords = torch.randn(8, 3, 2048)
    points_coords_cuda = points_coords.cuda()
    center_coords_cuda = center_coords.cuda()
    k = 3

    t0 = time.time()
    indices_cuda, dists_cuda = F.k_nearest_neighbors(points_coords_cuda, center_coords_cuda, k)
    t1 = time.time()
    indices_cpu, dists_cpu = pytorch_k_nearest_neighbours(points_coords, center_coords, k)
    t2 = time.time()

    print(f'cuda = {t1-t0:.6f}, cpu = {t2-t1:.6f}')
    self.assertTrue(torch.allclose(indices_cuda.cpu().long(), indices_cpu))

if __name__ == '__main__':
  unittest.main()