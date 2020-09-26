import modules.functional as F
import torch
import numpy as np
import time

import unittest

b = 1
r = 2
n = 1
c = 1

features = torch.randn(b, c, n).cuda()

coords = torch.clamp(torch.rand(b, 3, n) * r, 0, r).cuda()

print(features)
print(coords)

print('====================================')

vox = F.trilinear_voxelize(features, coords, r)

print('====================================')

print(vox)
