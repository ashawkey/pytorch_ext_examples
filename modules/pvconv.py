import torch.nn as nn

import modules.functional as F
from modules.voxelization import TrilinearVoxelization
from modules.shared_mlp import SharedMLP
from modules.se import SE3d


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = TrilinearVoxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        if len(inputs) == 2:
            features, coords = inputs
        else:
            features, coords, indices, weights = inputs

        # coords are normalized to voxel_coords before voxelization!
        voxel_features, voxel_coords = self.voxelization(features, coords)

        voxel_features = self.voxel_layers(voxel_features)

        # voxel_coords is in [0, r-1]
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        fused_features = voxel_features + self.point_features(features)

        if len(inputs) == 2:
            return fused_features, coords
        else:
            return fused_features, coords, indices, weights

