import torch
import torch.nn as nn

import modules.functional as F
from modules.ball_query import BallQuery
from modules.shared_mlp import SharedMLP


class PointNetAModule(nn.Module):
    def __init__(self, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]]
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels]

        mlps = []
        total_out_channels = 0
        for _out_channels in out_channels:
            mlps.append(
                SharedMLP(in_channels=in_channels + (3 if include_coordinates else 0),
                          out_channels=_out_channels, dim=1)
            )
            total_out_channels += _out_channels[-1]

        self.include_coordinates = include_coordinates
        self.out_channels = total_out_channels
        self.mlps = nn.ModuleList(mlps)

    def forward(self, inputs):
        features, coords = inputs
        if self.include_coordinates:
            features = torch.cat([features, coords], dim=1)
        coords = torch.zeros((coords.size(0), 3, 1), device=coords.device)
        if len(self.mlps) > 1:
            features_list = []
            for mlp in self.mlps:
                features_list.append(mlp(features).max(dim=-1, keepdim=True).values)
            return torch.cat(features_list, dim=1), coords
        else:
            return self.mlps[0](features).max(dim=-1, keepdim=True).values, coords

    def extra_repr(self):
        return f'out_channels={self.out_channels}, include_coordinates={self.include_coordinates}'


class PointNetSAModule(nn.Module):
    def __init__(self, num_centers, radius, num_neighbors, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        if not isinstance(radius, (list, tuple)):
            radius = [radius]
        if not isinstance(num_neighbors, (list, tuple)):
            num_neighbors = [num_neighbors] * len(radius)
        assert len(radius) == len(num_neighbors)
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]] * len(radius)
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels] * len(radius)
        assert len(radius) == len(out_channels)

        groupers, mlps = [], []
        total_out_channels = 0
        for _radius, _out_channels, _num_neighbors in zip(radius, out_channels, num_neighbors):
            groupers.append(
                BallQuery(radius=_radius, num_neighbors=_num_neighbors, include_coordinates=include_coordinates)
            )
            mlps.append(
                SharedMLP(in_channels=in_channels + (3 if include_coordinates else 0), out_channels=_out_channels, dim=2)
            )
            total_out_channels += _out_channels[-1]

        self.num_centers = num_centers
        self.out_channels = total_out_channels
        self.groupers = nn.ModuleList(groupers)
        self.mlps = nn.ModuleList(mlps)

    def forward(self, inputs):
        features, coords = inputs
        center_indices = F.furthest_point_sample(coords, self.num_centers) # [b, m]
        centers_coords = F.gather(coords, center_indices) # [b, 3, m]

        features_list = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            features_list.append(mlp(grouper(coords, centers_coords, features)).max(dim=-1).values)

        if len(features_list) > 1:
            return torch.cat(features_list, dim=1), centers_coords, center_indices
        else:
            return features_list[0], centers_coords, center_indices

    def extra_repr(self):
        return f'num_centers={self.num_centers}, out_channels={self.out_channels}'


class PointNetFPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP(in_channels=in_channels, out_channels=out_channels, dim=1)

    def forward(self, inputs):
        if len(inputs) == 3:
            points_coords, centers_coords, centers_features = inputs
            points_features = None
        else:
            points_coords, centers_coords, centers_features, points_features = inputs
        interpolated_features = F.three_nearest_neighbors_interpolate(points_coords, centers_coords, centers_features)
        if points_features is not None:
            interpolated_features = torch.cat(
                [interpolated_features, points_features], dim=1
            )
        return self.mlp(interpolated_features), points_coords

class PointNetSWFPModule(nn.Module):
    def __init__(self, in_channels, sa_in_channels, out_channels,k):
        super().__init__()
        self.k = k
        self.mlp = SharedMLP(in_channels=in_channels + sa_in_channels, out_channels=out_channels, dim=1)
        self.sim = nn.Sequential(
            nn.Conv2d(sa_in_channels*2, sa_in_channels, kernel_size=1),
            nn.BatchNorm2d(sa_in_channels),
            nn.ReLU(),
            nn.Conv2d(sa_in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        if len(inputs) == 3:
            print('[ERROR] point_features is None.')
            points_coords, centers_coords, centers_features = inputs 
            points_features = None
        else:
            points_coords, centers_coords, centers_features, points_features = inputs

        indices, dists = F.k_nearest_neighbors(points_coords, centers_coords, self.k) # [b, 3, n], [b, 3, m] -> [b, k, n]

        # swfp
        # centers_features = [b, in_ch, m], points_features = [b, sa_in_ch, n]
        features0 = F.grouping(points_features, indices) # [b, c, k, n]
        features1 = points_features.unsqueeze(2).repeat(1,1,self.k,1) # [b, c, k, n]

        weights = self.sim(torch.cat([features0, features1], dim=1)) # [b, 2c, k, n] -> [b, 1, k, n]
        weights = weights.squeeze(1) # [b, k, n]

        interpolated_features = F.k_nearest_neighbors_weighted_interpolate(centers_features, indices, weights)

        if points_features is not None:
            interpolated_features = torch.cat([interpolated_features, points_features], dim=1)

        return self.mlp(interpolated_features), points_coords, indices, weights