from torch.autograd import Function

from modules.functional.backend import _backend


class ThreeNearestNeighborsInterpolation(Function):
    @staticmethod
    def forward(ctx, points_coords, centers_coords, centers_features):
        """
        :param ctx:
        :param points_coords: coordinates of points, FloatTensor[B, 3, N]
        :param centers_coords: coordinates of centers, FloatTensor[B, 3, M]
        :param centers_features: features of centers, FloatTensor[B, C, M]
        :return:
            points_features: features of points, FloatTensor[B, C, N]
        """
        centers_coords = centers_coords.contiguous()
        points_coords = points_coords.contiguous()
        centers_features = centers_features.contiguous()
        points_features, indices, weights = _backend.three_nearest_neighbors_interpolate_forward(
            points_coords, centers_coords, centers_features
        )
        ctx.save_for_backward(indices, weights)
        ctx.num_centers = centers_coords.size(-1)
        return points_features

    @staticmethod
    def backward(ctx, grad_output):
        indices, weights = ctx.saved_tensors
        grad_centers_features = _backend.three_nearest_neighbors_interpolate_backward(
            grad_output.contiguous(), indices, weights, ctx.num_centers
        )
        return None, None, grad_centers_features

three_nearest_neighbors_interpolate = ThreeNearestNeighborsInterpolation.apply

#################################################################################

def k_nearest_neighbors(points_coords, centers_coords, k):
    """
    :param points_coords: coordinates of points, FloatTensor[B, 3, N]
    :param centers_coords: coordinates of centers, FloatTensor[B, 3, M]
    :param k: number of neighbors, Int
    :return:
        indices: indices of neighbors, IntTensor[B, K, N]
        weights: distance of neighbors, FloatTensor[B, K, N]
    """
    centers_coords = centers_coords.contiguous()
    points_coords = points_coords.contiguous()
    return _backend.k_nearest_neighbors(points_coords, centers_coords, k)

class KNearestNeighborsInterpolation(Function):
    @staticmethod
    def forward(ctx, centers_features, indices, weights):
        """
        :param ctx:
        :param centers_features: features of centers, FloatTensor[B, C, M]
        :param indices: indices of neighbors, IntTensor[B, K, N]
        :param weights: distance of neighbors, FloatTensor[B, K, N]
        :return:
            points_features: features of points, FloatTensor[B, C, N]
        """
        centers_features = centers_features.contiguous()
        
        points_features = _backend.k_nearest_neighbors_interpolate_forward(
            centers_features, indices, weights
        )
        ctx.save_for_backward(indices, weights)
        ctx.num_centers = centers_features.size(-1)
        return points_features

    @staticmethod
    def backward(ctx, grad_output):
        indices, weights = ctx.saved_tensors
        grad_centers_features = _backend.k_nearest_neighbors_interpolate_backward(
            grad_output.contiguous(), indices, weights, ctx.num_centers
        )
        return grad_centers_features, None, None

k_nearest_neighbors_interpolate = KNearestNeighborsInterpolation.apply

class KNearestNeighborsWeightedInterpolation(Function):
    @staticmethod
    def forward(ctx, centers_features, indices, weights):
        """
        :param ctx:
        :param centers_features: features of centers, FloatTensor[B, C, M]
        :param indices: indices of neighbors, IntTensor[B, K, N]
        :param weights: distance of neighbors, FloatTensor[B, K, N]
        :return:
            points_features: features of points, FloatTensor[B, C, N]
        """
        centers_features = centers_features.contiguous()
        
        points_features = _backend.k_nearest_neighbors_interpolate_forward(
            centers_features, indices, weights
        )
        ctx.save_for_backward(indices, weights, centers_features)
        ctx.num_centers = centers_features.size(-1)
        return points_features

    @staticmethod
    def backward(ctx, grad_output):
        indices, weights, centers_features = ctx.saved_tensors
        grad_centers_features, grad_weights = _backend.k_nearest_neighbors_weighted_interpolate_backward(
            grad_output.contiguous(), indices, weights, centers_features, ctx.num_centers
        )
        return grad_centers_features, None, grad_weights

k_nearest_neighbors_weighted_interpolate = KNearestNeighborsWeightedInterpolation.apply