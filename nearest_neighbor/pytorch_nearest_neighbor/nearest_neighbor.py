from torch.autograd import Function

from . backend import _backend

def nearest_neighbor(A, B):
    A = A.contiguous()
    B = B.contiguous()

    if 'cuda' in str(A.device):
        idx = _backend.nearest_neighbor_gpu(A, B)
    else:
        idx = _backend.nearest_neighbor_cpu(A, B)
    
    return idx