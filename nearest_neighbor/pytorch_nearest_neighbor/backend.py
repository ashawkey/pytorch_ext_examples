import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

_backend = load(name='_pvcnn_backend',
                extra_cflags=['-g', '-O3', '-fopenmp', '-lgomp'],
                extra_cuda_cflags=['-arch=compute_30', '-O3'],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    #'nearest_neighbor/nearest_neighbor_gpu.h', # headers shouldn't be added! this causes linking error: multiple definitions
                    'nearest_neighbor/nearest_neighbor_gpu.cpp',
                    'nearest_neighbor/nearest_neighbor_gpu.cu',
                    #'nearest_neighbor/nearest_neighbor_cpu.h',
                    'nearest_neighbor/nearest_neighbor_cpu.cpp',
                    'bindings.cpp',
                ]]
                )

__all__ = ['_backend']