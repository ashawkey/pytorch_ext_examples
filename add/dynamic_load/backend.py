import os
import glob
from torch.utils.cpp_extension import load

source_files = glob.glob('src/*.cpp') + ['bindings.cpp']

'''
dynamic load of extensions.
    name: should be the same module_name as in bindings.cpp
'''
_backend = load(
    name='_backend',
    extra_cflags=['-O3', '-std=c++17'],
    sources=source_files,
)
