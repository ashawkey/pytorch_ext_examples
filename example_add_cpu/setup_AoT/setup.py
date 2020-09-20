import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

source_files = glob.glob('src/*.cpp') + ['bindings.cpp']

'''
name: import name in python, should be the same as in bindings.cpp
'''
setup(
    name = 'add_cpp',
    ext_modules = [
        CppExtension(
            name = 'add_cpp', 
            sources = source_files,
            extra_compile_args = {
                'cxx': ['-O3'],
            }
        ),
    ],
    cmdclass = {'build_ext': BuildExtension}
)
