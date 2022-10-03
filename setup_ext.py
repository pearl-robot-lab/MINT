from setuptools import setup
from torch.utils import cpp_extension
import os

# export cuda path
os.environ["CUDA_HOME"] = "~/miniconda3/envs/mint/"

setup(name='entropy_layer',
      ext_modules=[
            cpp_extension.CUDAExtension(name='entropy_layer',
                  sources=['src/mint/entropy_layer/entropy_cuda.cpp',
                   'src/mint/entropy_layer/entropy_cuda_kernel.cu'],
                  )
            ],
      cmdclass={'build_ext': cpp_extension.BuildExtension} )