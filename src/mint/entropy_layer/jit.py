from torch.utils.cpp_extension import load
lltm_cuda = load(
    'entropy_layer', ['src/mint/entropy_layer/entropy_cuda.cpp',
                   'src/mint/entropy_layer/entropy_cuda_kernel.cu'], verbose=True)
help(lltm_cuda)