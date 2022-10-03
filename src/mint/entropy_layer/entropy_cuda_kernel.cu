# include <torch/types.h>
  
#include <ATen/ATen.h>

# include <cuda.h>
# include <cuda_runtime.h>

namespace{
    // the sigmoid function
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
        return 1.0 / (1.0 + exp(-z));
    }
    // the derivative of the sigmoid function
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }
    // the entropy function
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t entropy(scalar_t z) {
        return -z * log(z+1e-8);
    }
    // the derivative of the entropy function
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t d_entropy(scalar_t z) {
        return -1.0 / log(10.0) * (1.0+log(z+1e-8));
    }
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t hist_bin(scalar_t d, scalar_t L, scalar_t B) {
        return sigmoid((d+L/2)/B) - sigmoid((d-L/2)/B);
    }
    // the derivative of the kernel function
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t d_hist_bin(scalar_t d, scalar_t L, scalar_t B) {
        return 1/B * (d_sigmoid((d+L/2)/B) - d_sigmoid((d-L/2)/B));
    }
    // constant memory for the max probability
    __constant__ float d_max;
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __global__ void rgb_entropy_cuda_forward_kernel(torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
                                                        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> entropy_output,
                                                        float L,
                                                        float B,
                                                        int patch_size                                                       
                                                        ){
        // the image index
        int n = blockIdx.y;
        // the index of the first patch
        int p = blockIdx.x;
        // the thread index
        int t = threadIdx.x;
        float hist=0;
        // Rule of sum of probabilities
        #pragma unroll
        for(int i=0;i<patch_size;i++){
            hist += hist_bin(input[n][p][i][0]-t,L,B);
            hist += hist_bin(input[n][p][i][1]-t,L,B);
            hist += hist_bin(input[n][p][i][2]-t,L,B);
        }
        float prob = hist / (3.0*patch_size*d_max);
        // update the output
        atomicAdd(&entropy_output[n][p],entropy(prob));
    }
    // the backward kernel function to compute the gradient
    template<typename scalar_t>
    __global__ void rgb_entropy_cuda_backward_kernel(torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
                                                torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_entropy_out,
                                                torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_out,
                                                float L,
                                                float B,
                                                int patch_size
                                                ){
        // the image index
        int n = blockIdx.y;
        // the index of the patch
        int p = blockIdx.x;
        // the thread index
        int t = threadIdx.x;
        float hist = 0;
        #pragma unroll
        for(int i=0;i<patch_size;i++){
            hist += hist_bin(input[n][p][i][0]-t,L,B);
            hist += hist_bin(input[n][p][i][1]-t,L,B);
            hist += hist_bin(input[n][p][i][2]-t,L,B);
        }
        float prob = hist / (3.0*patch_size*d_max);
        // the derivative of the entropy function
        float d_prob = d_entropy_out[n][p] * d_entropy(prob);
        float d_hist = 0;
        // compute the gradient
        #pragma unroll
        for(int i=0;i<patch_size;i++){
            d_hist += d_hist_bin(input[n][p][i][0]-t,L,B);
            d_hist += d_hist_bin(input[n][p][i][1]-t,L,B);
            d_hist += d_hist_bin(input[n][p][i][2]-t,L,B);
            grad_out[n][p][i] += d_prob * d_hist / (3.0*patch_size*d_max);
        }
    }
} // namespace

// the forward pass of the entropy layer
// x: input tensor
// bandwidth: the bandwidth of the kernel
torch::Tensor entropy_cuda_forward(torch::Tensor x, float bandwidth){
    // the parameters for the kernel function
    const float L=1.0/256.0;
    const float B=bandwidth;
    // move max to the constant memory
    const float max_=1.0 / (1.0 + exp(-(L/2)/B)) - 1.0 / (1.0 + exp((L/2)/B));;
    cudaMemcpyToSymbol(d_max, &max_, sizeof(float));
    // get the shape of the input tensor
    // N x P x R x C # C=3
    int N = x.size(0);
    int P = x.size(1);
    int R = x.size(2);
    int C = x.size(3);
    // block size
    dim3 threads(256);
    // grid size
    dim3 grid(P,N);
    // define the output tensor
    // N x P
    auto entropy_output = torch::zeros({N,P}).to(x.device());
    cudaFuncSetAttribute(rgb_entropy_cuda_forward_kernel<float>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
    cudaFuncSetCacheConfig(rgb_entropy_cuda_forward_kernel<float>,cudaFuncCachePreferL1);
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"entropy_cuda_forward",([&]{
    rgb_entropy_cuda_forward_kernel<float><<<grid,threads>>>(
        x.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        entropy_output.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            L,  B,  R);
    }));
    cudaDeviceSynchronize();
    // return the output
    return entropy_output;
}

// the backward pass of the entropy layer
// x: input tensor
// bandwidth: the bandwidth of the kernel
torch::Tensor entropy_cuda_backward(torch::Tensor x,
                            torch::Tensor d_entropy_out,
                            float bandwidth){
    // the parameters for the kernel function
    const float L=1.0/256.0;
    const float B=bandwidth;
    // move max to the constant memory
    const float max_=1.0 / (1.0 + exp(-(L/2)/B)) - 1.0 / (1.0 + exp((L/2)/B));;
    cudaMemcpyToSymbol(d_max, &max_, sizeof(float));
    // get the shape of the input tensor
    // N x P x R x C # C=3
    int N = x.size(0);
    int P = x.size(1);
    int R = x.size(2);
    int C = x.size(3);
    // block size
    dim3 threads(256);
    // grid size
    dim3 grid(P,N);
    // define the output tensor (the gradient)
    // N x P x R 
    auto grad_out = torch::zeros({N,P,R}).to(x.device());
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"entropy_cuda_backward",([&]{
    rgb_entropy_cuda_backward_kernel<float><<<grid,threads>>>(
        x.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        d_entropy_out.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        grad_out.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            L,  B,  R);
    }));
    cudaDeviceSynchronize();
    // return the output
    return grad_out;
}