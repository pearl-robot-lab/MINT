from torch import nn
from torch.autograd import Function
import torch

import matplotlib.pyplot as plt
import numpy as np

import entropy_layer

class EntropyFunction(Function):
    @staticmethod
    def forward(ctx, inputs, bandwidth):
        output=entropy_layer.forward(inputs, bandwidth)
        ctx.save_for_backward(inputs)
        ctx.bandwidth=bandwidth
        return output

    @staticmethod
    def backward(ctx, d_entropy):
        inputs= ctx.saved_tensors
        d_input=entropy_layer.backward(inputs, d_entropy, ctx.bandwidth)
        # None for the gradient of the bandwidth
        return d_input , None

class Entropy(nn.Module):
    def __init__(self, region_size, bandwidth):
        super(Entropy, self).__init__()
        self.region_size=region_size
        self.bandwidth=bandwidth
        # the blur is average blur of size 5x5
        self.blur=nn.AvgPool2d(kernel_size=(5,5), stride=1, padding=2,ceil_mode=True, count_include_pad=True)

    def forward(self, input):
        # get the size of the input
        N,SF,C,H,W=input.shape
        R=self.region_size
        # reshape the input to have a have a batch of images
        # N * SF x C x H xW
        input=input.view(-1,C,H,W)
        # blur the image
        smooth=self.blur(input).round()
        # sharp the image by add weighted sum of the blurred image
        # N * SF x C x H xW
        sharp=torch.clamp(2.5*input-1.25*smooth,min=0,max=255).round()
        # divide the image by the blur
        division=torch.clamp(torch.div(sharp*255,smooth+1e-8),min=0,max=255).round()
        # reshape the input
        # N x SF x C x H xW
        input=division.view(N,SF,C,H,W)
        # strides
        sN,sSF,sC,sH,sW=input.stride()
        # instead of iterating using for loops, we can factorize the computation by generating patches
        # overlapping patches of size (R x R)
        size=(N,SF,C,H-R+1,W-R+1,R,R)
        # strides
        stride=(sN,sSF,sC,sH,sW,sH,sW)
        # generate the patches
        # N x SF x C x H-R+1 x W-R+1 x R x R
        patches=input.as_strided(size, stride)
        # reshape to N*SF, C x (H-R+1) * (W-R+1) x R^2
        patches=patches.contiguous()
        patches=patches.view(N*SF, C,(H-R+1)*(W-R+1),R*R)
        # move channels to the last dimension
        patches=patches.permute(0,2,3,1)
        output=EntropyFunction.apply(patches, self.bandwidth)
        # reshape the output to N*SF*C x (H-R+1) x (W-R+1)
        output=output.view(N*SF,H-R+1,W-R+1)
        # add zero padding to the output to match the size of the input
        row_pad=torch.zeros(N*SF,H-R+1,int(R/2)).to(input.device)
        output=torch.cat([row_pad, output,row_pad], dim=2)
        col_pad=torch.zeros(N*SF,int(R/2),W).to(input.device)
        output=torch.cat([col_pad, output,col_pad], dim=1)
        output=output.view(N,SF,H,W)
        return output
