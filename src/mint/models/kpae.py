import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchsummary import summary

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300

import wandb

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SpatialSoftmaxLayer(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.width=width
        self.height=height
        y_grid,x_grid=torch.meshgrid([torch.arange(0,height),torch.arange(0,width)])
        # Parameters in the model, which should be saved and restored in the state_dict, but not trained by the optimizer
        self.register_buffer('x_grid', x_grid.flatten())
        self.register_buffer('y_grid', y_grid.flatten())
    
    def forward(self, x):
        N,KP,H,W=x.shape
        img=x[0,0,:,:].detach().cpu().numpy()
        x=x.view(N,KP,H*W)
        # compute the softmax with the subtraction trick
        # subtract the maximum value doens't change the output of the softmax
        # good for numerical stability
        exp=torch.exp(x-torch.max(x,dim=-1,keepdim=True)[0])
        # the weights are the softmax of the feature maps
        weights=exp/(torch.sum(exp,dim=-1,keepdim=True)+1e-8)
        # compute the expected coordinates of the softmax
        # the expected values of the coordinates are the weighted sum of the softmax
        expected_x=torch.sum(weights*self.x_grid,dim=-1,keepdim=True)
        expected_y=torch.sum(weights*self.y_grid,dim=-1,keepdim=True)
        # concatenate the expected coordinates to the feature maps
        coords=torch.cat([expected_x,expected_y],dim=-1)
        # normalize the coordinates to [0,1]
        coords[...,0]/=self.width
        coords[...,1]/=self.height
        # return the expected coordinates
        return coords

class GaussiansAroundKeypoints(nn.Module):
    def __init__(self,height, width, std):
        super().__init__()
        self.width=width
        self.height=height
        x_vals, y_vals =(torch.arange(0, width), torch.arange(0, height))
        self.std=std
        # Parameters in the model, which should be saved and restored in the state_dict, but not trained by the optimizer
        self.register_buffer('x_vals', x_vals.view(1,1,1,width))
        self.register_buffer('y_vals', y_vals.view(1,1,height,1))
        
    def forward(self, x):
        N,KP,_=x.shape
        mu_x=x[...,0,None,None]*self.width
        mu_y=x[...,1,None,None]*self.height
        exp_x=(self.x_vals-mu_x)**2
        exp_y=(self.y_vals-mu_y)**2
        exp=(exp_x+exp_y)/(self.std**2)
        gaussian=torch.exp(-exp) 
        return gaussian
        
 
class KeypointDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.active_threshold=config["activation_score_threshold"]
        self.std=config['std_for_heatmap']
        C=config["channels"]
        W=config["width"]
        H=config["height"]
        KP=config['num_keypoints']
        inputs= [ C, KP, KP]
        outputs= inputs[1:]+[2*KP]
        kernels=[ 5, 3, 3]
        strides=[ 3, 2, 2]
        paddings=[ 1, 1, 1]
        if W<100:
            strides[-1]=1
        # create the hourglass architecture
        layers=[]
        for i in range(len(inputs)):
            layers.append(nn.Conv2d(inputs[i], outputs[i], kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.BatchNorm2d(outputs[i]))
        inputs= [2*KP, 2*KP, KP]
        outputs= inputs[1:]+[KP]
        kernels=[3, 3, 3]
        strides=[1, 2, 2]
        if W<100:
            strides[0]=2
        for i in range(len(inputs)):
            layers.append(nn.ConvTranspose2d(inputs[i], outputs[i], kernel_size=kernels[i], stride=strides[i]))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.BatchNorm2d(outputs[i]))
        layers.append(nn.Softplus())
        self.model=nn.Sequential(*layers)
        HH,WW=self.model(torch.zeros(1,C,H,W)).shape[-2:]
        # add the spatial softmax layer
        self.coordinates_from_feature_maps=SpatialSoftmaxLayer(HH,WW)
        self.gaussians=GaussiansAroundKeypoints(HH, WW, self.std)
        self.model.apply(self.weight_init)
        # normalize data
        self.normalize=transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        self.count=0
    
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # normalize
        x=self.normalize(x.float())
        feature_maps=self.model(x)
        keypoints=self.coordinates_from_feature_maps(feature_maps)
        activation_score=torch.amax(feature_maps,dim=(-1,-2))
        # wandb.log({'activation_score_{0}'.format(i):activation_score[0,i].item() for i in range(activation_score.shape[1])})
        status=torch.sigmoid(1000*(activation_score-self.active_threshold)).unsqueeze(-1)
        self.count+=1
        if self.training:
            gaussian=self.gaussians(keypoints)
            return keypoints, status, gaussian
        else:
            return keypoints, status