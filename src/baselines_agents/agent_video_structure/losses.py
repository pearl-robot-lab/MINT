"""Losses for the video representation model."""
  
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

import wandb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size']= 5
# matplotlib.use('Qt5Agg')

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SeparationLoss(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.sigma=config['sigma_for_separation_loss']
    self.weight=config['separation_loss_weight']

  def forward(self, coords):
    """Encourages keypoint to have different temporal trajectories.
      If two keypoints move along trajectories that are identical up to a time-
      invariant translation (offset), this suggest that they both represent the same
      object and are redundant, which we want to avoid.
      To measure this similarity of trajectories, we first center each trajectory by
      subtracting its mean. Then, we compute the pairwise distance between all
      trajectories at each timepoint. These distances are higher for trajectories
      that are less similar. To compute the loss, the distances are transformed by
      a Gaussian and averaged across time and across trajectories.
    """
    N,SF,KP,_=coords.shape
    # encourage keypoints to have different temporal trajectory
    x=coords[...,0]
    y=coords[...,1]
    # Center trajectories:
    x=x-torch.mean(x, dim=1, keepdim=True)
    y=y-torch.mean(y, dim=1, keepdim=True)
    # Compute pairwise distance matrix:
    d=(x.unsqueeze(-1)-x.unsqueeze(-2))**2.0+(y.unsqueeze(-1)-y.unsqueeze(-2))**2.0
    # Temporal mean:
    d= torch.mean(d, dim=1)
    # Apply Gaussian function such that loss falls off with distance:
    loss_matrix=torch.exp(-d/(2.0*self.sigma**2.0))
    # Mean across batch.
    loss_matrix=loss_matrix.mean(dim=0)
    # Sum matrix elements.
    separation_loss=loss_matrix.sum()
    # Subtract sum of values on diagonal, which are always 1:
    separation_loss-=KP
    # Normalize by maximal possible value. The loss is now scaled between 0 (all
    # keypoints are infinitely far apart) and 1 (all keypoints are at the same
    # location):
    separation_loss=separation_loss/(KP*(KP-1))
    # log the loss
    wandb.log({'Separation Loss':separation_loss})
    return separation_loss * self.weight

class ReconstructionLoss(nn.Module):
  def __init__(self, config):
    super(ReconstructionLoss, self).__init__()
    self.mse=nn.MSELoss(reduction='sum',reduce=True)
    self.count=0
    self.weight=config['reconstruction_loss_weight']
  
  def inv_normalize(self, x):
    x=(x+0.5)*255
    return x

  def forward(self, prediction, images):
    N,SF,C,H,W = images.shape
    # normalize the images
    images=images.float()
    # skip the first frame
    images=images[:,1:]
    # fig,ax=plt.subplots(2,4)
    # for i in range(4):
    #     ax[0,i].imshow(prediction[0,i].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    #     ax[1,i].imshow(images[0,i].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
    # plt.show()
    # the reconstruction loss is the mean squared error between the prediction and the ground truth
    # loss=self.mse(prediction,images)*0.5
    loss=torch.sum((prediction-images)**2,dim=[-1,-2,-3])*0.5
    # mean over batch and time
    loss=loss.mean()
    # log the loss to wandb
    wandb.log({'Reconstruction Loss':loss})
    self.count+=1
    if self.count%100==0:
      # log the prediciton and the ground truth as images to wandb
      wandb.log({'Prediction':wandb.Image(self.inv_normalize(prediction[0,3]).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1]),
                  'Ground Truth':wandb.Image(self.inv_normalize(images[0,3]).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])})
    return loss * self.weight

class SparsityLoss(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.weight=config['sparsity_loss_weight']
  
  def forward(self, heatmaps):
    """L1-loss on mean heatmap activations, to encourage sparsity."""
    N,SF,KP,HH,WW=heatmaps.shape
    heatmap_mean=heatmaps.mean(dim=(-1,-2))
    loss=torch.abs(heatmap_mean).mean()
    wandb.log({'Sparsity Loss':loss})
    return loss * self.weight
