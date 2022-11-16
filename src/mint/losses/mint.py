from calendar import different_locale
from mint.entropy_layer.entropy import Entropy
from mint.utils.plot_entropy import *
from torchvision.transforms.functional import adjust_sharpness

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size']= 5
# matplotlib.use('Qt5Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MINT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.SF=config['number_of_stacked_frames']
        self.h_threshold=config['threshold_for_heatmap']
        self.thresholded_heatmap_scale=config['thresholded_heatmap_scale']
        self.num_keypoints=config['num_keypoints']
        self.entropy_layer=Entropy(config['region_size'],config['bandwidth']).to(device)
        self.masked_entropy_loss_weight=config['masked_entropy_loss_weight']
        self.conditional_entropy_loss_weight=config['conditional_entropy_loss_weight']
        self.information_transport_loss_weight=config['information_transport_loss_weight']
        self.movement_weight = config['movement_weight']
        self.beta=config['beta_for_overlapping_loss']
        self.kappa=config['kappa_for_it_loss']
        self.status_weight=config['status_weight']
        self.overlap_weight=config['overlap_weight']
        self.count=0

    def forward(self, coords, gaussians, active_status, images):
        N,KP,_=coords.shape
        N,KP,HH,WW=gaussians.shape
        N,C,H,W=images.shape
        # resize the images to the size of the heatmaps
        images=F.interpolate(images, size=(HH,WW), mode='bilinear', align_corners=False)
        if W<100:
            images=(adjust_sharpness(images/255, 10.0)*255).round()
        # split the batches into sequences N -> N' x SF
        coords=coords.reshape(-1,self.SF,KP,2)
        active_status=active_status.reshape(-1,self.SF,KP)
        gaussians=gaussians.reshape(-1,self.SF,KP,HH,WW)
        images=images.reshape(-1,self.SF,C,HH,WW)
        # threshold the heatmaps for information maximization
        heatmaps=gaussians-self.h_threshold
        heatmaps=self.thresholded_heatmap_scale*F.relu(heatmaps)
        heatmaps=torch.clamp(heatmaps,min=0,max=1)
        heatmap_size=heatmaps[0,0,0].sum()
        # print(heatmap_size)
        # plt.imshow(heatmaps[0,0,0].detach().cpu().numpy(),cmap='gray')
        # plt.show()
        #########################################
        ######### Entropy computation ###########
        #########################################
        # Entropy of the images (current frame)
        # H(Y)
        # N x SF x H x W
        rgb_entropy=self.entropy_layer(images)
        # shift one step forward SF [0,1,2,3] -> [3,0,1,2]
        # Emtropy of the images (previous frame)
        # H(X)
        shifted_rgb_entropy=torch.roll(rgb_entropy,1,dims=1)
        # Joint entropy
        # H(X,Y) = max(H(X),H(Y))
        # j [-,(1,0),(2,1),(3,2)]
        joint_entropy=torch.maximum(rgb_entropy,shifted_rgb_entropy)
        # Conditional entropy
        # H(Y|X) = H(X,Y) - H(X)
        # j [-,(1,0),(2,1),(3,2)] - e[3,0,1,2] = c[-,1,2,3]
        conditional_entropy=joint_entropy-shifted_rgb_entropy
        # plot_conditional_entropy(images[0],conditional_entropy[0],joint_entropy[0],rgb_entropy[0])
        #########################################
        ######### Information Maximization ######
        #########################################
        # sum the entropy of each image
        # N x SF
        rgb_entropy_sum=rgb_entropy.sum(dim=(-1,-2))
        conditional_entropy_sum=conditional_entropy.sum(dim=(-1,-2))
        # multiply the feature maps with the status to consider only the active keypoints
        # N x SF x KP x H x W
        active_heatmaps=active_status[...,None,None]*heatmaps
        # aggregate feature maps of all keypoints
        # N x SF x H x W
        aggregated_active_heatmaps=active_heatmaps.sum(dim=2)
        aggregated_active_mask=torch.clamp(aggregated_active_heatmaps,min=0,max=1)
        # mask the entropy
        # N x SF x H x W
        masked_entropy=rgb_entropy*aggregated_active_mask
        masked_conditional_entropy= conditional_entropy*aggregated_active_mask
        # we want to encourage maximizing the entropy in the masked regions
        # sum the masked entropy for each time frame
        # N x SF
        # masked_entropy_sum=torch.sum(masked_entropy,dim=(-1,-2))
        # masked_conditional_entropy_sum=torch.sum(masked_conditional_entropy,dim=(-1,-2))
        # the loss is the percebtage of the entropy in the masked regions
        # masked_entropy_loss=1-masked_entropy_sum/(rgb_entropy_sum+1e-10)
        masked_entropy_loss=(rgb_entropy-masked_entropy).sum(dim=(-1,-2))/(rgb_entropy_sum+1e-10)
        # masked_conditional_entropy_loss=1-masked_conditional_entropy_sum/(conditional_entropy_sum+1e-10)
        masked_conditional_entropy_loss=(conditional_entropy-masked_conditional_entropy).sum(dim=(-1,-2))/(conditional_entropy_sum+1e-10)
        # we have to drop the first frame since we don't have the previous frame
        masked_conditional_entropy_loss[:,0]=0
        #######################################
        ####### Information transport #########
        #######################################
        shifted_coords=torch.roll(coords,1,dims=1)
        # N x SF x KP
        distance_travelled=torch.norm(coords-shifted_coords,dim=-1)
        # N x SF x KP
        shifted_active_status=torch.roll(active_status,1,dims=1)
        distance_travelled = distance_travelled * active_status * shifted_active_status
        # N x SF
        movement_cost=distance_travelled.sum(dim=-1)
        # target is the current frame = rgb_entropy H(Y)
        # source is the previous frame = shifted_rgb_entropy H(X)
        # heatmaps represnt the patch of information carried by the keypoint in the current frame KI(Y)
        # shifted heatmaps represent the patch of information carried by the keypoint in the previous frame KI(X)
        # N x SF x KP x H x W
        shifted_heatmaps=torch.roll(active_heatmaps,1,dims=1)
        # Remove the two patches from the source frame
        # source information I(S)
        # H(S)= H(X) * (1-KI(X)) * (1-KI(Y))
        # N x SF x KP x H x W
        source_info = shifted_rgb_entropy[:,:,None,:,:]*(1-active_heatmaps)*(1-shifted_heatmaps)
        # Get the patch of information in the target frame and add the new information outside the patch
        # target information I(T)
        # H(T)= H(Y) * KI(Y) + H(Y|X) * (1-KI(Y))
        # N x SF x KP x H x W
        target_info= rgb_entropy[:,:,None,:,:] * active_heatmaps + self.kappa*conditional_entropy[:,:,None,:,:] * (1-active_heatmaps)
        # Inplant the target information into the source information
        # the reconstructed information I(R)
        # H(R) = H(S) + H(T)
        # N x SF x KP x H x W
        reconstructed_information = source_info + target_info
        # Compute the mutual informaiotn between the reconstructed information and the current information
        # I(R,Y) = H(R) + H(Y) - H(R,Y) = H(R) + H(Y) - max(H(R),H(Y))
        # N x SF x KP x H x W
        mutual_information= reconstructed_information + rgb_entropy[:,:,None,:,:] - torch.maximum(reconstructed_information,rgb_entropy[:,:,None,:,:])
        # sum the mutual information
        # N x SF x KP
        mutual_information_sum=torch.sum(mutual_information,dim=(-1,-2))
        # plot_transport(images[0],rgb_entropy[0],conditional_entropy[0],source_info[0,:,0],target_info[0,:,0], reconstructed_information[0,:,0], mutual_information[0,:,0])
        # the mutual information is less or equal to the entropy of the current frame
        # sum the entropy of the current frame
        # I(R,Y) <= H(Y)
        # N x SF x 1
        sum_current=torch.sum(rgb_entropy[:,:,None],dim=(-1,-2))
        # minimize the difference between the mutual information and the entropy of the current frame
        # N x SF x KP
        reconstruction_cost=(sum_current-mutual_information_sum)/heatmap_size 
        # reconstruction_cost=torch.sum((rgb_entropy[:,:,None]-mutual_information)**2,dim=(-1,-2))*0.5
        reconstruction_cost=reconstruction_cost.mean(dim=-1)
        # mean for each time frame
        # N x SF
        information_transport_loss= reconstruction_cost + self.movement_weight * movement_cost
        #######################################
        ############ Status loss ##############
        #######################################
        # status loss is the sum of active status
        # N x SF
        status_sum=active_status.sum(dim=-1)
        status_loss = status_sum / KP
        #########################################
        ############ overlapping loss ###########
        #########################################
        # the maximum of the aggreagated heatmaps normalized by the number of keypoints
        # N x SF
        aggregated_gaussians= gaussians.sum(dim=2)
        overlapping_loss=torch.clamp(torch.amax(aggregated_gaussians,dim=(-1,-2)) - self.beta, min=0) / self.num_keypoints
        #########################################
        ########## Total loss ###################
        #########################################
        # the mint loss
        # N x SF
        mint_loss = self.masked_entropy_loss_weight*masked_entropy_loss\
                    + self.conditional_entropy_loss_weight*masked_conditional_entropy_loss \
                    + self.information_transport_loss_weight*information_transport_loss \
                    + self.overlap_weight*overlapping_loss \
                    + (1-masked_entropy_loss) * self.status_weight*status_loss
        # mean over time
        # N
        mint_loss=mint_loss.mean(dim=-1)
        # mean over the batch
        mint_loss=mint_loss.mean()
        # log to wandb
        wandb.log({'mint_loss':mint_loss.item(),
            'mint/masked_entropy_percentage':masked_entropy_loss.mean().item(),
            'mint/conditional_entropy_percentage':masked_conditional_entropy_loss.mean().item(),
            'mint/information_transfer_loss':information_transport_loss.mean().item(),
            'mint/status_loss':status_loss.mean().item(),
            'mint/overlapping':overlapping_loss.mean().item(),
            })
        self.count+=1
        torch.cuda.empty_cache()
        return mint_loss