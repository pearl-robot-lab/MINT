"""Vision-related components of the structured video representation model.

These components perform the pixels <--> keypoints transformation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

from torchsummary import summary

import wandb

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size']= 5
# matplotlib.use('Qt5Agg')

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# vision models
class Images2Keypoints(nn.Module):
    """Builds a model that encodes an image sequence into a keypoint sequence.
        The model applies the reflect convolutional feature extractor to all images in
        the sequence. The feature maps are then reduced to num_keypoints heatmaps, and
        the heatmaps to (x, y, scale)-keypoints.
    """
    def __init__(self, config):
        super(Images2Keypoints, self).__init__()
        self.num_keypoints=config['num_keypoints']
        # instead of constraining the heatmap to be square and to have
        # the ration between in the input width and the heatmap width a perfect square
        # we use the heatmap_ratio as aperfect square no matter what it the size of the input
        heatmap_ratio=config['heatmap_ratio']
        layers_per_scale=config['layers_per_scale']
        # initial num filters for the encoder
        F=config['num_encoder_filters']
        KP=config['num_keypoints']
        # adjust channel number to account for add_coord_channels
        C=config['channels']+2
        # build feature extractor
        """Extracts feature maps from images.
            The encoder iteratively halves the resolution and doubles the number of
            filters until the size of the feature maps is output_map_width by
            output_map_width.
        """
        # conv layer args (in pytorch we have to define the input and the output channels)
        # first layer : expand the image to initial_num_filters whcih equals num_encoder_filters
        inputs= [ C, F]
        kernels= [ 3]
        strides=[ 1]
        padding='same'
        # add layers to the encoder
        for _ in range(layers_per_scale):
            inputs.append(F)
            kernels.append(3)
            strides.append(1)
        # Apply downsampling blocks until feature map width is output_map_width:
        width=config['width']
        heatmap_width=int(width/heatmap_ratio)
        while width > heatmap_width:
            width//=2
            F*=2
            inputs.append(F)
            kernels.append(3)
            strides.append(2)
            for _ in range(layers_per_scale):
                inputs.append(F)
                kernels.append(3)
                strides.append(1)
        # Build final layer that maps to the desired number of heatmaps:
        outputs= inputs[1:]+[KP]
        kernels.append(1)
        strides.append(1)
        # stack the layers
        layers=[]
        for i in range(len(inputs)):
            if strides[i]==1:
                layers.append(nn.Conv2d( inputs[i], outputs[i], kernel_size=kernels[i], stride=strides[i], padding=padding))
            else:
                # pytorch doesn't support same padding for strides > 1
                # add zero padding to get something similar to padding same in keras
                pad=int(kernels[i]/2)
                layers.append(nn.ZeroPad2d((pad,pad-1,pad,pad-1)))
                layers.append(nn.Conv2d(inputs[i], outputs[i], kernel_size=kernels[i], stride=strides[i]))
            if i<len(inputs)-1:
                # last layer has softplus as activation
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.BatchNorm2d(outputs[i],affine=True))
        # Heatmaps must be non-negative.
        layers.append(nn.Softplus())
        self.image_encoder=nn.Sequential(*layers)
        # initialize the weights with h2_uniform
        self.image_encoder.apply(self.weight_init)
        # we will return heatmaps and apply the sparsity loss outside the model
        # print(summary(self.image_encoder.to(device),(5,config['height'],config['width'])))
        self.count=0
    
    def weight_init(self, m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_uniform_(m.weight)
    
    def add_coord_channels(self, x):
        """Adds channels containing pixel indices (x and y coordinates) to an image.
            Note: This has nothing to do with keypoint coordinates. It is just a data
            augmentation to allow convolutional networks to learn non-translation-
            equivariant outputs. This is similar to the "CoordConv" layers:
            https://arxiv.org/abs/1603.09382.
        """
        N,C,H,W=x.shape
        # pixel coordinates
        # x_map - N x SF x 1 x H x W
        x_map = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(N,1,H,W)
        # y_map - N x SF x 1 x H x W is the transpose of x_map
        y_map = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(N,1,H,W)
        # concatenate x and y to images in the channel dimension
        x=torch.cat([x,x_map,y_map], dim=1)
        return x

    def heatmaps_to_keypoints(self, heatmaps):
        """Turns feature-detector heatmaps into (x, y, scale) keypoints.
            This function takes a tensor of feature maps as input. Each map is normalized
            to a probability distribution and the location of the mean of the distribution
            (in image coordinates) is computed. This location is used as a low-dimensional
            representation of the heatmap (i.e. a keypoint).
            To model keypoint presence/absence, the mean intensity of each feature map is
            also computed, so that each keypoint is represented by an (x, y, scale)
            triplet.
            Coordinate range is [-1, 1] for x and y, and [0, 1] for scale.
        """
        N,C,H,W=heatmaps.shape
        # soft argmax
        # extracting the expeected x
        x_grid=torch.linspace(-1,1,W,device=device)
        x_grid=x_grid.view(1,1,1,W)
        # fig,ax=plt.subplots(2,2)
        # Normalize the heatmaps to a probability distribution (i.e. sum to 1):
        weights_x=torch.sum(heatmaps+1e-10,dim=-2,keepdim=True)
        # the weights are the softmax of the feature maps
        weights_x=weights_x/(torch.sum(weights_x,dim=-1,keepdim=True)+1e-8)
        # compute the expected coordinates of the softmax
        # the expected values of the coordinates are the weighted sum of the x_grid
        expected_x=torch.sum(weights_x*x_grid,dim=-1,keepdim=True).squeeze(-1)
        # extracting the expeected y
        y_grid=torch.linspace(-1,1,H,device=device)
        y_grid=y_grid.view(1,1,H,1)
        # Normalize the heatmaps to a probability distribution (i.e. sum to 1):
        weights_y=torch.sum(heatmaps+1e-10,dim=-1,keepdim=True)
        # the weights are the softmax of the feature maps
        weights_y=weights_y/(torch.sum(weights_y,dim=-2,keepdim=True)+1e-8)
        # the expected values of the coordinates are the weighted sum of the x_grid
        expected_y=torch.sum(weights_y*y_grid,dim=-2,keepdim=True).squeeze(-1)
        # map scale
        map_scale=torch.mean(heatmaps,dim=(-1,-2)).unsqueeze(-1)
        # Normalize map scales to [0.0, 1.0] across keypoints. This removes a
        # degeneracy between the encoder and decoder heatmap scales
        map_scale=map_scale/(1e-10+torch.amax(map_scale,dim=-1,keepdim=True))
        # concatenate the expected coordinates and the map_scale
        coords=torch.cat([expected_x,expected_y, map_scale],dim=-1)
        return coords

    def forward(self, x):
        input_shape=len(x.shape)
        if input_shape>4:
          N,SF,C,H,W=x.shape
          x=x.view(N*SF,C,H,W)
        # images to keypoints
        # add the coordinate channels
        # N x SF x C+2 x H x W
        x=self.add_coord_channels(x)
        # pass to the image encoder to get the heatmaps
        # N x SF x KP x HH x WW
        x=self.image_encoder(x)
        # copy x to heatmaps variables to return it
        heatmaps=x.clone()
        # extract the keyoints from the heatmaps
        x=self.heatmaps_to_keypoints(x)
        self.count+=1
        if input_shape>4:
          # reshape to N x SF x KP x 3
          x=x.view(N,SF,self.num_keypoints,3)
          heatmaps=heatmaps.view(N,SF,*heatmaps.shape[1:])
        return x, heatmaps


class Keypoints2Images(nn.Module):
    """Builds a model to reconstructs an image sequence from a keypoint sequence.

    Model architecture:

      (keypoint_sequence, image[0], keypoints[0]) --> reconstructed_image_sequence

    For all frames image[t] we also we also concatenate the Gaussian maps for
    the keypoints obtained from the initial frame image[0]. This helps the
    decoder "inpaint" the image regions that are occluded by objects in the first
    frame.

    """
    def __init__(self, config):
        super(Keypoints2Images, self).__init__()
        self.channels=config['channels']
        self.width=config['width']
        self.num_of_stacked_frames=config['number_of_stacked_frames']
        self.num_keypoints=config['num_keypoints']
        heatmap_ratio=config['heatmap_ratio']
        layers_per_scale=config['layers_per_scale']
        # initial num filters for the encoder
        F=config['num_encoder_filters']
        KP=config['num_keypoints']
        # adjust channel number to account for add_coord_channels
        C=config['channels']
        # conv layer args
        self.kernel_size=3 #config['kernel_size']
        self.padding=config['padding'] # same
        # Build encoder net to extract appearance features from the first frame:
        """Extracts feature maps from images.
            The encoder iteratively halves the resolution and doubles the number of
            filters until the size of the feature maps is output_map_width by
            output_map_width.
        """
        # conv layer args (in pytorch we have to define the input and the output channels)
        # first layer : expand the image to initial_num_filters whcih equals num_encoder_filters
        inputs= [ C, F]
        kernels= [ 3]
        strides=[ 1]
        padding='same'
        # add layers to the encoder
        for _ in range(layers_per_scale):
            inputs.append(F)
            kernels.append(3)
            strides.append(1)
        # Apply downsampling blocks until feature map width is output_map_width:
        width=config['width']
        heatmap_width=int(width/heatmap_ratio)
        while width > heatmap_width:
            width//=2
            F*=2
            inputs.append(F)
            kernels.append(3)
            strides.append(2)
            for _ in range(layers_per_scale):
                inputs.append(F)
                kernels.append(3)
                strides.append(1)
        outputs= inputs[1:]+[F]
        kernels.append(3)
        strides.append(1)
        # stack the layers
        layers=[]
        for i in range(len(inputs)):
            if strides[i]==1:
                layers.append(nn.Conv2d(inputs[i], outputs[i], kernel_size=kernels[i], stride=strides[i], padding=padding))
            else:
                # pytorch doesn't support same padding for strides > 1
                # add zero padding to get something similar to padding same in keras
                pad=int(kernels[i]/2)
                layers.append(nn.ZeroPad2d((pad,pad-1,pad,pad-1)))
                layers.append(nn.Conv2d(inputs[i], outputs[i], kernel_size=kernels[i], stride=strides[i]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm2d(outputs[i],affine=True))
        self.image_encoder=nn.Sequential(*layers)
        # Build image decoder that goes from Gaussian maps to reconstructed images:
        """Decodes images from feature maps.
          The encoder iteratively doubles the resolution and halves the number of
          filters until the size of the feature maps is output_width.
        """
        decoder_input= F + KP*2 + 2
        inputs= [ decoder_input, F]
        kernels= [ 3]
        strides=[ 1]
        padding='same'
        upsample=[False]
        # upsample to get the same size as the input image
        while width<config['width']:
            width*=2
            upsample.append(True)
            inputs.append(F)
            kernels.append(3)
            strides.append(1)
            F//=2
            for _ in range(layers_per_scale):
                inputs.append(F)
                kernels.append(3)
                strides.append(1)
                upsample.append(False)
        # Build layers to adjust channel numbers for output image:
        outputs= inputs[1:]+[config['channels']]
        kernels.append(1)
        strides.append(1)
        upsample.append(False)
        # stack the layers
        layers=[]
        for i in range(len(inputs)):
            if upsample[i]:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
                continue
            layers.append(nn.Conv2d(inputs[i], outputs[i], kernel_size=kernels[i], stride=strides[i], padding=padding))
            if i!=len(inputs)-1: 
                layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm2d(outputs[i],affine=True))
        self.image_decoder=nn.Sequential(*layers)
        self.std=config['std_for_gaussian_maps']
        # initialize the weights with h2_uniform
        self.image_encoder.apply(self.weight_init)
        self.image_decoder.apply(self.weight_init)
        HH,WW=self.image_encoder(torch.zeros(1,C,config['height'],config['width'])).shape[-2:]
        # print(summary(self.image_encoder.to(device),(3,config['height'],config['width'])))
        # print(summary(self.image_decoder.to(device),(decoder_input,HH,WW)))
    
    
    def weight_init(self, m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_uniform_(m.weight)

    def keypoints_to_gaussian_maps(self, keypoints, H, W):
        """Turns (x, y, scale)-tuples into pixel maps with a Gaussian blob at (x, y).
        """
        N,KP,_=keypoints.shape
        # grid
        x_grid=torch.linspace(-1,1,W,device=device)
        x_grid=x_grid.view(1,1,1,W).repeat(N,KP,H,1)
        y_grid=torch.linspace(-1,1,H,device=device)
        y_grid=y_grid.view(1,1,H,1).repeat(N,KP,1,W)
        # Expand to [batch_size, num_keypoints, 1, 1] for broadcasting later:
        x_coordinates = keypoints[:, :, 0, None, None]
        y_coordinates = keypoints[:,:, 1, None, None]
        # Create two 1-D Gaussian vectors (marginals) and multiply to get a 2-d map:
        sigma = torch.tensor(self.std).to(device)
        keypoint_width = 2.0 * (sigma / W) ** 2.0
        keypoint_height = 2.0 * (sigma / H) ** 2.0
        x_vec = torch.exp(-torch.square(x_grid - x_coordinates)/keypoint_width)
        y_vec = torch.exp(-torch.square(y_grid - y_coordinates)/keypoint_height)
        maps = torch.multiply(x_vec, y_vec)
        # multiply by scale
        maps = maps * keypoints[:,:,2, None, None]
        # print(x_coordinates[0,0], y_coordinates[0,0])
        # plt.imshow(maps[0,0,:,:].detach().cpu().numpy(), cmap='jet')
        # plt.show()
        return maps

    def add_coord_channels(self, x):
        """Adds channels containing pixel indices (x and y coordinates) to an image.
            Note: This has nothing to do with keypoint coordinates. It is just a data
            augmentation to allow convolutional networks to learn non-translation-
            equivariant outputs. This is similar to the "CoordConv" layers:
            https://arxiv.org/abs/1603.09382.
        """
        N,C,H,W=x.shape
        # pixel coordinates
        # x_map - N x SF x 1 x H x W
        x_map = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(N,1,H,W)
        # y_map - N x SF x 1 x H x W is the transpose of x_map
        y_map = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(N,1,H,W)
        # concatenate x and y to images in the channel dimension
        x=torch.cat([x,x_map,y_map], dim=1)
        return x

    def forward(self, keypoints, first_image, heatmaps):
        N,SF,KP,_=keypoints.shape
        N,C,H,W=first_image.shape
        N,SF,KP,HH,WW=heatmaps.shape
        # generate the gaussian for the first frame
        first_frame_gaussians=self.keypoints_to_gaussian_maps(keypoints[:,0], HH, WW)
        first_frame_features=self.image_encoder(first_image)
        # Get features and maps for first frame:
        # Note that we cannot use the Gaussian maps above because the
        # first_frame_keypoints may be different than the keypoints (i.e. obs vs
        # pred).
        output=[]
        # Loop over timesteps:
        for i in range(1,SF):
            # Convert keypoints to pixel maps:
            current_frame_gaussians=self.keypoints_to_gaussian_maps(keypoints[:,i], HH, WW)
            # Reconstruct image:
            combined_represetnation=torch.cat((current_frame_gaussians, first_frame_features, first_frame_gaussians), dim=1)
            # add coordinate channels
            combined_represetnation=self.add_coord_channels(combined_represetnation)
            # pass the combined representation through the decoder
            decoded_representation=self.image_decoder(combined_represetnation)
            # Add in the first frame of the sequence such that the model only needs to
            # predict the change from the first frame:
            output.append(decoded_representation)
        # concatenate the output over the first dimension
        output=torch.stack(output, dim=1)
        # Add in the first frame of the sequence such that the model only needs to
        # predict the change from the first frame
        output=output+first_image[:,None,...]
        return output


        
