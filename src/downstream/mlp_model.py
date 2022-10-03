import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300

import wandb

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MLPModel(nn.Module):
    def __init__(self, config, task, action_dim=None):
        super().__init__()
        self.interaction_layers=config['interaction_layers']
        self.latent_size=config['interaction_latent_size']
        self.history=config['interaction_history']
        KP = config['num_keypoints']
        keypoint_dim=4 # pos_encoding, x, y, activation_status
        # positional encoder a number between -1 and 1 for each keypoint
        self.positional_encoding=torch.linspace(-1,1,KP).to(device)[None,None,:,None]
        # project the output
        self.action_model=nn.Sequential(
            # magical gives 4 images in the demonstration, we follow that
            nn.Linear(4*keypoint_dim*KP, 4*self.latent_size),
            nn.BatchNorm1d(4*self.latent_size),
            # nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(4*self.latent_size, 2*self.latent_size),
            nn.BatchNorm1d(2*self.latent_size),
            # nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(2*self.latent_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            # nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            # nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(self.latent_size, action_dim),
        )

    def forward(self, keypoints):
        N,SF,KP,F=keypoints.shape
        # flatten the keypoints over batches
        positional_encoding=self.positional_encoding.repeat(N,SF,1,1)
        keypoints=torch.cat((positional_encoding, keypoints), dim=-1)
        x=keypoints.view(N,-1)
        x=self.action_model(x)
        return x