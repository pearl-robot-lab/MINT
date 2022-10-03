from mint.data.dataset_from_MIME import DatasetFromMIME
from mint.data.dataset_from_SIMITATE import DatasetFromSIMITATE
from mint.data.dataset_from_CLEVRER import DatasetFromCLEVRER
from mint.data.dataset_from_magical import DatasetFromMAGICAL
from mint.models.kpae import KeypointDetector
from mint.utils.trajectory_visualization import TrajectoryVisualizer
from mint.losses.mint import MINT
from mint.utils.clevrer_qual_results import ResultCollector

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

import wandb
import numpy as np
from tqdm import tqdm, trange
import os

import warnings
warnings.simplefilter("ignore", UserWarning)

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MINT_agent(nn.Module):
    def __init__(self,config, dataset):
        super().__init__()
        # load the dataset
        if dataset=='MIME':
            self.dataset=DatasetFromMIME(config)
        elif dataset=='SIMITATE':
            self.dataset=DatasetFromSIMITATE(config)
        elif dataset=='CLEVRER':
            self.dataset=DatasetFromCLEVRER(config)
            self.result_collector=ResultCollector(config)
        elif dataset=='MAGICAL':
            self.dataset=DatasetFromMAGICAL(config)
        self.width=config['width']
        self.height=config['height']
        # self.dataset.show_sample(100)
        # initialize the dataloaer
        self.dataloader=DataLoader(self.dataset,batch_size=config['batch_size'],shuffle=True)
        # initialize the model
        self.model=KeypointDetector(config).to(device)
        # print model summary
        # print(summary(self.model,(3,self.height,self.width)))
        # initialize the optimizer
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config['learning_rate'], weight_decay=config['weight_decay'])
        # initialize the mint loss
        self.mint_loss=MINT(config)
        # initialize the wandb
        wandb.watch(self.model,log_freq=1000)
        # initialize the trajectory visualizer
        self.visualizer=TrajectoryVisualizer(config)
        self.save=config['save_model']
        if self.save:
            # create a folder to store the model
            os.makedirs('saved_models',exist_ok=True)
        self.epochs=config['epochs']
        self.num_keypoints=config['num_keypoints']
        self.model_name=config['model_name']
        self.load_model=config['load_model']
        self.evaluation_counter=0
        self.clip_value=config['clip_value']

    def log_trajectory(self, epoch):
        # get the data
        sample=self.dataset.sample_video_from_data()
        # freeze the encoder
        self.model.eval()
        with torch.no_grad():
            coords=torch.zeros(1,sample.shape[0],self.num_keypoints,2)
            active_status=torch.zeros(1,sample.shape[0],self.num_keypoints,1)
            for i in range(0,sample.shape[0],10):
                data=torch.tensor(sample[i:i+10]).float().permute(0,3,1,2).to(device)
                kp,status=self.model(data)
                coords[0,i:i+10]=kp
                active_status[0,i:i+10]=status
            # the coordinates between 0 and 1
            # we change them to fit the original image size
            coords[...,0]=coords[...,0]*self.width
            coords[...,1]=coords[...,1]*self.height
            self.visualizer.log_video(sample[...,:3],coords, active_status, label='Epoch {0}'.format(epoch))
        self.model.train()

    def eval(self):
        self.evaluation_counter+=1
        if self.load_model and os.path.exists('saved_models/{0}.pt'.format(self.model_name)):
            # load the model
            self.model.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
        # get the data
        sample=self.dataset.sample_video_from_data(video_idx=self.evaluation_counter-1)
        # freeze the encoder
        self.model.eval()
        with torch.no_grad():
            coords=torch.zeros(1,sample.shape[0],self.num_keypoints,2)
            active_status=torch.zeros(1,sample.shape[0],self.num_keypoints,1)
            for i in range(0,sample.shape[0],10):
                data=torch.tensor(sample[i:i+10]).float().permute(0,3,1,2).to(device)
                kp,status=self.model(data)
                coords[0,i:i+10]=kp
                active_status[0,i:i+10]=status
            # the coordinates between 0 and 1
            # we change them to fit the original image size
            coords[...,0]=coords[...,0]*self.width
            coords[...,1]=coords[...,1]*self.height
            self.visualizer.log_video(sample[...,:3],coords, active_status, label='Evaluation {0}'.format(self.evaluation_counter))
            
    def collect_qual_results(self):
        self.evaluation_counter+=1
        if self.load_model and os.path.exists('saved_models/{0}.pt'.format(self.model_name)):
            # load the model
            self.model.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
        # get the data
        sample=self.dataset.sample_video_from_data(video_idx=self.evaluation_counter-1)
        # freeze the encoder
        self.model.eval()
        with torch.no_grad():
            coords=torch.zeros(1,sample.shape[0],self.num_keypoints,2)
            active_status=torch.zeros(1,sample.shape[0],self.num_keypoints,1)
            for i in range(0,sample.shape[0],10):
                data=torch.tensor(sample[i:i+10]).float().permute(0,3,1,2).to(device)
                kp,status=self.model(data)
                coords[0,i:i+10]=kp
                active_status[0,i:i+10]=status
            # the coordinates between 0 and 1
            # we change them to fit the original image size
            coords[...,0]=coords[...,0]*self.width
            coords[...,1]=coords[...,1]*self.height
            self.result_collector.collect(video_idx=self.evaluation_counter-1, coords=coords, status=active_status)
            self.visualizer.log_video(sample[...,:3],coords, active_status, label='Evaluation {0}'.format(self.evaluation_counter))
    
    def report_results(self):
        return self.result_collector.save_to_file(self.model_name)

    def train(self):
        if self.load_model and os.path.exists('saved_models/{0}.pt'.format(self.model_name)):
            # load the model
            self.model.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
        # train the model
        # for epoch in range(self.epochs):
        for epoch in trange(self.epochs, desc="Training the model"):
            for sample in self.dataloader:
            # for sample in tqdm(self.dataloader,desc='Epoch {0}'.format(epoch), leave=False):
                # permute the data and move them to the device, enable the gradients
                data=sample.float().permute(0,1,4,2,3).to(device)
                # reshape data
                data=data.view(-1,*data.shape[-3:])
                # get the output
                coords, active_status, heatmaps=self.model(data)
                # compute the loss
                loss=self.mint_loss(coords, heatmaps, active_status, data)
                # compute the gradients
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                # update the parameters
                self.optimizer.step()
            # log the trajectory
            if epoch%5==0:
                self.log_trajectory(epoch)
        self.log_trajectory(self.epochs)
        # save the model
        if self.save:
            torch.save(self.model.state_dict(),'saved_models/{0}.pt'.format(self.model_name))