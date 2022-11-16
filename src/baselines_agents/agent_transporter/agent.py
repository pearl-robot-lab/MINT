from mint.data.dataset_from_MIME import DatasetFromMIME
from mint.data.dataset_from_SIMITATE import DatasetFromSIMITATE
from mint.data.dataset_from_CLEVRER import DatasetFromCLEVRER
from mint.data.dataset_from_magical import DatasetFromMAGICAL
from mint.utils.trajectory_visualization import TrajectoryVisualizer
from mint.utils.clevrer_qual_results import ResultCollector
from baselines_agents.agent_transporter.models_kp import KeyPointNet
# for transporter modified uncomment this line
# from baselines_agents.agent_transporter.models_kp_modified import KeyPointNet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
import numpy as np
from tqdm import tqdm, trange
import os

from torchvision.transforms.functional import resize

import warnings
warnings.simplefilter("ignore", UserWarning)

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class TransporterAgent(nn.Module):
    def __init__(self,config, args , dataset):
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
        self.input_widtdh=config['input_width']
        self.input_height=config['input_height']
        # self.dataset.show_sample(100)
        # initialize the dataloaer
        self.dataloader=DataLoader(self.dataset,batch_size=config['batch_size'],shuffle=True)
        # initialize the model
        use_gpu = torch.cuda.is_available()
        self.model=KeyPointNet(args, use_gpu=use_gpu).to(device)
        # initialize the optimizer
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=args.lr, betas=(args.beta1, 0.999))
        # criterion
        self.criterionMSE = nn.MSELoss().to(device)
        # initialize the wandb
        wandb.watch(self.model,log_freq=1000)
        # initialize the trajectory visualizer
        self.visualizer=TrajectoryVisualizer(config)
        self.save=config['save_model']
        if self.save:
            # create a folder to store the model
            os.makedirs('saved_models',exist_ok=True)
        self.epochs=args.n_epoch
        self.num_keypoints=config['num_keypoints']
        self.model_name=config['model_name']
        self.load_model=config['load_model']
        self.evaluation_counter=0

    def log_trajectory(self, epoch):
        # get the data
        sample=self.dataset.sample_video_from_data()
        # freeze the encoder
        self.model.eval()
        with torch.no_grad():
            coords=torch.zeros(1,sample.shape[0],self.num_keypoints,2)
            active_status=torch.ones(1,sample.shape[0],self.num_keypoints,1)
            for i in range(0,sample.shape[0],10):
                data=torch.tensor(sample[i:i+10]).float().permute(0,3,1,2).to(device)
                data=self.format_data(data)
                kp=self.model.predict_keypoint(data)
                coords[0,i:i+10]=kp
            # the coordinates between -1 and 1
            # we change them to fit the original image size
            coords[...,0]=(coords[...,0]+1)*self.width/2
            coords[...,1]=(coords[...,1]+1)*self.height/2
            self.visualizer.log_video(sample[...,:3],coords, active_status, label='Epoch {0}'.format(epoch))
        self.model.train()
    
    def eval(self):
        self.evaluation_counter+=1
        # load the model
        self.model.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
        # get the data
        sample=self.dataset.sample_video_from_data(video_idx=self.evaluation_counter-1)
        # freeze the encoder
        self.model.eval()
        with torch.no_grad():
            coords=torch.zeros(1,sample.shape[0],self.num_keypoints,2)
            active_status=torch.ones(1,sample.shape[0],self.num_keypoints,1)
            for i in range(0,sample.shape[0],10):
                data=torch.tensor(sample[i:i+10]).float().permute(0,3,1,2).to(device)
                data=self.format_data(data)
                kp=self.model.predict_keypoint(data)
                coords[0,i:i+10]=kp
            # the coordinates between -1 and 1
            # we change them to fit the original image size
            coords[...,0]=(coords[...,0]+1)*self.width/2
            coords[...,1]=(coords[...,1]+1)*self.height/2
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
            active_status=torch.ones(1,sample.shape[0],self.num_keypoints,1)
            for i in range(0,sample.shape[0],10):
                data=torch.tensor(sample[i:i+10]).float().permute(0,3,1,2).to(device)
                data=self.format_data(data)
                kp=self.model.predict_keypoint(data)
                coords[0,i:i+10]=kp
            # the coordinates between -1 and 1
            # we change them to fit the original image size
            coords[...,0]=(coords[...,0]+1)*self.width/2
            coords[...,1]=(coords[...,1]+1)*self.height/2
            self.result_collector.collect(video_idx=self.evaluation_counter-1, coords=coords, status=active_status)
            self.visualizer.log_video(sample[...,:3],coords, active_status, label='Evaluation {0}'.format(self.evaluation_counter))
    
    def report_results(self):
        return self.result_collector.save_to_file(self.model_name)
            
    def format_data(self,data):
        N,C,H,W=data.shape
        """Formats the uint8 input image to float32 in the range [-0.5, 0.5]."""
        data=data/255.0 - 0.5
        data=data.view(-1,C,H,W)
        # reshape the image to 64x64
        data=resize(data,[self.input_height, self.input_widtdh])
        data=data.view(N,C, self.input_height, self.input_widtdh)
        return data

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
                # N x SF x C x H x W
                src=sample.float().permute(0,1,4,2,3).to(device)
                # destination are the src shifted by a random number between 1 and number of stacked frames
                des=torch.roll(src,1,dims=1)
                # reshape data to fit the model
                # N*SF x C x H x W
                src=src.reshape(src.shape[0]*src.shape[1],src.shape[2],src.shape[3],src.shape[4])
                src=self.format_data(src)
                des=des.reshape(des.shape[0]*des.shape[1],des.shape[2],des.shape[3],des.shape[4])
                des=self.format_data(des)
                # get the output
                des_reconstruction, arc_kp, des_kp=self.model(src,des)
                # compute the loss
                loss=self.criterionMSE(des_reconstruction,des) * 10
                # log the loss to wandb
                wandb.log({'loss':loss.item()})
                # compute the gradients
                self.optimizer.zero_grad()
                loss.backward()
                # update the parameters
                self.optimizer.step()
            if epoch%100==0:
                self.log_trajectory(epoch)
        # save the model
        if self.save:
            torch.save(self.model.state_dict(),'saved_models/{0}.pt'.format(self.model_name))
