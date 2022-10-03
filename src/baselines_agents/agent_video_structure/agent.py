from mint.data.dataset_from_MIME import DatasetFromMIME
from mint.data.dataset_from_SIMITATE import DatasetFromSIMITATE
from mint.data.dataset_from_CLEVRER import DatasetFromCLEVRER
from mint.data.dataset_from_magical import DatasetFromMAGICAL
from mint.utils.trajectory_visualization import TrajectoryVisualizer
from mint.utils.clevrer_qual_results import ResultCollector

from baselines_agents.agent_video_structure.vision import Images2Keypoints, Keypoints2Images
from baselines_agents.agent_video_structure.losses import ReconstructionLoss, SeparationLoss, SparsityLoss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm, trange
import os

from torchsummary import summary

from torchvision.transforms.functional import resize

import warnings
warnings.simplefilter("ignore", UserWarning)

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class VideoStructureAgent(nn.Module):
    def __init__(self,config , dataset):
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
        self.images2keypoints=Images2Keypoints(config).to(device)
        self.keypoints2images=Keypoints2Images(config).to(device)
        # initialize the optimizer
        params=list(self.images2keypoints.parameters())+list(self.keypoints2images.parameters())
        self.optimizer=torch.optim.Adam(params,lr=config['learning_rate'],weight_decay=config['weight_decay'])
        # losses
        self.separation_loss=SeparationLoss(config)
        self.reconstruction_loss=ReconstructionLoss(config)
        self.sparsity_loss=SparsityLoss(config)
        # initialize the wandb
        wandb.watch(self.images2keypoints,log_freq=1000)
        wandb.watch(self.keypoints2images,log_freq=1000)
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
        self.clip_value=config['clip_value']
        self.evaluation_counter=0

    def log_trajectory(self, epoch):
        # get the data
        sample=self.dataset.sample_video_from_data()
        # freeze the encoder
        self.images2keypoints.eval()
        with torch.no_grad():
            coords=torch.zeros(1,sample.shape[0],self.num_keypoints,2)
            active_status=torch.ones(1,sample.shape[0],self.num_keypoints,1)
            for i in range(0,sample.shape[0],10):
                data=torch.tensor(sample[i:i+10]).float().permute(0,3,1,2).to(device).unsqueeze(0)
                data=self.format_data(data)
                kp,_=self.images2keypoints(data)
                coords[0,i:i+10]=kp[...,:2]
                active_status[0,i:i+10]=kp[...,2].unsqueeze(-1)
            # make active_status 0 if it is less than 0.7 and 1 otherwise
            active_status[active_status<0.7]=0
            active_status[active_status>=0.7]=1
            # the coordinates between -1 and 1
            # we change them to fit the original image size
            coords[...,0]=(coords[...,0]+1)*self.width/2
            coords[...,1]=(coords[...,1]+1)*self.height/2
            self.visualizer.log_video(sample[...,:3],coords, active_status, label='Epoch {0}'.format(epoch))
        self.images2keypoints.train()
    
    def eval(self):
        self.evaluation_counter+=1
        # load the model
        self.images2keypoints.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
        # get the data
        sample=self.dataset.sample_video_from_data(video_idx=self.evaluation_counter-1)
        # freeze the encoder
        self.images2keypoints.eval()
        with torch.no_grad():
            coords=torch.zeros(1,sample.shape[0],self.num_keypoints,2)
            active_status=torch.zeros(1,sample.shape[0],self.num_keypoints,1)
            for i in range(0,sample.shape[0],10):
                data=torch.tensor(sample[i:i+10]).float().permute(0,3,1,2).to(device).unsqueeze(0)
                data=self.format_data(data)
                kp,_=self.images2keypoints(data)
                coords[0,i:i+10]=kp[...,:2]
                active_status[0,i:i+10]=kp[...,2].unsqueeze(-1)
            # make active_status 0 if it is less than 0.7 and 1 otherwise
            active_status[active_status<0.9]=0
            active_status[active_status>=0.9]=1
            # the coordinates between -1 and 1
            # we change them to fit the original image size
            coords[...,0]=(coords[...,0]+1)*self.width/2
            coords[...,1]=(coords[...,1]+1)*self.height/2
            self.visualizer.log_video(sample[...,:3],coords, active_status, label='Evaluation {0}'.format(self.evaluation_counter))
            
    def collect_qual_results(self):
        self.evaluation_counter+=1
        if self.load_model and os.path.exists('saved_models/{0}.pt'.format(self.model_name)):
            # load the model
            self.images2keypoints.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
        # get the data
        sample=self.dataset.sample_video_from_data(video_idx=self.evaluation_counter-1)
        # freeze the encoder
        self.images2keypoints.eval()
        with torch.no_grad():
            coords=torch.zeros(1,sample.shape[0],self.num_keypoints,2)
            active_status=torch.zeros(1,sample.shape[0],self.num_keypoints,1)
            for i in range(0,sample.shape[0],10):
                data=torch.tensor(sample[i:i+10]).float().permute(0,3,1,2).to(device).unsqueeze(0)
                data=self.format_data(data)
                kp,_=self.images2keypoints(data)
                coords[0,i:i+10]=kp[...,:2]
                active_status[0,i:i+10]=kp[...,2].unsqueeze(-1)
            # make active_status 0 if it is less than 0.7 and 1 otherwise
            active_status[active_status<0.9]=0
            active_status[active_status>=0.9]=1
            # the coordinates between -1 and 1
            # we change them to fit the original image size
            coords[...,0]=(coords[...,0]+1)*self.width/2
            coords[...,1]=(coords[...,1]+1)*self.height/2
            self.result_collector.collect(video_idx=self.evaluation_counter-1, coords=coords, status=active_status)
            self.visualizer.log_video(sample[...,:3],coords, active_status, label='Evaluation {0}'.format(self.evaluation_counter))
            
    def report_results(self):
        return self.result_collector.save_to_file(self.model_name)
    
    def format_data(self,data):
        N,SF,C,H,W=data.shape
        """Formats the uint8 input image to float32 in the range [-0.5, 0.5]."""
        data=data/255.0 - 0.5
        data=data.view(-1,C,H,W)
        # reshape the image to 64x64
        data=resize(data,[self.input_height, self.input_widtdh])
        data=data.view(N,SF,C, self.input_height, self.input_widtdh)
        return data
    
    def train(self):
        if self.load_model and os.path.exists('saved_models/{0}.pt'.format(self.model_name)) and os.path.exists('saved_models/{0}_reconstruction.pt'.format(self.model_name)):
            # load the model
            self.images2keypoints.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
            self.keypoints2images.load_state_dict(torch.load('saved_models/{0}_reconstruction.pt'.format(self.model_name)))
        # train the model
        # for epoch in range(self.epochs):
        for epoch in trange(self.epochs, desc="Training the model"):
            for sample in self.dataloader:
            # for sample in tqdm(self.dataloader,desc='Epoch {0}'.format(epoch), leave=False):
                # permute the data and move them to the device, enable the gradients
                # N x SF x C x H x W
                data=sample.float().permute(0,1,4,2,3).to(device)
                data=self.format_data(data)
                # get the output
                kp,heatmaps=self.images2keypoints(data)
                # get the reconstruction
                reconstruction=self.keypoints2images(kp, data[:,0], heatmaps)
                # compute the loss
                loss = self.separation_loss(kp) + self.reconstruction_loss(reconstruction, data) + self.sparsity_loss(heatmaps)
                # compute the gradients
                self.optimizer.zero_grad()
                loss.backward()
                # clip the gradients
                torch.nn.utils.clip_grad_norm_(self.images2keypoints.parameters(), self.clip_value)
                torch.nn.utils.clip_grad_norm_(self.keypoints2images.parameters(), self.clip_value)
                # update the parameters
                self.optimizer.step()
            if epoch%100==0:
                self.log_trajectory(epoch)
        # save the model
        if self.save:
            torch.save(self.images2keypoints.state_dict(),'saved_models/{0}.pt'.format(self.model_name))
            torch.save(self.keypoints2images.state_dict(),'saved_models/{0}_reconstruction.pt'.format(self.model_name))
