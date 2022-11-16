from tokenize import Hexnumber
from downstream.kpInteract import KeypointInteractionNetwork
from downstream.dynamics_evaluator import DynamicsEvaluator
from mint.models.kpae import KeypointDetector
from baselines_agents.agent_transporter.models_kp import KeyPointNet
from baselines_agents.agent_video_structure.vision import Images2Keypoints
from mint.data.dataset_from_CLEVRER import DatasetFromCLEVRER
from mint.utils.trajectory_visualization import TrajectoryVisualizer

import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

import wandb
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import os

import warnings
warnings.simplefilter("ignore", UserWarning)

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DynamicsAgent(nn.Module):
    def __init__(self,config, args=None, method='MINT'):
        super().__init__()
        self.width=config['width']
        self.height=config['height']
        self.input_width=config['input_width']
        self.input_height=config['input_height']
        # load the dataset
        self.dataset=DatasetFromCLEVRER(config)
        # initialize the dataloaer
        self.dataloader=DataLoader(self.dataset,batch_size=config['batch_size'],shuffle=True)
        # self.dataset.show_sample(100)
        # initialize the keypoints detector
        if method == "MINT" or method =="MINTwoR":
            self.kp_detector=KeypointDetector(config).to(device)
        elif method=='Transporter':
            # initialize the model
            use_gpu = torch.cuda.is_available()
            self.kp_detector=KeyPointNet(args, use_gpu=use_gpu).to(device)
        elif method=='Video_structure':
            self.kp_detector=Images2Keypoints(config).to(device)
        if os.path.exists('dynamics/models/{0}_CLEVRER.pt'.format(method)):
            # load the model
            self.kp_detector.load_state_dict(torch.load('dynamics/models/{0}_CLEVRER.pt'.format(method)))
        self.method=method
        self.kp_detector.eval()
        # initialize the dynamics model
        self.dynamics_model=KeypointInteractionNetwork(config, task='learn_dynamics').to(device)
        # initialize the optimizer
        self.optimizer=torch.optim.Adam(self.dynamics_model.parameters(),lr=config['learning_rate'], weight_decay=config['weight_decay'])
        # initialize the evaluator
        self.evaluator=DynamicsEvaluator(config)
        # the loss
        self.huber_loss=nn.HuberLoss()
        self.mse_loss=nn.MSELoss()
        # initialize the trajectory visualizer
        self.visualizer=TrajectoryVisualizer(config, task='prediction')
        # initialize the wandb
        wandb.watch(self.dynamics_model,log_freq=1000)
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
        self.skip=config['interaction_history']
        self.horizon=config['prediciton_horizon']
        
    def collect_qual_results(self):
        self.evaluation_counter+=1
        if self.load_model and os.path.exists('saved_models/{0}.pt'.format(self.model_name)):
            # load the model
            self.dynamics_model.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
        # get the data
        sample=self.dataset.sample_video_from_data(video_idx=self.evaluation_counter-1)
        # freeze the encoder
        self.dynamics_model.eval()
        with torch.no_grad():
            kps=torch.zeros(1,sample.shape[0],self.num_keypoints,3).to(device)
            coords=torch.zeros(1,sample.shape[0],self.num_keypoints,2)
            status=torch.zeros(1,sample.shape[0],self.num_keypoints,1)
            data=torch.tensor(sample).float().permute(0,3,1,2).to(device).unsqueeze(0)
            # get the keypoints
            kp=self.detect_keypoints(data)
            kps=kp
            coords=kp[...,:2]
            status=kp[...,-1].unsqueeze(-1)
            # pass the keypoints to the dynamics model
            # N x SF-1 x KP x 3
            predictions=self.dynamics_model(kps)
            predicted_coords=predictions[...,:2]
            predicted_status=predictions[...,-1].unsqueeze(-1)
            # the coordinates between -1 and 1
            # we change them to fit the original image size
            coords[...,0]=(coords[...,0]+1)*self.width/2
            coords[...,1]=(coords[...,1]+1)*self.height/2
            predicted_coords[...,0]=(predicted_coords[...,0]+1)*self.width/2
            predicted_coords[...,1]=(predicted_coords[...,1]+1)*self.height/2
            self.evaluator.collect(self.evaluation_counter-1, coords, status, predicted_coords, predicted_status)
            self.visualizer.log_video_prediction(sample[...,:3],coords, status, predicted_coords=predicted_coords[:,:,:,0], label='Evaluation {0}'.format(self.evaluation_counter))
    
    def report_results(self):
        return self.evaluator.save_to_file(self.model_name)
    
    def format_data(self,data):
        N,SF,C,H,W=data.shape
        """Formats the uint8 input image to float32 in the range [-0.5, 0.5]."""
        data=data/255.0 - 0.5
        data=data.view(-1,C,H,W)
        # reshape the image to 64x64
        data=resize(data,[self.input_height, self.input_width])
        return data

    def detect_keypoints(self, data):
        N,SF,C,H,W=data.shape
        if self.method == "MINT" or self.method =="MINTwoR":
            # reshape data
            data=data.view(-1,*data.shape[-3:])
            coords, status = self.kp_detector(data)
            # change the range of the coordinates from [0,1] to [-1,1]
            coords=(coords-0.5)*2.0
            coords=coords.view(N,SF,-1,2)
            status=status.view(N,SF,-1,1)
        elif self.method=='Transporter':
            data=self.format_data(data)
            coords=self.kp_detector.predict_keypoint(data)
            coords=coords.view(N,SF,-1,2)
            status=torch.ones(*coords.shape[:-1],1).to(device)
        elif self.method=='Video_structure':
            data=self.format_data(data)
            kp, _=self.kp_detector(data)
            kp=kp.view(N,SF,-1,3)
            coords=kp[...,:2]
            status=kp[...,-1].unsqueeze(-1)
            status[status<0.9]=0
            status[status>=0.9]=1
        kp=torch.cat((coords, status), dim=-1)
        return kp

    def train(self):
        if self.load_model and os.path.exists('saved_models/{0}.pt'.format(self.model_name)):
            # load the model
            self.dynamics_model.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
        # train the model
        # for epoch in range(self.epochs):
        for epoch in trange(self.epochs, desc="Training the model"):
            for sample in self.dataloader:
            # for sample in tqdm(self.dataloader,desc='Epoch {0}'.format(epoch), leave=False):
                # permute the data and move them to the device, enable the gradients
                data=sample.float().permute(0,1,4,2,3).to(device)
                # get the keypoints
                with torch.no_grad():
                    kp=self.detect_keypoints(data)
                # pass the keypoints to the dynamics model
                # N x SF-1 x KP x 3
                predictions=self.dynamics_model(kp)
                loss = 0
                for h in range(self.horizon):
                    # shift the prediction one time step to compare to the ground truth
                    predicted_coords=torch.roll(predictions[...,h,:2],-(h+1), dims=1)
                    # skip the first time frames
                    predicted_coords[:,:self.skip]=0
                    predicted_coords[:,-(h+1):]=0
                    # put it here for correct gradeint computation (pytorch modified inplace error)
                    ground_truth_coords=kp[...,:2].clone()
                    ground_truth_coords[:,:self.skip]=0
                    ground_truth_coords[:,-(h+1):]=0
                    # compute the loss
                    loss+=self.huber_loss(predicted_coords,ground_truth_coords)
                wandb.log({'loss': loss.item()})
                # compute the gradients
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), self.clip_value)
                # update the parameters
                self.optimizer.step()
        # save the model
        if self.save:
            torch.save(self.dynamics_model.state_dict(),'saved_models/{0}.pt'.format(self.model_name))