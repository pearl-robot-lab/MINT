from downstream.kpInteract import KeypointInteractionNetwork
from mint.models.kpae import KeypointDetector
from mint.utils.trajectory_visualization import TrajectoryVisualizer
from downstream.mlp_model import MLPModel

import torch
import torch.nn as nn
from torchvision.transforms.functional import resize
from torch.distributions import Categorical

import wandb
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

import gym
import magical
import imitation.data.types as il_types
import glob
import pandas as pd

import warnings
warnings.simplefilter("ignore", UserWarning)

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')        

demo_paths_by_env = {
    'MoveToRegion': glob.glob('demos/move-to-region/demo-*.pkl.gz'),
    'MoveToCorner': glob.glob('demos/move-to-corner/demo-*.pkl.gz'),
    'MakeLine': glob.glob('demos/make-line/demo-*.pkl.gz'),
    'ClusterColour': glob.glob('demos/cluster-colour/demo-*.pkl.gz'),
}

magical.register_envs() 

class ImitationAgent(nn.Module):
    def __init__(self,config, env_name=None):
        super().__init__()
        # get the demos
        self.preproc_name=config['preproc_name']
        self.gym_env_name='{0}-{1}-{2}-v0'.format(env_name, config['eval_type'], config['preproc_name'])
        self.env=gym.make(self.gym_env_name)
        self.env_name=env_name
        # dataset=MagicalData(env_name)
        demo_path=demo_paths_by_env[env_name]
        demo_dicts = magical.load_demos(demo_path[:10])
        demo_trajs = []
        orig_env_name = None  # we will read this from the demos dicts
        for demo_dict in demo_dicts:
            # each demo dict has keys ['trajectory', 'score', 'env_name']
            # (trajectory contains the actual data, and score is generally 1.0 for demonstrations)
            orig_env_name = demo_dict['env_name']
            demo_trajs.append(demo_dict['trajectory'])
            # break
        demo_trajs_preproc = magical.preprocess_demos_with_wrapper(demo_trajs, orig_env_name, preproc_name=config['preproc_name'])

        # Build dataset in the format required by imitation. Note that traj.obs contains the final observation after the last
        # action, so we drop the last observation when concatenating trajectories.
        all_obs = np.concatenate([traj.obs[:-1] for traj in demo_trajs_preproc], axis=0)
        all_acts = np.concatenate([traj.acts for traj in demo_trajs_preproc], axis=0)
        dataset = il_types.TransitionsMinimal(obs=all_obs, acts=all_acts, infos=[{}] * len(all_obs))
        self.dataloader = DataLoader(dataset, batch_size=config['batch_size'])
        self.ce_loss=nn.CrossEntropyLoss()
        self.huber_loss=nn.HuberLoss()
        
        # initialize the keypoint detector
        # obs_sample=self.env.observation_space.sample()
        self.width=config['width']
        self.height=config['height']
        self.input_width=config['input_width']
        self.input_height=config['input_height']
        # num actions
        action_dim=self.env.action_space.n
        # self.dataset.show_sample(100)
        # initialize the keypoints detector
        self.kp_detector=KeypointDetector(config).to(device)
        if os.path.exists('imitation/models/MINT_MAGICAL.pt'):
            # load the model
            self.kp_detector.load_state_dict(torch.load('imitation/models/MINT_MAGICAL.pt'))
        self.kp_detector.eval()
        # initialize the dynamics model
        if config['policy_model']=='interaction':
            self.policy=KeypointInteractionNetwork(config, 'imitation', action_dim).to(device)
        else:
            self.policy=MLPModel(config, 'imitation', action_dim).to(device)
        self.stochastic=config['stochastic_policy']
        self.ent_weight=config['ent_weight']
        self.l2_weight=config['l2_weight']
        # if self.stochastic:
        #     self.policy_dist=Categorical()
        # initialize the optimizer
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=config['learning_rate'], weight_decay=config['weight_decay'])
        # initialize the wandb
        wandb.watch(self.policy,log_freq=1000)
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
        self.visualizer=TrajectoryVisualizer(config)
        self.n_rollouts=config['n_rollouts']
        # log the demo
        images=[]
        # for obs in all_obs:
        #     images.append(obs[...,-3:])
        # images=np.array(images)
        # coords=np.zeros((images.shape[0], self.num_keypoints,2))
        # status=np.zeros((images.shape[0], self.num_keypoints,1))
        # coords=torch.tensor(coords).float().to(device).unsqueeze(0)
        # status=torch.tensor(status).float().to(device).unsqueeze(0)
        # self.visualizer.log_video(images,coords, status, label='Demos{0}'.format(self.env_name))
        # # check the dataloader
        # images=[]
        # for sample in self.dataloader:
        #     obs=sample['obs']
        #     for i in range(obs.shape[0]):
        #         images.append(obs[i,:,:,-3:].numpy())
        #     break
        # images=np.array(images)
        # coords=np.zeros((images.shape[0], self.num_keypoints,2))
        # status=np.zeros((images.shape[0], self.num_keypoints,1))
        # coords=torch.tensor(coords).float().to(device).unsqueeze(0)
        # status=torch.tensor(status).float().to(device).unsqueeze(0)
        # self.visualizer.log_video(images,coords, status, label='Demos{0}'.format(self.env_name))
    
    def set_env(self, eval):
        self.gym_env_name='{0}-{1}-{2}-v0'.format(self.env_name, eval, self.preproc_name)
        self.env=gym.make(self.gym_env_name)

    def evaluate_policy(self, log_video=True):
        if os.path.exists('saved_models/{0}.pt'.format(self.model_name)):
            # load the model
            self.policy.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
        self.kp_detector.eval()
        self.policy.eval()
        self.scores=[]
        for i in trange(self.n_rollouts, desc="Evaluating the policy"):
            obs=self.env.reset()
            done=False
            images=[]
            all_coords=[]
            all_status=[]
            while not done:
                data=torch.tensor(obs)
                data=data.permute(2,0,1)
                data=data.view(1,4,3,*data.shape[-2:]).contiguous()
                data=data.float().to(device)
                with torch.no_grad():
                    kp=self.detect_keypoints(data)
                    if self.stochastic:
                        _,logits = self.policy(kp)
                        acts=Categorical(logits=logits).sample()
                    else:
                        _,logits = self.policy(kp)
                        acts=torch.argmax(logits, dim=-1).detach().cpu()
                obs, rew, done, info = self.env.step(acts)
                if log_video:
                    coords=kp[0,-1,:,:2]
                    status=kp[0,-1,:,2].unsqueeze(-1)
                    all_coords.append(coords.detach().cpu().numpy())
                    all_status.append(status.detach().cpu().numpy())
                    images.append(obs[...,-3:])
            self.scores.append(self.env.score_on_end_of_traj())
            if log_video:
                images=np.array(images)
                coords=np.array(all_coords)
                status=np.array(all_status)
                coords[...,0]=(coords[...,0]+1)*self.width/2
                coords[...,1]=(coords[...,1]+1)*self.height/2
                coords=torch.tensor(coords).float().to(device).unsqueeze(0)
                status=torch.tensor(status).float().to(device).unsqueeze(0)
                self.visualizer.log_video(images,coords, status, label='BC_Evaluation_{0}_{1}'.format(self.gym_env_name,i))
    
    def report_results(self, eval):
        # create a folder to store the dataset
        os.makedirs('results',exist_ok=True)
        file_path="results/{0}_{1}.xlsx".format(eval,self.model_name)
        writer=pd.ExcelWriter(file_path, engine='openpyxl')
        scores_frame=pd.DataFrame(self.scores, columns=['score'])
        table=wandb.Table(dataframe=scores_frame)
        wandb.log({"{0}_Scores".format(self.gym_env_name) : table})
        scores_frame.to_excel(writer, sheet_name='Rollouts', index=False)
        result_stats=scores_frame.describe()
        table=wandb.Table(dataframe=result_stats)
        wandb.log({"Statistics over all evaluations {0}".format(self.gym_env_name) : table})
        result_stats.to_excel(writer, sheet_name='Statistics')
        writer.save()
        writer.close()
        return scores_frame
    
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
        # reshape data
        data=data.view(-1,*data.shape[-3:])
        coords, status = self.kp_detector(data)
        # change the range of the coordinates from [0,1] to [-1,1]
        coords=(coords-0.5)*2.0
        coords=coords.view(N,SF,-1,2)
        status=status.view(N,SF,-1,1)
        coords*=status
        kp=torch.cat((coords, status), dim=-1)
        return kp

    def train(self):
        if self.load_model and os.path.exists('saved_models/{0}.pt'.format(self.model_name)):
            # load the model
            self.policy.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
        # train the model
        # for epoch in range(self.epochs):
        for epoch in trange(self.epochs, desc="Training the model"):
            for sample in self.dataloader:
            # for sample in tqdm(self.dataloader,desc='Epoch {0}'.format(epoch), leave=False):
                # permute the data and move them to the device, enable the gradients
                ground_truth_actions=sample['acts'].to(device).long()
                obs=sample['obs']
                obs=obs.permute(0,3,1,2)
                obs=obs.view(-1,4,3,*obs.shape[2:]).contiguous()
                data=obs.float().to(device)
                # get the keypoints
                with torch.no_grad():
                    kp=self.detect_keypoints(data)
                # predict the actions
                # N x SF x KP x 3
                predictions, action_logits=self.policy(kp)
                # consider only 1-step prediction
                predictions = predictions[:,:,:,0]
                # shift the prediction one time step to compare to the ground truth
                predicted_coords=torch.roll(predictions[...,:2],-1, dims=0)
                ground_truth_coords=kp[...,:2]
                predicted_coords[0]=ground_truth_coords[0]
                prediction_loss=self.huber_loss(predicted_coords,ground_truth_coords)
                if self.stochastic:
                    dist=Categorical(logits=action_logits)
                    log_prob = dist.log_prob(ground_truth_actions).mean()
                    ent_loss = dist.entropy().mean()
                    l2_norms = [
                        torch.sum(torch.square(w)) for w in self.policy.parameters()
                    ]
                    l2_loss_raw = sum(l2_norms) / 2
                    ent_term = -self.ent_weight * ent_loss
                    neglogp = -log_prob
                    l2_term = self.l2_weight * l2_loss_raw
                    imitation_loss = neglogp + ent_term + l2_term
                    wandb.log({'{0}/neglogp'.format(self.env_name): neglogp.item(),
                               '{0}/ent_term'.format(self.env_name): ent_term.item(),
                               '{0}/l2_term'.format(self.env_name): l2_term.item(),
                               })
                else:
                    # action_logits.register_hook(lambda grad: print("grad : ",grad.mean()))
                    # compute the loss
                    imitation_loss=self.ce_loss(action_logits, ground_truth_actions)
                loss=prediction_loss+10*imitation_loss
                wandb.log({'loss_{0}'.format(self.env_name): loss.item(),
                            'imitation_loss_{0}'.format(self.env_name): imitation_loss.item(),
                            'prediction_loss_{0}'.format(self.env_name): prediction_loss.item()})
                # compute the gradients
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_value)
                # update the parameters
                self.optimizer.step()
        # save the model
        if self.save:
            torch.save(self.policy.state_dict(),'saved_models/{0}.pt'.format(self.model_name))