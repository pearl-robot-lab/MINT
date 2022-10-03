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

class MAGICALNet(nn.Module):
    """Custom CNN for MAGICAL policies."""
    def __init__(self, observation_shape, action_dim, width=2):
        super().__init__()
        w = width
        feature_dim=256
        self.latent_size=32
        def conv_block(i, o, k, s, p, b=False):
            return [
                # batch norm has its own bias, so don't add one to conv layers by default
                nn.Conv2d(i, o, kernel_size=k, stride=s, padding=p, bias=b,
                          padding_mode='zeros'),
                nn.ReLU(),
                nn.BatchNorm2d(o)
            ]
        conv_layers = [
            *conv_block(i=observation_shape[0], o=32*w, k=5, s=1, p=2, b=True),
            *conv_block(i=32*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
        ]
        # final FC layer to make feature maps the right size
        test_tensor = torch.zeros((1,) + observation_shape)
        for layer in conv_layers:
            test_tensor = layer(test_tensor)
        fc_in_size = np.prod(test_tensor.shape)
        reduction_layers = [
            nn.Flatten(),
            nn.Linear(fc_in_size, feature_dim),
            # Stable Baselines will add extra affine layer on top of this reLU
            nn.ReLU(),
        ]
        self.features_dim = feature_dim
        all_layers = [*conv_layers, *reduction_layers]
        self.feature_generator = nn.Sequential(*all_layers)
        self.policy=nn.Sequential(
            # magical gives 4 images in the demonstration, we follow that
            nn.Linear(feature_dim, 4*self.latent_size),
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

    def forward(self, x):
        x= self.feature_generator(x)
        x= self.policy(x)
        return x

class ImitationAgentCNN(nn.Module):
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
        demo_trajs_preproc = magical.preprocess_demos_with_wrapper(demo_trajs, orig_env_name, preproc_name=config['preproc_name'])

        # Build dataset in the format required by imitation. Note that traj.obs contains the final observation after the last
        # action, so we drop the last observation when concatenating trajectories.
        all_obs = np.concatenate([traj.obs[:-1] for traj in demo_trajs_preproc], axis=0)
        all_acts = np.concatenate([traj.acts for traj in demo_trajs_preproc], axis=0)
        dataset = il_types.TransitionsMinimal(obs=all_obs, acts=all_acts, infos=[{}] * len(all_obs))
        self.dataloader = DataLoader(dataset, batch_size=config['batch_size'])
        self.ce_loss=nn.CrossEntropyLoss()
        
        # num actions
        action_dim=self.env.action_space.n
        observation_shape=self.env.observation_space.shape
        observation_shape=(observation_shape[2],observation_shape[0], observation_shape[1])
        self.policy=MAGICALNet(observation_shape, action_dim).to(device)
        
        self.stochastic=config['stochastic_policy']
        self.ent_weight=config['ent_weight']
        self.l2_weight=config['l2_weight']

        # initialize the optimizer
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=config['learning_rate'], weight_decay=config['weight_decay'])
        # initialize the wandb
        wandb.watch(self.policy,log_freq=1000)
        self.save=config['save_model']
        if self.save:
            # create a folder to store the model
            os.makedirs('saved_models',exist_ok=True)
        self.num_keypoints=config['num_keypoints']
        self.epochs=config['epochs']
        self.model_name=config['model_name']
        self.load_model=config['load_model']
        self.evaluation_counter=0
        self.clip_value=config['clip_value']
        self.visualizer=TrajectoryVisualizer(config)
        self.n_rollouts=config['n_rollouts']
        
    def set_env(self, eval):
        self.gym_env_name='{0}-{1}-{2}-v0'.format(self.env_name, eval, self.preproc_name)
        self.env=gym.make(self.gym_env_name)

    def evaluate_policy(self, log_video=True):
        if os.path.exists('saved_models/{0}.pt'.format(self.model_name)):
            # load the model
            self.policy.load_state_dict(torch.load('saved_models/{0}.pt'.format(self.model_name)))
        self.policy.eval()
        self.scores=[]
        for i in trange(self.n_rollouts, desc="Evaluating the policy"):
            obs=self.env.reset()
            done=False
            images=[]
            while not done:
                data=torch.tensor(obs).permute(2,0,1).float().to(device).unsqueeze(0)
                data=self.format_data(data)
                with torch.no_grad():
                    if self.stochastic:
                        logits = self.policy(data)
                        acts=Categorical(logits=logits).sample()
                    else:
                        logits = self.policy(data)
                        acts=torch.argmax(logits, dim=-1).detach().cpu()
                obs, rew, done, info = self.env.step(acts)
                if log_video:
                    images.append(obs[...,-3:])
            self.scores.append(self.env.score_on_end_of_traj())
            if log_video:
                images=np.array(images)
                coords=np.zeros((images.shape[0], self.num_keypoints,2))
                status=np.zeros((images.shape[0], self.num_keypoints,1))
                coords=torch.tensor(coords).float().to(device).unsqueeze(0)
                status=torch.tensor(status).float().to(device).unsqueeze(0)
                self.visualizer.log_video(images,coords, status, label='BC_Evaluation_{0}_{1}'.format(self.env_name,i))
    
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
        """Formats the input image to the range [-0.5, 0.5]."""
        data=data/255.0 - 0.5
        return data

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
                data=sample['obs'].permute(0,3,1,2).float().to(device)
                data=self.format_data(data)
                # predict the actions
                # N x SF x KP x 3
                action_logits=self.policy(data)
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
                    loss = neglogp + ent_term + l2_term
                    wandb.log({'{0}/neglogp'.format(self.env_name): neglogp.item(),
                               '{0}/ent_term'.format(self.env_name): ent_term.item(),
                               '{0}/l2_term'.format(self.env_name): l2_term.item(),
                               })
                else:
                    # action_logits.register_hook(lambda grad: print("grad : ",grad.mean()))
                    # compute the loss
                    loss=self.ce_loss(action_logits, ground_truth_actions)
                wandb.log({'loss_{0}'.format(self.env_name): loss.item()})
                # compute the gradients
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_value)
                # update the parameters
                self.optimizer.step()
        # save the model
        if self.save:
            torch.save(self.policy.state_dict(),'saved_models/{0}.pt'.format(self.model_name))