from downstream.dynamics_agent import DynamicsAgent
from tqdm import trange
import wandb

import pandas as pd
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)

config_init={
    'model_name':'',
    # parameters for dataset
    'training_demos':20,
    'evaluation_demos':80,
    'number_of_stacked_frames':8,
    'width':480, # Clevrer 480
    'height':320, # Clevrer 320
    'channels':3, # RGB
    'num_frames':128, # number of frames sampled from each video
    'input_width':480,
    'input_height':320,
    # parameters for agent
    'batch_size':32,
    'learning_rate':0.001,
    'clip_value':10.0,
    'weight_decay':0.000001,
    'num_keypoints':25,
    'epochs':50,
    'save_model':True,
    'load_model':False, # for evaluation change to True
    'evaluation_epochs':100,
    'activation_score_threshold':15,
    'std_for_heatmap':9.0,
    # parameters for the interaction network
    'interaction_layers':1,
    'interaction_latent_size':32,
    'interaction_history':4,
    'prediciton_horizon':3,
    'batch_norm':False,
}

seeds=[1,2,3,4,5]

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
        
def run_experiment(model_name, params=None):
  print('Running experiment {0}'.format(model_name))
  if params is not None:
    for key in params:
      config[key]=params[key]
  results=[]
  for seed in seeds:
    set_seed(seed)
    model_name_seed=model_name+"_seed={0}".format(seed)
    config['model_name']=model_name_seed
    wandb.init(project="Dynamics", name=model_name_seed,
              # anonymous="allow",
              config=config, 
              # mode="disabled"
              )
    agent=DynamicsAgent(config, method='MINT')
    agent.train()
    for i in trange(config['evaluation_epochs'], desc='Evaluation'):
      agent.collect_qual_results()
    result=agent.report_results()
    results.append(result)
    wandb.run.finish()
  results=pd.concat(results)
  mean=results.mean()
  std=results.std()
  confidence_interval=2*std / np.sqrt(len(seeds))
  data={}
  for key in mean.keys():
      data[key]=[mean[key],confidence_interval[key]]
  final_results=pd.DataFrame.from_dict(data, orient='index', columns=['mean','confidence_interval'])
  wandb.init(project="Dynamics", name=model_name,
              # anonymous="allow",
              config=config, 
              # mode="disabled"
              )
  table=wandb.Table(dataframe=final_results)
  wandb.log({"Statistics over seeds" : table})
  wandb.run.finish()

# reset the config to config init
config=config_init.copy()

run_experiment(model_name='LearnDynamics_MINT_CLEVRER', params={})