from downstream.imitation_agent import ImitationAgent
from tqdm import trange
import wandb
import math

import pandas as pd
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)

config_init={
    'model_name':'',
    # parameters for dataset
    'eval_type':'TestJitter', # Demo or TestJitter
    'preproc_name': 'LoRes4A',
    'policy_model':'interaction', # MLLP or interaction
    'stochastic_policy':False,
    'ent_weight':0.1,
    'l2_weight':0.0001,
    'width':96, # Clevrer 480
    'height':96, # Clevrer 320
    'channels':3, # RGB
    'input_width':96,
    'input_height':96,
    # parameters for agent
    'n_rollouts':25, # for evaluation
    'batch_size':32,
    'learning_rate':0.001,
    'clip_value':10.0,
    'weight_decay':0.000001,
    'num_keypoints':25,
    'epochs':10,
    'save_model':True,
    'load_model':False, # for evaluation change to True
    'activation_score_threshold':15,
    'std_for_heatmap':9.0,
    # parameters for the interaction network
    'interaction_layers':1,
    'interaction_latent_size':32,
    'interaction_history':1,
    'prediciton_horizon':1,
    'batch_norm':False,
}

seeds=[1,2,3,4,5]

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

envs=['MoveToRegion', 'MoveToCorner', 'MakeLine']
epochs=[1000, 2500, 5000]
eval_type=['Demo','TestJitter']

def run_experiment(model_name, params):
  print('Running experiment {0}'.format(model_name))
  for key in params:
    config[key]=params[key]
  for i,env in enumerate(envs):
    results={}
    for eval in eval_type:
      results[eval]=[]
    config['epochs']=epochs[i]
    for seed in seeds:
      set_seed(seed)
      model_name_seed=model_name+"_{0}_seed={1}".format(env,seed)
      config['model_name']=model_name_seed
      wandb.init(project="Seeds_imitation", name=model_name_seed,
              # anonymous="allow",
              config=config, 
              # mode="disabled"
              )
      agent=ImitationAgent(config, env_name=env)
      agent.train()
      for eval in eval_type:
        agent.set_env(eval)
        agent.evaluate_policy()
        result=agent.report_results(eval)
        results[eval].append(result)
      wandb.run.finish()
    wandb.init(project="Seeds_imitation", name=model_name+"_{0}".format(env),
                  # anonymous="allow",
                  config=config, 
                  # mode="disabled"
                  )
    for eval in eval_type:
      concat_results=pd.concat(results[eval])
      mean=concat_results.mean()
      std=concat_results.std()
      confidence_interval=2*std / np.sqrt(len(seeds))
      data={}
      for key in mean.keys():
          data[key]=[mean[key],confidence_interval[key]]
      final_results=pd.DataFrame.from_dict(data, orient='index', columns=['mean','confidence_interval'])
      table=wandb.Table(dataframe=final_results)
      wandb.log({"Statistics over seeds {0}".format(eval) : table})
    wandb.run.finish()

# reset the config to config init
config=config_init.copy()

run_experiment(model_name='Imitation_MINT', params={})