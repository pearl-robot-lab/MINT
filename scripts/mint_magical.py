from ast import Load
from mint.agents.agent import MINT_agent
from tqdm import trange
import wandb

import torch
torch.autograd.set_detect_anomaly(True)

config_init={
    'model_name':'',
    # parameters for dataset
    'tasks':3,
    'training_demos':24,
    'evaluation_demos':0,
    'number_of_stacked_frames':3,
    'width':96, # Clevrer 480
    'height':96, # Clevrer 320
    'channels':3, # RGB
    'num_frames':128, # number of frames sampled from each video
    'input_width':96,
    'input_height':96,
    # parameters for agent
    'batch_size':16,
    'learning_rate':0.001,
    'clip_value':10.0,
    'weight_decay':0.0001,
    'num_keypoints':25,
    'epochs':20,
    'save_model':True,
    'load_model':False, # for evaluation change to True
    'evaluation_epochs':3,
    # parameters for MINT loss
    'activation_score_threshold':10,
    'region_size':3,
    'bandwidth':0.001,
    'beta_for_overlapping_loss':4.0, # don't allow more than 4 keypoints to overlap
    'kappa_for_it_loss':0.9, # the contribution of the conditional entropy in the construction
    'std_for_heatmap':7.0,
    'threshold_for_heatmap':0.2,
    'thresholded_heatmap_scale':3.5,
    'masked_entropy_loss_weight':100.0,
    'conditional_entropy_loss_weight':100.0,
    'status_weight':10.0,
    'information_transport_loss_weight':20.0,
    'overlap_weight': 30.0,
}

def run_experiment(model_name, params):
  print('Running experiment {0}'.format(model_name))
  for key in params:
    config[key]=params[key]
  config['model_name']=model_name
  wandb.init(project="MAGICAL", name=model_name,
              # anonymous="allow",
              config=config, 
              # mode="disabled"
              )
  agent=MINT_agent(config, dataset='MAGICAL')
  agent.train()
  for i in trange(config['evaluation_epochs'], desc='Evaluation'):
    agent.eval()
  wandb.run.finish()

# reset the config to config init
config=config_init.copy()

run_experiment(model_name='MINT_MAGICAL', params={})