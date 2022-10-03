from mint.agents.agent import MINT_agent
from tqdm import trange
import wandb

import torch
torch.autograd.set_detect_anomaly(True)

config_init={
    'model_name':'',
    # parameters for dataset
    'tasks':'1,2,3,4',#,5,6,7,8,9,10,11,12,13,15,16,17,19,20', # 14 and 18 are not available 
    'training_demos':20,
    'evaluation_demos':5,
    'number_of_stacked_frames':2,
    'width':320, # MIME 640 -> after crop to 320x240
    'height':240, # MIME 240
    'channels':3, # RGB
    'num_frames':50, # number of frames sampled from each video
    # parameters for agent
    'num_layers':2,
    'batch_size':32,
    'learning_rate':0.001,
    'clip_value':10.0,
    'weight_decay':0.0001,
    'num_keypoints':25,
    'epochs':100,
    'save_model':True,
    'load_model':False, # for evaluation change to True
    'evaluation_epochs':100,
    # parameters for MINT loss
    'activation_score_threshold':15,
    'region_size':3,
    'bandwidth':0.001,
    'beta_for_overlapping_loss':4.0, # don't allow more than 4 keypoints to overlap
    'kappa_for_it_loss':0.9, # the contribution of the conditional entropy in the construction
    'std_for_heatmap':9.0,
    'threshold_for_heatmap':0.1,
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
  wandb.init(project="MIME", name=model_name,
              # anonymous="allow",
              config=config, 
              # mode="disabled"
              )
  agent=MINT_agent(config, dataset='MIME')
  agent.train()
  for i in trange(config['evaluation_epochs'], desc='Evaluation'):
    agent.eval()
  wandb.run.finish()

# reset the config to config init
config=config_init.copy()

run_experiment(model_name='MINT_MIME', params={})