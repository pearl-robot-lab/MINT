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
    'num_frames':100, # number of frames sampled from each video
    # parameters for agent
    'num_layers':2,
    'batch_size':32,
    'learning_rate':0.001,
    'clip_value':10.0,
    'weight_decay':0.0001,
    'num_keypoints':25,
    'num_filters':16,
    'epochs':200,
    'save_model':True,
    'load_model':False,
    'evaluation_epochs':100,
    # parameters for MINT loss
    'activation_score_threshold':15,
    'region_size':3,
    'bandwidth':0.001,
    'alpha_for_status_loss':5.0,
    'beta_for_overlapping_loss':4.0,
    "kappa_for_it_loss": 0.9,  # the contribution of the conditional entropy in the construction
    "movement_weight": 1.0,
    "std_for_heatmap": 9.0,
    "threshold_for_heatmap": 0.1,
    "thresholded_heatmap_scale": 3.5,
    "masked_entropy_loss_weight": 0.0,
    "conditional_entropy_loss_weight": 0.0,
    "status_weight": 0.0,
    "information_transport_loss_weight": 0.0,
    "overlap_weight": 0.0,
}

def run_experiment(model_name, params):
  print('Running experiment {0}'.format(model_name))
  for key in params:
    config[key]=params[key]
  config['model_name']=model_name
  wandb.init(project="MINT_MIME Ablation", name=model_name ,
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
config = config_init.copy()
ablation_params = {
    "activation_score_threshold": 15,
    "region_size": 3,
    "std_for_heatmap": 9.0,
    "threshold_for_heatmap": 0.1,
    "thresholded_heatmap_scale": 3.5,
    "bandwidth": 0.001,
    "beta_for_overlapping_loss": 0.0,  # Regularization # don't allow more than 4 keypoints to overlap
    "kappa_for_it_loss": 0.9,  # the contribution of the conditional entropy in the construction
    "movement_weight": 0.0, # Regularization
    "masked_entropy_loss_weight": 100.0, # Info.
    "conditional_entropy_loss_weight": 100.0, # Info.
    "status_weight": 0.0, # Regularization
    "information_transport_loss_weight": 20.0, # Info.
    "overlap_weight": 0.0, # Regularization
}
run_experiment(model_name="MINTwoReg_MIME", params=ablation_params)

# reset the config to config init
config = config_init.copy()
ablation_params = {
    "activation_score_threshold": 15,
    "region_size": 3,
    "std_for_heatmap": 9.0,
    "threshold_for_heatmap": 0.1,
    "thresholded_heatmap_scale": 3.5,
    "bandwidth": 0.001,
    "beta_for_overlapping_loss": 4.0, # Static
    "kappa_for_it_loss": 0.0, # Temporal 
    "movement_weight": 0.0, # Temporal
    "masked_entropy_loss_weight": 100.0, # Static
    "conditional_entropy_loss_weight": 0.0, # Temporal
    "status_weight": 10.0, # Static
    "information_transport_loss_weight": 0.0, # Temporal
    "overlap_weight": 30.0, # Static
}
run_experiment(model_name="MINT_static_MIME", params=ablation_params)