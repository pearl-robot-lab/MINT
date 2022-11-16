from mint.agents.agent import MINT_agent
from tqdm import trange
import wandb

import torch
torch.autograd.set_detect_anomaly(True)

config_init={
    'model_name':' ',
    # parameters for dataset
    'tasks':'bring,rearrange,pick_and_place,stack',
    'training_demos':20,
    'evaluation_demos':5,
    'number_of_stacked_frames':2,
    'width':480, # SIMITATE 960
    'height':272, # SIMITATE 540
    'channels':3, # RGB
    'num_frames':100, # number of frames sampled from each video
    # parameters for agent
    'num_layers':2,
    'batch_size':8,
    'learning_rate':0.001,
    'clip_value':10.0,
    'weight_decay':0.0001,
    'num_keypoints':64,
    'num_filters':16,
    'epochs':50,
    'save_model':True,
    'load_model':False,
    'evaluation_epochs':100,
    # parameters for MINT loss
    'activation_score_threshold':10,
    'region_size':3,
    'bandwidth':0.001,
    'alpha_for_status_loss':5.0,
    'beta_for_overlapping_loss':4.0,
    "kappa_for_it_loss": 0.9,  # the contribution of the conditional entropy in the construction
    "movement_weight": 1.0,
    "std_for_heatmap": 9.0,
    "threshold_for_heatmap": 0.5,
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
  wandb.init(project="MINT_SIMITATE Ablation", name=model_name ,
            # anonymous="allow",
            config=config, 
            # mode="disabled"
            )
  agent=MINT_agent(config, dataset='SIMITATE')
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
run_experiment(model_name="MINTwoReg_SIMITATE", params=ablation_params)

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
run_experiment(model_name="MINT_static_SIMITATE", params=ablation_params)