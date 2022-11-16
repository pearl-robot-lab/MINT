from mint.agents.agent import MINT_agent
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
    'number_of_stacked_frames':3,
    'width':480, # Clevrer 480
    'height':320, # Clevrer 320
    'channels':3, # RGB
    'num_frames':128, # number of frames sampled from each video
    # parameters for agent
    'num_layers':2,
    'batch_size':32,
    'learning_rate':0.001,
    'clip_value':10.0,
    'weight_decay':0.00001,
    'num_keypoints':25,
    'epochs':50,
    'save_model':True,
    'load_model':False, # for evaluation change to True
    'evaluation_epochs':100,
    # parameters for MINT loss
    'activation_score_threshold':15,
    'region_size':3,
    'bandwidth':0.001,
    'beta_for_overlapping_loss':4.0, # don't allow more than 4 keypoints to overlap
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

seeds=[1,2,3,4,5]

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
        
def run_experiment(model_name, params=None):
    print("Running experiment {0}".format(model_name))
    if params is not None:
        for key in params:
            config[key] = params[key]
    results = []
    for seed in seeds:
        set_seed(seed)
        model_name_seed = model_name + "_seed={0}".format(seed)
        config["model_name"] = model_name_seed
        wandb.init(
            project="CLEVRER_seeds",
            name=model_name_seed,
            group=model_name,
            # anonymous="allow",
            config=config,
            # mode="disabled"
        )
        agent = MINT_agent(config, dataset="CLEVRER")
        agent.train()
        for i in trange(config["evaluation_epochs"], desc="Evaluation"):
            agent.collect_qual_results()
        result = agent.report_results()
        results.append(result)
        wandb.run.finish()
    results = pd.concat(results)
    mean = results.mean()
    std = results.std()
    confidence_interval = 2 * std / np.sqrt(len(seeds))
    data = {}
    for key in mean.keys():
        data[key] = [mean[key], confidence_interval[key]]
    final_results = pd.DataFrame.from_dict(
        data, orient="index", columns=["mean", "confidence_interval"]
    )
    wandb.init(
        project="CLEVRER_seeds",
        name=model_name,
        group=model_name,
        # anonymous="allow",
        config=config,
        # mode="disabled"
    )
    table = wandb.Table(dataframe=final_results)
    wandb.log({"Statistics over seeds": table})
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
run_experiment(model_name="MINTwoReg", params=ablation_params)

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
run_experiment(model_name="MINT_static", params=ablation_params)

# reset the config to config init
config = config_init.copy()
ablation_params = {
    "activation_score_threshold": 15,
    "region_size": 3,
    "std_for_heatmap": 9.0,
    "threshold_for_heatmap": 0.1,
    "thresholded_heatmap_scale": 3.5,
    "bandwidth": 0.001,
    "beta_for_overlapping_loss": 0.0, 
    "kappa_for_it_loss": 0.0,
    "movement_weight": 0.0,
    "masked_entropy_loss_weight": 100.0,
    "conditional_entropy_loss_weight": 100.0,
    "status_weight": 0.0,
    "information_transport_loss_weight": 0.0,
    "overlap_weight": 0.0,
}
region_sizes = [3, 5, 7]
ks = [0, 0.5, 0.9, 1]
ms = [0, 1]
os = [2, 4]
# enropy region_size ablation
for r in region_sizes:
    name = f"(Ablation) IM ({r}x{r})"
    ablation_params["region_size"] = r
    run_experiment(model_name=name, params=ablation_params)
ablation_params["region_size"] = 3
ablation_params["information_transport_loss_weight"] = 20
# ablation for it loss
for m in ms:
    ablation_params["movement_weight"] = m
    for k in ks:
        ablation_params["kappa_for_it_loss"] = k
        name = f"(Ablation) IM +IT (m={m},k={k})"
        run_experiment(model_name=name, params=ablation_params)
ablation_params["kappa_for_it_loss"] = 0.5
ablation_params["movement_weight"] = 1.0
m=config['movement_weight']
k=config['kappa_for_it_loss']
# ablation for overlapping loss
ablation_params["overlap_weight"] = 30
for o in os:
    ablation_params["beta_for_overlapping_loss"] = o
    name = f"(Ablation) IM +IT (m={m},k={k}) + O({o})"
    run_experiment(model_name=name, params=ablation_params)
ablation_params["beta_for_overlapping_loss"] = 4
b=config['beta_for_overlapping_loss']
# run the full experiment
name = f"(Ablation) IM +IT (m={m},k={k}) + O({b})"
run_experiment(model_name=name, params=ablation_params)