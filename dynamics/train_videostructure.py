from downstream.dynamics_agent import DynamicsAgent
from tqdm import trange
import wandb

import torch
import numpy as np
import pandas as pd
torch.autograd.set_detect_anomaly(True)

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
            project="Dynamics",
            name=model_name_seed,
            group=model_name,
            # anonymous="allow",
            config=config,
            # mode="disabled"
        )
        agent = DynamicsAgent(config, method="Video_structure")
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
        project="Dynamics",
        name=model_name,
        group=model_name,
        # anonymous="allow",
        config=config,
        # mode="disabled"
    )
    table = wandb.Table(dataframe=final_results)
    wandb.log({"Statistics over seeds": table})
    wandb.run.finish()


config = {
    "model_name": "",
    # parameters for dataset
    "training_demos": 20,
    "evaluation_demos": 80,
    "number_of_stacked_frames": 16,
    "width": 480,  # Clevrer 480
    "height": 320,  # Clevrer 320
    "channels": 3,  # RGB
    "num_frames": 128,  # number of frames sampled from each video
    "input_width": 64,
    "input_height": 64,
    # parameters for model
    "batch_size": 32,
    "save_model": True,
    "load_model": False,
    "epochs": 15,
    "num_keypoints": 25,
    "learning_rate": 0.001,
    "weight_decay": 0.000001,  # equivalent to L2 regularization in keras
    "padding": "same",
    "std_for_gaussian_maps": 1.5,
    "evaluation_epochs": 100,
    # parameters for losses
    "sigma_for_separation_loss": 0.002,
    "num_encoder_filters": 32,
    "heatmap_ratio": 4.0,
    "layers_per_scale": 2,
    "clip_value": 10.0,
    # parameters for the interaction network
    "interaction_layers": 1,
    "interaction_latent_size": 32,
    "interaction_history": 4,
    "prediciton_horizon": 3,
    "batch_norm": False,
}

run_experiment(model_name="LearnDynamics_Video_structure_CLEVRER", params={})