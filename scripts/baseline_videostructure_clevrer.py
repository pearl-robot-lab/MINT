from baselines_agents.agent_video_structure.agent import VideoStructureAgent
from tqdm import trange
import wandb

import pandas as pd
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)

seeds=[1,2,3,4,5]

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

def run_experiment(model_name, params):
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
        agent = VideoStructureAgent(config, dataset="CLEVRER")
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

config={
    'model_name':'',
    # parameters for dataset
    'training_demos':20,
    'evaluation_demos':80,
    'number_of_stacked_frames':16,
    'width':480, # Clevrer 480
    'height':320, # Clevrer 320
    'channels':3, # RGB
    'num_frames':128, # number of frames sampled from each video
    'input_width':64,
    'input_height':64,
    # parameters for model
    'batch_size':16,
    'save_model':True,
    'load_model':False,
    'epochs':100,
    'num_keypoints':25,
    'learning_rate':0.001,
    'weight_decay':0.00005,#equivalent to L2 regularization in keras
    'padding':'same',
    'std_for_gaussian_maps':1.5,
    'evaluation_epochs':100,
    # parameters for losses
    'sigma_for_separation_loss':0.002,
    'num_encoder_filters':32,
    'heatmap_ratio':4.0,
    'layers_per_scale':2,
    'clip_value':1.0,
    # weights for losses
    'separation_loss_weight':0.02,
    'sparsity_loss_weight':0.1, # heatmap_regularization
    'reconstruction_loss_weight':0.01,
}

run_experiment(model_name='Video_structure_CLEVRER', params={})