import argparse
from downstream.dynamics_agent import DynamicsAgent
from tqdm import trange
import wandb

import torch
import numpy as np
import pandas as pd

seeds=[1,2,3,4,5]

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
        
def run_experiment(model_name, params=None):
    print("Running experiment {0}".format(model_name))
    if params is not None:
        for key in params:
            config_data[key] = params[key]
    results = []
    for seed in seeds:
        set_seed(seed)
        model_name_seed = model_name + "_seed={0}".format(seed)
        config_data["model_name"] = model_name_seed
        wandb.init(
            project="Dynamics",
            name=model_name_seed,
            group=model_name,
            # anonymous="allow",
            config=config_data,
            # mode="disabled"
        )
        agent = DynamicsAgent(config_data, args, method="Transporter")
        agent.train()
        for i in trange(config_data["evaluation_epochs"], desc="Evaluation"):
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
        config=config_data,
        # mode="disabled"
    )
    table = wandb.Table(dataframe=final_results)
    wandb.log({"Statistics over seeds": table})
    wandb.run.finish()


parser = argparse.ArgumentParser()
parser.add_argument("--nf_hidden_kp", type=int, default=50)
parser.add_argument("--norm_layer", default="Batch", help="Batch|Instance")
parser.add_argument("--n_kp", type=int, default=0, help="the number of keypoint")
parser.add_argument("--height", type=int, default=0)
parser.add_argument("--width", type=int, default=0)
parser.add_argument(
    "--inv_std", type=float, default=10.0, help="the inverse of std of gaussian mask"
)

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--n_epoch", type=int, default=100)

args = parser.parse_args()
args.lim = [-1.0, 1.0, -1.0, 1.0]

config_data = {
    "model_name": " ",
    # parameters for dataset
    "training_demos": 20,
    "evaluation_demos": 80,
    "number_of_stacked_frames": 8,
    "width": 480,  # Clevrer 480
    "height": 320,  # Clevrer 320
    "channels": 3,  # RGB
    "num_frames": 128,  # number of frames sampled from each video
    "input_width": 128,
    "input_height": 128,
    # parameters for agent
    "learning_rate": 0.001,
    "clip_value": 10.0,
    "weight_decay": 0.000001,
    "epochs": 50,
    "save_model": True,
    "load_model": False,
    "evaluation_epochs": 100,
    "num_keypoints": 25,
    "batch_size": 32,
    # parameters for the interaction network
    "interaction_layers": 1,
    "interaction_latent_size": 32,
    "interaction_history": 4,
    "prediciton_horizon": 3,
    "batch_norm": False,
}
# args.nf_hidden_kp=config_data["num_keypoints"] uncomment if transporter modified
args.height = config_data["input_height"]
args.width = config_data["input_width"]
args.n_kp = config_data["num_keypoints"]
args.n_epoch = config_data["epochs"]

run_experiment(model_name="LearnDynamics_Transporter_CLEVRER", params={})