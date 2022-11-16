import argparse
from baselines_agents.agent_transporter.agent import TransporterAgent
from tqdm import trange
import wandb


def run_experiment(model_name, params):
    print("Running experiment {0}".format(model_name))
    config_data["model_name"] = model_name
    wandb.init(
        project="MAGICAL",
        name=model_name,
        # anonymous="allow",
        config=config_data,
        mode="disabled",
    )
    agent = TransporterAgent(config_data, args, dataset="MAGICAL")
    agent.train()
    for i in trange(config_data["evaluation_epochs"], desc="Evaluation"):
        agent.eval()
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
    "tasks": 3,
    "training_demos": 24,
    "evaluation_demos": 0,
    "number_of_stacked_frames": 8,
    "width": 96,  # Clevrer 480
    "height": 96,  # Clevrer 320
    "channels": 3,  # RGB
    "num_frames": 128,  # number of frames sampled from each video
    "input_width": 96,
    "input_height": 96,
    # parameters for agent
    "save_model": True,
    "load_model": False,
    "num_epochs": 100,
    "evaluation_epochs": 5,
    "num_keypoints": 25,
    "batch_size": 8,
}
args.nf_hidden_kp=config_data["num_keypoints"]
args.height = config_data["input_height"]
args.width = config_data["input_width"]
args.n_kp = config_data["num_keypoints"]
args.n_epoch = config_data["num_epochs"]

run_experiment(model_name="Transporter_MAGICAL", params={})
