import argparse
from baselines_agents.agent_transporter.agent import TransporterAgent
from tqdm import trange
import wandb

def run_experiment(model_name, params):
    print('Running experiment {0}'.format(model_name))
    config_data['model_name']=model_name
    wandb.init(project="SIMITATE", name=model_name ,
                # anonymous="allow",
                config=config_data, 
                # mode="disabled"
                )
    agent=TransporterAgent(config_data, args, dataset='SIMITATE')
    agent.train()
    for i in trange(config_data['evaluation_epochs'], desc='Evaluation'):
        agent.eval()
    wandb.run.finish()


parser = argparse.ArgumentParser()
parser.add_argument('--nf_hidden_kp', type=int, default=50)
parser.add_argument('--norm_layer', default='Batch', help='Batch|Instance')
parser.add_argument('--n_kp', type=int, default=0, help="the number of keypoint")
parser.add_argument('--height', type=int, default=0)
parser.add_argument('--width', type=int, default=0)
parser.add_argument('--inv_std', type=float, default=10., help='the inverse of std of gaussian mask')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--n_epoch', type=int, default=100)

args = parser.parse_args()
args.lim = [-1., 1., -1., 1.]

config_data={
    'model_name':' ',
    # parameters for dataset
    'tasks':'bring,rearrange,pick_and_place,stack',
    'training_demos':20,
    'evaluation_demos':5,
    'number_of_stacked_frames':6,
    'width':480, # SIMITATE 960
    'height':272, # SIMITATE 540
    'channels':3, # RGB
    'num_frames':100, # number of frames sampled from each video
    'input_width':128,
    'input_height':128,
    'save_model':True,
    'load_model':False,
    'num_epochs':100,
    'evaluation_epochs':100,
    'num_keypoints':25,
    'batch_size':8,
}
args.nf_hidden_kp=config_data["num_keypoints"]
args.height = config_data['input_height']
args.width = config_data['input_width']
args.n_kp=config_data['num_keypoints']
args.n_epoch=config_data['num_epochs']

run_experiment(model_name='Transporter SIMITATE', params={})