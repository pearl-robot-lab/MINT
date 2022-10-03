import argparse
from baselines_agents.agent_transporter.agent import TransporterAgent
from tqdm import trange
import wandb

def run_experiment(model_name, params):
    print('Running experiment {0}'.format(model_name))
    config_data['model_name']=model_name
    wandb.init(project="MIME", name=model_name ,
                # anonymous="allow",
                config=config_data, 
                # mode="disabled"
                )
    agent=TransporterAgent(config_data, args, dataset='MIME')
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
    'tasks':'1,2,3,4',#,5,6,7,8,9,10,11,12,13,15,16,17,19,20', # 14 and 18 are not available 
    'training_demos':20,
    'evaluation_demos':5,
    'number_of_stacked_frames':8,
    'width': 320, # MIME 640 -> after crop to 320x240
    'height': 240, # MIME 240
    'channels':3, # RGB
    'num_frames':50, # number of frames sampled from each video
    'input_width':128,
    'input_height':128,
    'save_model':True,
    'load_model':False,
    'num_epochs':100,
    'evaluation_epochs':100,
    'num_keypoints':25,
    'batch_size':8,
}
args.height = config_data['input_height']
args.width = config_data['input_width']
args.n_kp=config_data['num_keypoints']
args.n_epoch=config_data['num_epochs']

run_experiment(model_name='Transporter_MIME', params={})