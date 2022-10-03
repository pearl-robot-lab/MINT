import os
from cv2 import sort
import numpy as np
import magical
import glob

import gym

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2

import magical

import gzip
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DatasetFromMAGICAL(Dataset):
    def __init__(self,config):
        super(DatasetFromMAGICAL,self).__init__()

        self.width = config['width']
        self.height = config['height']

        self.tasks=config['tasks']
        
        self.training_demos=config['training_demos']
        self.number_of_stacked_frames=config['number_of_stacked_frames']
        self.evaluation_demos=config['evaluation_demos']
        
        self.num_frames=config['num_frames']
        
        num_videos=self.training_demos
        # load the dataset if it's already been created
        data_path='datasets/MAGICAL_{0}videos_{1}frames_{2}x{3}_{4}training_{5}validation.pt'.format(num_videos,self.num_frames,self.width,self.height, self.training_demos, self.evaluation_demos)
        if os.path.exists(data_path):
            print('Loading dataset from',data_path)
            with open(data_path,'rb') as f:
                self.data,self.demo_start_idx=pickle.load(f)
        else:
            print('Creating dataset from data')
            magical.register_envs()
            # download trajectories
            magical.try_download_demos(dest="demos")
            self.data=[]
            self.demo_start_idx=[0]
            self.read_frames()
            # create a folder to store the dataset
            os.makedirs('datasets',exist_ok=True)
            with open(data_path,'wb') as f:
                pickle.dump((self.data,self.demo_start_idx), f , protocol=4)

    def __len__(self):
        # the length is the last element of the task_start_idx list
        return self.demo_start_idx[-(self.evaluation_demos*self.tasks+1)]

    def __getitem__(self, idx):
        # find to which task the idx belongs by using the task_start_idx
        task=np.where(self.demo_start_idx<=idx)[0][-1]
        # if the idx is less than task_start_idx[task]+self.number_of_stacked_frames then shift it
        if idx<self.demo_start_idx[task]+self.number_of_stacked_frames:
            idx=self.demo_start_idx[task]+self.number_of_stacked_frames
        # The sample is number_of_stacked_frames frames ending at idx
        samlpe=self.data[idx-self.number_of_stacked_frames:idx]
        return samlpe

    def show_sample(self,idx):
        sample=self.__getitem__(idx)
        # stack the frames of the human and robot
        frames=np.concatenate(sample, axis=0)
        # use RGB to show the frames
        frames=frames[:,:,:3].astype(np.uint8)
        # frames=cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)
        # show the frames
        cv2.imshow('sample',frames)
        cv2.waitKey(0)

    def sample_video_from_data(self, video_idx=None):
        # sample the index of the starting frame
        if video_idx is None:
            video_idx=np.random.randint(0,(self.training_demos+self.evaluation_demos)*self.tasks)
        else:
            video_idx=video_idx%((self.training_demos+self.evaluation_demos)*self.tasks)
        start_idx=self.demo_start_idx[video_idx]
        end_idx=self.demo_start_idx[video_idx+1]
        # The sample is n_frames frames ending at idx
        sample=self.data[start_idx:end_idx]
        return sample

    def read_frames(self):
        # iterate over the tasks
        folders=os.listdir('demos/')
        folders=sorted([folder for folder in folders if os.path.isdir('demos/'+folder)])
        folders=['move-to-region','move-to-corner', 'make-line']
        print("Training data:")
        for i in range(self.tasks):
            demos=os.listdir('demos/'+folders[i]+'/')
            demos=sorted([demo for demo in demos])
            for d in range(self.training_demos):
                print('demos/'+folders[i]+'/'+demos[d])
                demo_dicts = list(magical.load_demos(glob.glob('demos/'+folders[i]+'/'+demos[d])))
                for demo_dict in demo_dicts:
                    frames=[]
                    # each demo dict has keys ['trajectory', 'score', 'env_name']
                    # (trajectory contains the actual data, and score is generally 1.0 for demonstrations)
                    for obs in demo_dict['trajectory'][1]:
                        frame=obs['allo']
                        if self.width!=None and self.height!=None:
                            # resize the frame
                            frame=cv2.resize(frame,(self.width,self.height), interpolation=cv2.INTER_AREA)
                        frames.append(frame)
                    self.demo_start_idx.append(self.demo_start_idx[-1]+len(frames))
                    self.data.append(np.array(frames))
        print("Testing data:")
        for i in range(self.tasks):
            demos=os.listdir('demos/'+folders[i]+'/')
            demos=sorted([demo for demo in demos])
            for d in range(self.training_demos,self.training_demos+self.evaluation_demos):
                print('demos/'+folders[i]+'/'+demos[d])
                demo_dicts = list(magical.load_demos(glob.glob('demos/'+folders[i]+'/'+demos[d])))
                for demo_dict in demo_dicts:
                    frames=[]
                    # each demo dict has keys ['trajectory', 'score', 'env_name']
                    # (trajectory contains the actual data, and score is generally 1.0 for demonstrations)
                    for obs in demo_dict['trajectory'][1]:
                        frame=obs['allo']
                        if self.width!=None and self.height!=None:
                            # resize the frame
                            frame=cv2.resize(frame,(self.width,self.height))
                        frames.append(frame)
                    self.demo_start_idx.append(self.demo_start_idx[-1]+len(frames))
                    self.data.append(np.array(frames))
        # convert the lists to numpy arrays
        self.data=np.concatenate(self.data, axis=0)
        self.demo_start_idx=np.array(self.demo_start_idx)
        print('data shape:',self.data.shape)
        print('task_start_idx:',self.demo_start_idx)
