import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2

import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DatasetFromCLEVRER(Dataset):
    def __init__(self,config):
        super(DatasetFromCLEVRER,self).__init__()

        self.width = config['width']
        self.height = config['height']

        self.training_demos=config['training_demos']
        self.number_of_stacked_frames=config['number_of_stacked_frames']
        self.evaluation_demos=config['evaluation_demos']
        
        self.num_frames=config['num_frames']
        
        num_videos=self.training_demos
        # load the dataset if it's already been created
        data_path='datasets/CLEVRER_{0}videos_{1}frames_{2}x{3}_{4}training_{5}validation.pt'.format(num_videos,self.num_frames,self.width,self.height, self.training_demos, self.evaluation_demos)
        if os.path.exists(data_path):
            print('Loading dataset from',data_path)
            with open(data_path,'rb') as f:
                self.data,self.demo_start_idx=pickle.load(f)
        else:
            print('Creating dataset from data')
            self.data=[]
            self.demo_start_idx=[0]
            self.read_frames()
            # create a folder to store the dataset
            os.makedirs('datasets',exist_ok=True)
            with open(data_path,'wb') as f:
                pickle.dump((self.data,self.demo_start_idx), f , protocol=4)

    def __len__(self):
        # the length is the last element of the task_start_idx list
        return self.demo_start_idx[self.training_demos]

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
            video_idx=np.random.randint(0,(self.training_demos+self.evaluation_demos))
        else:
            video_idx=video_idx%((self.training_demos+self.evaluation_demos))
        start_idx=self.demo_start_idx[video_idx]
        end_idx=self.demo_start_idx[video_idx+1]
        # The sample is n_frames frames ending at idx
        sample=self.data[start_idx:end_idx]
        return sample

    def read_frames_from_video(self,video_path):
        # read the video
        video=cv2.VideoCapture(video_path)
        # get the number of frames
        n_frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # get the frames
        frames=[]
        if n_frames>self.num_frames:
            indices=np.linspace(0,n_frames-1,self.num_frames).round().astype(int)
        else:
            indices=np.arange(n_frames-1)
        for i in indices:
            # read the frame
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if self.width!=None and self.height!=None:
                # resize the frame
                frame=cv2.resize(frame,(self.width,self.height))
            frames.append(frame)
        # close the video
        video.release()
        # convert the list to a numpy array
        frames=np.array(frames)
        # return the frames
        return frames

    def read_frames(self):
        # iterate over the tasks
        print("Training data:")
        # list of the demos for this task
        videos=os.listdir('CLEVRER/video_train')
        videos=sorted([video for video in videos])
        # iterate over training demos from this task
        for i in range(self.training_demos):
            video=videos[i]
            print('CLEVRER/video_train/{0}'.format(video))
            rgb_frames=self.read_frames_from_video('CLEVRER/video_train/{0}'.format(video))
            # the length of the video
            n_frames=rgb_frames.shape[0]
            # add the task start index to the list
            self.demo_start_idx.append(self.demo_start_idx[-1]+n_frames)
            # append the frames to the list
            self.data.append(rgb_frames)
        print("Testing data:")
        # iterate over evaluation demos from this task
        for i in range(self.training_demos,self.training_demos+self.evaluation_demos):
            video=videos[i]
            print('CLEVRER/video_train/{0}'.format(video))
            rgb_frames=self.read_frames_from_video('CLEVRER/video_train/{0}'.format(video))
            # the length of the video
            n_frames=rgb_frames.shape[0]
            # add the task start index to the list
            self.demo_start_idx.append(self.demo_start_idx[-1]+n_frames)
            # append the frames to the list
            self.data.append(rgb_frames)
        # convert the lists to numpy arrays
        self.data=np.concatenate(self.data, axis=0)
        self.demo_start_idx=np.array(self.demo_start_idx)
        print('data shape:',self.data.shape)
        print('task_start_idx:',self.demo_start_idx)
