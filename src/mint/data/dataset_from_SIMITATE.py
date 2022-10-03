import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2

import sys
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DatasetFromSIMITATE(Dataset):
    def __init__(self,config):
        super(DatasetFromSIMITATE,self).__init__()

        self.width = config['width']
        self.height = config['height']

        self.training_demos=config['training_demos']
        self.number_of_stacked_frames=config['number_of_stacked_frames']
        self.evaluation_demos=config['evaluation_demos']

        self.tasks=config['tasks'].split(',')
        
        self.num_frames=config['num_frames']
        
        # self.exported_frames_counter=1

        num_videos=len(self.tasks)*self.training_demos
        # load the dataset if it's already been created
        data_path='datasets/Simitate_{0}videos_{1}frames_{2}x{3}_{4}tasks_{5}training_{6}validation.pt'.format(num_videos,self.num_frames,self.width,self.height,len(self.tasks), self.training_demos, self.evaluation_demos)
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
        return self.demo_start_idx[-(self.evaluation_demos*len(self.tasks)+1)]

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
            video_idx=np.random.randint(0,(self.training_demos+self.evaluation_demos)*len(self.tasks))
        else:
            video_idx=video_idx%((self.training_demos+self.evaluation_demos)*len(self.tasks))
        start_idx=self.demo_start_idx[video_idx]
        end_idx=self.demo_start_idx[video_idx+1]
        # The sample is n_frames frames ending at idx
        sample=self.data[start_idx:end_idx]
        return sample

    def read_frames_from_video(self,path):
        # frames_path=f'annotations/SIMITATE_{self.exported_frames_counter}/frames'
        # annotation_path=f'annotations/SIMITATE_{self.exported_frames_counter}/derender_proposal'
        # # create a folder to store the dataset
        # os.makedirs(frames_path,exist_ok=True)
        # os.makedirs(annotation_path,exist_ok=True)
        # self.exported_frames_counter+=1
        # list all the frames in the path
        file_names=os.listdir(path)
        # sort the frames by their names
        file_names.sort()
        # create a list to store the frames
        frames=[]
        # frame_counter=0
        if len(file_names)>self.num_frames:
            indices=np.linspace(0,len(file_names)-1,self.num_frames).round().astype(int)
        else:
            indices=np.arange(len(file_names)-1)
        for i in indices:
            # read the frame
            frame = cv2.imread("{0}/{1}".format(path,file_names[int(i)]))
            if self.width!=None and self.height!=None:
                # resize the frame
                frame=cv2.resize(frame,(self.width,self.height))
            # cv2.imwrite(f'{frames_path}/frame_{frame_counter}.png', frame)
            # frame_counter+=1
            frames.append(frame)
        # convert the list to a numpy array
        frames=np.array(frames)
        # return the frames
        return frames

    def read_frames(self):
        # iterate over the tasks
        print("Training data:")
        for task in self.tasks:
            demo_idx=0
            for i in range(self.training_demos):
                # choose the index of the demo randomly
                demo_idx+=1#np.random.randint(1,50)
                print('SIMITATE/{0}_{1}'.format(task,demo_idx))
                frames=self.read_frames_from_video('SIMITATE/{0}_{1}'.format(task,demo_idx))
                while frames.shape[0]==0:
                    demo_idx+=1
                    frames=self.read_frames_from_video('SIMITATE/{0}_{1}'.format(task,demo_idx))
                # The length of the frames
                n_frames=frames.shape[0]
                # add the task start index to the list
                self.demo_start_idx.append(self.demo_start_idx[-1]+n_frames)
                # append the frames to the list
                self.data.append(frames)
        print("Testing data:")
        for task in self.tasks:
            demo_idx=0
            for i in range(self.training_demos,self.training_demos+self.evaluation_demos):
                # choose the index of the demo randomly
                demo_idx+=1 #np.random.randint(1,50)
                print('SIMITATE/{0}_{1}'.format(task,demo_idx))
                frames=self.read_frames_from_video('SIMITATE/{0}_{1}'.format(task,demo_idx))
                while frames.shape[0]==0:
                    demo_idx+=1
                    frames=self.read_frames_from_video('SIMITATE/{0}_{1}'.format(task,demo_idx))
                # The length of the frames
                n_frames=frames.shape[0]
                # add the task start index to the list
                self.demo_start_idx.append(self.demo_start_idx[-1]+n_frames)
                # append the frames to the list
                self.data.append(frames)
        # convert the lists to numpy arrays
        self.data=np.concatenate(self.data, axis=0)
        self.demo_start_idx=np.array(self.demo_start_idx)
        # compute the mean and the std of the dataset for each channel
        # self.stats=[np.mean(self.data,axis=(0,1,2)),np.std(self.data,axis=(0,1,2))]
        print('data shape:',self.data.shape)
        print('task_start_idx:',self.demo_start_idx)
        # print('stats:',self.stats)