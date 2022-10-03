import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
import json
import pycocotools._mask as _mask
import wandb

class DynamicsEvaluator():
    def __init__(self, config):
        # list of the annotation files for the whole dataset
        proposals=os.listdir('CLEVRER/derender_proposals')
        # sort the annotations
        self.proposals=sorted([proposal for proposal in proposals])
        self.skip=config['interaction_history']
        self.num_keypoints=config['num_keypoints']
        self.width=config['width']
        self.height=config['height']
        self.horizon=config['prediciton_horizon']
        self.success_rates=[]
        for i in range(4):
            self.success_rates.append([])
    
    def collect(self, video_idx, coords, status, predicted_coords, predicted_status):
        predicted_coords=predicted_coords.detach().cpu().numpy().astype(np.int32)[0]
        predicted_status=predicted_status.detach().cpu().numpy().astype(np.int32)[0]
        coords= coords.detach().cpu().numpy().astype(np.int32)[0]
        status=status.detach().cpu().numpy().astype(np.int32)[0]
        # for multistep prediction
        for h in range(self.horizon):
            # shift the prediction one time step to compare to the ground truth
            predicted_coords[:,:,h]=np.roll(predicted_coords[:,:,h],-(h+1), axis=0)
        # print('Video_idx: {0} Annotation file: CLEVRER/derender_proposals/{1}'.format(video_idx, self.proposals[video_idx]))
        with open('CLEVRER/derender_proposals/{0}'.format(self.proposals[video_idx])) as f:
            data=json.load(f)
        # objects in the scene
        objects={}
        num_frames=len(data['frames'])-1
        num_objects=np.zeros(num_frames)
        detected_objects=np.zeros(num_frames)
        success_rate=[]
        for i in range(self.horizon):
            success_rate.append([])
        # iterate over the frames
        for i,frame in enumerate(data['frames']):
            if i>=126:
                continue
            # get the number of objects
            num_objects=len(frame['objects'])
            detected_objects=np.zeros(self.horizon)
            # iterate over objects
            for obj in frame['objects']:
                # object name
                object_name="{0}_{1}_{2}".format(obj['color'], obj['material'], obj['shape'])
                # create an entry in the dictionary if it's not already there
                if object_name not in objects.keys():
                    # kp the keypoint association at time step t-1
                    # pred predicted keypoint association for time step t (predicted at timestep t-1)
                    objects[object_name]={'kp':np.zeros((self.horizon+1,self.num_keypoints)), 'pred':np.zeros((self.horizon,self.num_keypoints))}
                # decode the mask (time step t)
                mask=_mask.decode([obj['mask']])
                # update the keypoints in the object
                for k in range(self.num_keypoints):
                    # fill in the association of keypoints and objects
                    # consider only active keypoints in time step t for predictions
                    if status[i,k,0]>0.9:
                        if i>self.skip and i+self.skip < num_frames:
                            for p in range(self.horizon):
                                x=predicted_coords[i,k,p,0]
                                y=predicted_coords[i,k,p,1]
                                if x<self.width and y<self.height and mask[y,x]==1:
                                    objects[object_name]['pred'][p][k]=1
                                else:
                                    objects[object_name]['pred'][p][k]=0
                                # keep the keypoint in the object only if the input coordinates (updated at time t-p) was detecting the object
                                objects[object_name]['pred'][p][k]*=objects[object_name]['kp'][p+1][k]
                        # update the keypoints in the object for time step t
                        # this will be used to compare with prediciton at next time step
                        if coords[i,k,0]<self.width and coords[i,k,1]<self.height and mask[coords[i,k,1],coords[i,k,0]]==1:
                            objects[object_name]['kp'][0][k]=1
                        else:
                            objects[object_name]['kp'][0][k]=0
                # shift the kp to add the last occurence
                objects[object_name]['kp']=np.roll(objects[object_name]['kp'],1,axis=0)
                if i>self.skip:
                    for p in range(self.horizon):
                        # an object is detected if a prediciton lies in it
                        if np.sum(objects[object_name]['pred'][p])>0:
                            detected_objects[p]+=1
            if i>self.skip+self.horizon and i+self.skip+self.horizon < num_frames:
                for p in range(self.horizon): 
                    # the success rate for this frame is the percentage of object detected by the prediciton
                    success_rate[p].append(detected_objects[p]/num_objects)
        for p in range(self.horizon):
            # the mean of success rates over all frames
            mean_success_rate=np.array(success_rate[p]).mean()
            self.success_rates[p].append(mean_success_rate)
        
    def save_to_file(self, model_name=None):
        # create a folder to store the dataset
        os.makedirs('results',exist_ok=True)
        file_path="results/{0}.xlsx".format(model_name)
        writer=pd.ExcelWriter(file_path, engine='openpyxl')
        data_frame=pd.DataFrame({"Success rate - 1 step prediction" : self.success_rates[0],
                                 "Success rate - 2 steps prediction" : self.success_rates[1],
                                 "Success rate - 3 steps prediction" : self.success_rates[2],
                                 "Success rate - 4 steps prediction" : self.success_rates[3],
                                 })
        table=wandb.Table(dataframe=data_frame)
        wandb.log({"Results" : table})
        data_frame.to_excel(writer, sheet_name='per-video results', index=False)
        result_stats=data_frame.describe()
        table=wandb.Table(dataframe=result_stats)
        wandb.log({"Statistics over all videos" : table})
        result_stats.to_excel(writer, sheet_name='Statistics')
        writer.save()
        writer.close()
        return data_frame
        