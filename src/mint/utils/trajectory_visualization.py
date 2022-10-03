import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
import matplotlib._color_data as mcd
from matplotlib.animation import FuncAnimation, FFMpegWriter

import torch
import numpy as np
import os
import random
import wandb

import time

class TrajectoryVisualizer():
    def __init__(self, config, task=None):
        if task=='prediction':
            self.skip=config['interaction_history']
            self.predict=True
        else:
            self.predict=False
        # drop the black color as it will be for inactive keypoints
        keys=random.sample(list(mcd.XKCD_COLORS.keys()), config['num_keypoints'])
        self.colors=[mcd.XKCD_COLORS[k] for k in keys]
        # create a folder to store the videos with the same name as the model name
        os.makedirs('videos',exist_ok=True)
        self.folder_path='videos/'+config['model_name']
        os.makedirs(self.folder_path, exist_ok=True)
        self.counter=0

    def log_video(self,images,coords, active_status, predicted_coords=None, label=None, tensorflow=False):
        # save status into a numpy file with the same name as the label
        # np.save('{0}/{1}.bpy'.format(self.folder_path,label), active_status.cpu().numpy())
        coords=coords.detach().cpu().numpy().astype(np.int32)[0]
        active_status=active_status.detach().cpu().numpy().astype(np.int32)[0]
        alpha_factor=1.0
        fig=plt.Figure()
        ax=fig.add_subplot()
        # remove axis
        ax.axis('off')
        fig.tight_layout()
        fig.subplots_adjust(left=0.015, bottom=0.015, right=0.985, top=0.985, wspace=0.01, hspace=0.01)
        lines = []
        for i in range(len(self.colors)):
            lobj = ax.plot([coords[0,i,0]],[coords[0,i,1]],lw=3,color=self.colors[i], alpha=active_status[0,i,0]*alpha_factor)[0]
            lines.append(lobj)
        scat = ax.scatter([coords[0,:,0]],[coords[0,:,1]], s=70, c=self.colors, marker='o', alpha=active_status[0,:,0]*alpha_factor)
        trajectory_length=np.zeros(len(self.colors), dtype=np.int32)
        factor=0.1
        images=((1-factor)*images+factor*255*np.ones_like(images)).astype(np.uint8)
        im=ax.imshow(images[0])
        def update(i):
            im.set_data(images[i][:,:,::-1])
            for lnum,line in enumerate(lines):
                line.set_color(self.colors[lnum])
                line.set_alpha(active_status[i,lnum,0]*alpha_factor)
                if active_status[i,lnum]>0.1:
                    # set data for each line separately.
                    start=max(i-min(trajectory_length[lnum],5),0)
                    end=min(i+1,len(images))
                    line.set_data(coords[start:end,lnum,0], coords[start:end,lnum,1])
                    trajectory_length[lnum]+=1
                else:
                    trajectory_length[lnum]=-1
            scat.set_color(self.colors)
            scat.set_alpha(active_status[i,:,0]*alpha_factor)
            scat.set_offsets(coords[i,:,:])
            return lines, scat
        animation=FuncAnimation(fig, update, frames=len(images), interval=20, repeat=False)
        writer=FFMpegWriter(fps=5)
        animation.save('{0}/{1}.mp4'.format(self.folder_path,label), writer=writer)
        video=wandb.Video('{0}/{1}.mp4'.format(self.folder_path,label),format="mp4")
        wandb.log({'Videos/{0}'.format(label):video})
        plt.close('all')

    def log_video_prediction(self,images,coords, active_status, predicted_coords=None, label=None, tensorflow=False):
        # save status into a numpy file with the same name as the label
        # np.save('{0}/{1}.bpy'.format(self.folder_path,label), active_status.cpu().numpy())
        alpha_factor=0.3
        coords=coords.detach().cpu().numpy().astype(np.int32)[0]
        active_status=active_status.detach().cpu().numpy().astype(np.int32)[0]
        predicted_coords=predicted_coords.detach().cpu().numpy().astype(np.int32)[0]
        images=images[self.skip:][::5]
        coords=coords[self.skip:][::5]
        active_status=active_status[self.skip:][::5]
        predicted_coords=predicted_coords[self.skip:][::5]
        predictions=[predicted_coords]
        shifted_pred_coords=predicted_coords.copy()
        for i in range(3):
            shifted_pred_coords=np.roll(shifted_pred_coords,-1,axis=0)
            predictions.append(shifted_pred_coords)
        pred_lines=[]
        fig=plt.Figure()
        ax=fig.add_subplot()
        # remove axis
        ax.axis('off')
        fig.tight_layout()
        fig.subplots_adjust(left=0.015, bottom=0.015, right=0.985, top=0.985, wspace=0.01, hspace=0.01)
        lines = []
        for i in range(len(self.colors)):
            lobj = ax.plot([coords[0,i,0]],[coords[0,i,1]],lw=3,color=self.colors[i], alpha=active_status[0,i,0])[0]
            lines.append(lobj)
            lobj = ax.plot([predictions[0][0,i,0]],[predictions[0][0,i,1]],lw=3,color=self.colors[i], alpha=active_status[0,i,0]*alpha_factor)[0]
            pred_lines.append(lobj)
        scat = ax.scatter([coords[0,:,0]],[coords[0,:,1]], s=50, c=self.colors, marker='o', alpha=active_status[0,:,0])
        pred_scat=[]
        for i in range(4):
            pred_scat.append(ax.scatter([predictions[i][0,:,0]],[predictions[i][0,:,1]], s=50, c=self.colors, marker='o', alpha=active_status[0,:,0]*alpha_factor))
        trajectory_length=np.zeros(len(self.colors), dtype=np.int32)
        pred_trajectory_length=np.zeros(len(self.colors), dtype=np.int32)
        # overlay the next 3 images over the current image
        factor=0.5
        shifted_images=images.copy()
        for i in range(3):
            shifted_images= (1-factor)*shifted_images + factor*np.roll(shifted_images,-1,axis=0)
        images= ((1-factor)*images+factor*shifted_images).astype(np.uint8)
        im=ax.imshow(images[0])
        def update(i):
            im.set_data(images[i][:,:,::-1])
            for lnum,line in enumerate(lines):
                line.set_color(self.colors[lnum])
                line.set_alpha(active_status[i,lnum,0])
                if active_status[i,lnum]>0.1:
                    # set data for each line separately.
                    start=max(i-min(trajectory_length[lnum],2),0)
                    end=min(i+1,len(images))
                    line.set_data(coords[start:end,lnum,0], coords[start:end,lnum,1])
                    trajectory_length[lnum]+=1
                else:
                    trajectory_length[lnum]=-1
                for lnum,line in enumerate(pred_lines):
                    line.set_color(self.colors[lnum])
                    line.set_alpha(active_status[i,lnum,0]*alpha_factor)
                    x=[]
                    y=[]
                    if i+self.skip < len(images):
                        for p in range(4):
                            if active_status[i+p,lnum]>0.1:
                                x.append(predictions[p][i,lnum,0])
                                y.append(predictions[p][i,lnum,1])
                        line.set_data(x, y)
            scat.set_color(self.colors)
            scat.set_alpha(active_status[i,:,0])
            scat.set_offsets(coords[i,:,:])
            for p in range(4):
                if i+self.skip<len(images):
                    pred_scat[p].set_color(self.colors)
                    pred_scat[p].set_alpha(active_status[i,:,0]*alpha_factor)
                    pred_scat[p].set_offsets(predictions[p][i,:,:])
            return lines, scat, pred_lines, pred_scat[0], pred_scat[1], pred_scat[2] 
        animation=FuncAnimation(fig, update, frames=len(images), interval=20, repeat=False)
        writer=FFMpegWriter(fps=5)
        animation.save('{0}/{1}.mp4'.format(self.folder_path,label), writer=writer)
        video=wandb.Video('{0}/{1}.mp4'.format(self.folder_path,label),format="mp4")
        wandb.log({'Videos/{0}'.format(label):video})
        plt.close('all')