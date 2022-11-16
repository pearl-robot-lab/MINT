import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import json
import pycocotools._mask as _mask
import wandb

class ResultCollector:
    def __init__(self, config):
        # list of the annotation files for the whole dataset
        proposals = os.listdir("CLEVRER/derender_proposals")
        # sort the annotations
        self.proposals = sorted([proposal for proposal in proposals])
        self.num_keypoints = config["num_keypoints"]
        self.width = config["width"]
        self.height = config["height"]
        self.percentage_detected = []
        self.missed_keypoints = []
        self.redundant_keypoints = []
        self.percentage_tracked = []
        self.mean_object_area = 1796.8
        # self.get_mean_area()

    def get_mean_area(self, num_videos=100):
        video_mean_area = []
        for video_idx in range(num_videos):
            # print('Video_idx: {0} Annotation file: CLEVRER/derender_proposals/{1}'.format(video_idx, self.proposals[video_idx]))
            with open(
                "CLEVRER/derender_proposals/{0}".format(self.proposals[video_idx])
            ) as f:
                data = json.load(f)
            mean_areas = []
            # iterate over the frames
            for i, frame in enumerate(data["frames"]):
                if i == 127:
                    continue
                areas = []
                # iterate over objects
                for obj in frame["objects"]:
                    # decode the mas
                    mask = _mask.decode([obj["mask"]])
                    # the area of the object's mask
                    area = mask.sum()
                    areas.append(area)
                mean_areas.append(np.array(areas).mean())
            video_mean_area.append(np.array(mean_areas).mean())
        mean_area = np.array(video_mean_area).mean()
        print(mean_area)
        # mean area for the first 100 videos = 1796.8
        # input()

    def collect(self, video_idx, coords, status):
        coords = coords.detach().cpu().numpy().astype(np.int32)[0]
        status = status.detach().cpu().numpy().astype(np.int32)[0]
        # print('Video_idx: {0} Annotation file: CLEVRER/derender_proposals/{1}'.format(video_idx, self.proposals[video_idx]))
        with open(
            "CLEVRER/derender_proposals/{0}".format(self.proposals[video_idx])
        ) as f:
            data = json.load(f)
        # objects in the scene
        objects = {}
        num_frames = len(data["frames"]) - 1
        num_objects = np.zeros(num_frames)
        detected_objects = np.zeros(num_frames)
        tracked_objects = np.zeros(num_frames)
        missed_keypoints = np.zeros(num_frames)
        redundant = np.ones(num_frames)
        areas = np.zeros(num_frames)
        # iterate over the frames
        for i, frame in enumerate(data["frames"]):
            if i == 127:
                continue
            # get the number of objects
            num_objects[i] = len(frame["objects"])
            redundancy = []
            aggregated_mask = np.zeros((self.height, self.width, 1))
            # iterate over objects
            for obj in frame["objects"]:
                # object name
                object_name = "{0}_{1}_{2}".format(
                    obj["color"], obj["material"], obj["shape"]
                )
                # create an entry in the dictionary if it's not already there
                if object_name not in objects.keys():
                    objects[object_name] = {
                        "kp": np.zeros(self.num_keypoints),
                        "old_kp": np.zeros(self.num_keypoints),
                    }
                # decode the mas
                mask = _mask.decode([obj["mask"]])
                # the area of the object's mask
                area = mask.sum()
                # update the keypoints in the object
                for k in range(self.num_keypoints):
                    # keep the old keypoint to check the tracking
                    objects[object_name]["old_kp"][k] = objects[object_name]["kp"][k]
                    # consider only active points
                    if status[i, k, 0]:
                        # if the keypoint lay on the object
                        if (
                            coords[i, k, 0] < self.width
                            and coords[i, k, 1] < self.height
                            and mask[coords[i, k, 1], coords[i, k, 0]] == 1
                        ):
                            # update the kp array
                            objects[object_name]["kp"][k] = 1
                        else:
                            objects[object_name]["kp"][k] = 0
                    # keep the old_kp only if the keypoint track the object
                    objects[object_name]["old_kp"][k] *= objects[object_name]["kp"][k]
                if np.sum(objects[object_name]["kp"]) > 0:
                    detected_objects[i] += 1
                    # A_k = 1796.84
                    redundancy.append(
                        abs(
                            area
                            - np.sum(objects[object_name]["kp"] * self.mean_object_area)
                        )
                        / area
                    )
                if np.sum(objects[object_name]["old_kp"]) > 0:
                    tracked_objects[i] += 1
                aggregated_mask += mask
            redundant[i] = np.array(redundancy).mean()
            # find the number of missed points in each frame
            for k in range(self.num_keypoints):
                # consider only active points
                if (
                    status[i, k, 0]
                    and coords[i, k, 0] < self.width
                    and coords[i, k, 1] < self.height
                    and aggregated_mask[coords[i, k, 1], coords[i, k, 0]] == 0
                ):
                    missed_keypoints[i] += 1
        # percentage of detected object
        percentage_detected = detected_objects / num_objects
        self.percentage_detected.append(percentage_detected.mean())
        # percentage of tracked object
        percentage_tracked = tracked_objects / num_objects
        self.percentage_tracked.append(percentage_tracked.mean())
        # average number of missed keypoints
        self.missed_keypoints.append(missed_keypoints.mean())
        # average density over the time
        self.redundant_keypoints.append(redundant.mean())
        # print(self.redundant_keypoints)

    def save_to_file(self, model_name=None):
        # create a folder to store the dataset
        os.makedirs("results", exist_ok=True)
        file_path = "results/{0}.xlsx".format(model_name)
        writer = pd.ExcelWriter(file_path, engine="openpyxl")
        data_frame = pd.DataFrame(
            {
                "DOP - Detected objects percentage (higher is better)": self.percentage_detected,
                "TOP - Tracked objects percentage (higher is better)": self.percentage_tracked,
                "UAK - Unsuccessful keypoint assignment (lower is better)": self.missed_keypoints,
                "RAK - Redundant keypoint assignment (lower is better)": self.redundant_keypoints,
            }
        )
        table = wandb.Table(dataframe=data_frame)
        wandb.log({"Results": table})
        data_frame.to_excel(writer, sheet_name="per-video results", index=False)
        result_stats = data_frame.describe()
        table = wandb.Table(dataframe=result_stats)
        wandb.log({"Statistics over all videos": table})
        result_stats.to_excel(writer, sheet_name="Statistics")
        writer.save()
        writer.close()
        return data_frame
        