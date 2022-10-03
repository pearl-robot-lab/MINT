# MINT
Code for submission : An information-theoretic approach to unsupervised keypoint representation learning <br>
[Project website](https://sites.google.com/view/mint-kp/home)

The code structre:
```
--- scripts (scripts to train the keypoint detectors, you can configure the training from the scripts)
\-- dynamics (scripts to train the dynamic models)
\-- imitation (scripts to train the imitation agent)
\-- src --- downstream (the dynamics and imitation agents with utils)
        \-- baselines (the baseline agent (transporter and video structure) with utils)
        \-- mint --- agents (mint agent)
                 \-- data (collecting the dataset from video to train the agents)
                 \-- entropy_layer (pytorch CUDA-extension for the entropy layer)
                 \-- losses (mint loss)
                 \-- models (the keypoint model for mint)
                 \-- utils (trajectory visualizer and results collector for CLVRER dataset)
```

## Prepare environment and datasets

1. Prepare the environment:
```
conda env create --file mint.yml
sh prepare_env.sh
```

2. Download sample datasets, uncomment the dataset that you want to use (the order is the same as in the paper):
```
bash download_datasets.sh
```
You have to change the number of training videos, evaluation videos depending and the tasks (for MIME and SIMITATE) accordingly. 

## Additional videos from datasets
- You can download additional videos (Very big files) and then change the scripts accordingly:
1. From CLEVRER dataset:
```
bash download_clevrer.sh
```
2. From MIME data set by listing numbers [1-20]:
```
bash download_mime.sh 1 2 
```

3. From SIMITATE data set by listing their numbers [1-4]:
```
bash download_simitate.sh 1 2 
```

## Training keypoint detectors:
 - To train MINT keypoint detector on a specific '<dataset>':
 ```
conda activate mint
python scripts/mint_<dataset>.py
 ```
 - To train a baseline keypoint detector <method> (videostructure or transporter) on a specific dataset:
 ```
conda activate mint
python scripts/baseline_<method>_<dataset>.py
 ```
## Running the experiments:
### DOWNSTREAM TASK I: Object detection and tracking:
You have to train the three keypoint detectors on CLEVRER dataset:
```
conda activate mint
python scripts/mint_clevrer.py
python scripts/baseline_transporter_clevrer.py
python scripts/baseline_videostructure_clevrer.py
```
Each scripts will train the corresponding keypoint detector for 5 seeds.
The evaluations run automatically after the training.
We report the results to Weight and Biases, but also we save the results to excel files to the `results` folder
and the evaluation videos with the predicted keypoints to `videos` folder.
You can get the final statstics offline by running:
```
python get_final_results.py
```

### DOWNSTREAM TASK II: Learning dynamics:
We provide pre-trained models (best seed from the last experiment) for each keypoint detector.
You can train the dynamics models to evaluate the dynamics prediciton:
```
conda activate mint
python dynamics/train_mint.py
python dynamics/train_transporter.py
python dynamics/train_videostructre.py
```
Each scripts will train the corresponding dynamics model for 5 seeds.
The evaluations run automatically after the training.
We report the results to Weight and Biases, but also we save the results to excel files to the `results` folder
and the evaluation videos with multistep prediciton to `videos` folder.
You can get the final statstics offline by running:
```
python get_final_results.py
```

### DOWNSTREAM TASK III: Object discovery in realistic scenes:
You can train the keypoint detectors on MIME and SIMITATE datasets:
```
conda activate mint
python scripts/mint_<dataset>.py
python scripts/baseline_transporter_<dataset>.py
python scripts/baseline_videostructure_<dataset>.py
```
The evaluations run automatically after the training.
We save the evaluation videos with the keypoints to `videos` folder.

### DOWNSTREAM TASK IV: Imitation learning:
You can train mint on MAGICAL dataset by running:
```
conda activate mint
python dynamics/mint_magical.py
```
We provide to pretrained keypoint model to experiment with.
You can train the imitation agent for MINT and the CNN agnet:
```
conda activate mint
python imitation/train_mint.py
python imitation/train_cnn.py
```
Each scripts will train the corresponding imitation agent for 5 seeds.
The evaluations on 25 rollouts run automatically after the training.
We report the results to Weight and Biases, but also we save the results to excel files to the `results` folder
and the evaluation videos with multistep prediciton to `videos` folder.
You can get the final statstics offline by running:
```
    python get_final_results.py
```

## Miscellaneous
- If your python compliter can't find a module from mint package, add the following before the imports :
```
import sys
sys.path.append('src')
```




