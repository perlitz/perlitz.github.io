---

title: "Implement an Image-to_Image project from scratch"
layout: post
date: 2021-02-13 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- bash
star: false
category: blog
author: yotam
description:  Lessons learned while implementing an Image-to_Image project from scratch
---

# Lessons learned while implementing an Image-to-Image project from scratch

[TOC]

## Listen to [Karpathy](http://karpathy.github.io/)

Follow this [great blog post](http://karpathy.github.io/2019/04/25/recipe/) were Karpathy goes through the stages to construct the project.

## Think Fast

### CUDA optimization

Follow [Pytorch`s own performance tuning guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) or TFs [gpu performance analysis](https://www.tensorflow.org/guide/gpu_performance_analysis) and keep GPU utilization high, this will save a lot of time in the experimentation phase later

### Faster data-loading

In many cases when training a NN the main bottleneck turns out to be the CPU and I\O instead of the GPU. This effect is usually cause by inefficient data loading this, since other parts of the training cycle is usually rather standard and comes optimized own of the box by the DL frameworks.

#### How to notice inefficient GPU usage

Use `watch -n 0.1 nvidia-smi` while training is going, if `GPU-Util` is below say 80% you should probably be improving your data-loading.

#### Things to look into when trying to improve your data-loading.

##### Preprocessing 

If you are for example, load 2K images but end up using $256^2$ crop you are wasting a lot of time *I\Oing* pixels you don't need, consider pre-cropping your dataset and loading crops to save time.

##### Caching

In case your dataset is small enough to fit into you machine`s RAM (should be $\ge$32GB on modern machines), consider caching the entire dataset and training from RAM. 

An example for a caching dataset can be seen below. By default the datasets starts by caching the data  it gets in `__getitem__`, once the first epoch is done, `dataset.set_cache_status(phase='use')` should be called to start using the the cached dataset.

```python
import glob
import os
import pickle
import random

from torch.utils.data import Dataset


class SIDD(Dataset):

    def __init__(self, data_dir, transform, phase, shuffle, gt_type='GT', light_mode=''):
        self.transform = transform
        self.data_dir = os.path.join(data_dir, phase)
        self.shuffle = shuffle

        self.gt_list = [path for path in
                        glob.glob(os.path.join(self.data_dir, f'*{light_mode}', f'*_{gt_type}_*[0-9].pkl'))]
        self.bp_list = [path for path in
                        glob.glob(os.path.join(self.data_dir, f'*{light_mode}', '*[0-9]_BP_*[0-9].pkl'))]
        

        assert len(self.target_list) > 0, 'Dataset specification in erroneous, images were not found'

        self.random_indices = []
        self.cached_gt = []
        self.cached_bp = []
      
        self.cache_phase = 'fill'

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, index):

        if self.cache_phase == 'fill':
            gt = self.load_pickle(self.gt_list[index])
            self.cached_gt.append(gt)

            bp = self.load_pickle(self.bp_list[index])
            self.cached_bp.append(bp)

            self.random_indices.append(index)

        elif self.cache_phase == 'use':
            index = self.random_indices[index]

            gt = self.cached_gt[index]
            bp = self.cached_bp[index]

        else:
            gt = self.load_pickle(self.gt_list[index])
            bp = self.load_pickle(self.bp_list[index])

        gt, bp = self.transform([gt, bp])

        return gt, bp

    def set_cache_status(self, phase):
        if phase == 'use':

            self.cache_phase = phase
            if self.shuffle:
                random.shuffle(self.random_indices)

        elif phase == 'restart':
            del self.cached_gt
            del self.cached_bp

            self.cached_gt = []
            self.cached_bp = []

            self.cache_phase = 'fill'

        if phase == 'no_cache':
            self.cache_phase = 'no_cache'

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
```

Make sure to make `num_workers=0` in the `dataloader` since caching will not succeed if using multiprocessing as there will be many dataset instances running in parallel. 

## Think Reproducible

Use a combination of [YACS](https://github.com/rbgirshick/yacs) and some HPO package and plug in your parameters as iterables from the start, for example, instead of defining your loss function as a `float` define it as a list of floats  this way, experimenting can be made automatic. 

### Train from commit

Training from commits will make sure you will always have the code you trained with at hand in case you want to, along with the saved configuration from YACS, it holds the key to experiment reproducibility. Use this script to run your experiments:

```python
import os
import sys
import argparse

import git

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_url", type=str, default='',
                        help="URL of the git repository. This is the same URL that is used for cloning.")
    parser.add_argument("--branch_name", type=str, required=True,
                        help="The branch that will be checked out before running.")
    parser.add_argument("--commit_hash", type=str, default=None,
                        help="The commit that we want to run. If not specified, HEAD will be used. Note that this commit has to exist in the specified branch.")
    parser.add_argument("--cfg_path", type=str,
                        help="The config file for the script we run.")
    parser.add_argument("--cfg_path_arg_name", type=str, required=True,
                        help="The name of the command-line argument for ``cfg_path`` in the script we run.")
    parser.add_argument("--copies_base_path", type=str, required=True,
                        help="The directory that will hold all the repository clones.")
    parser.add_argument("--module_path", type=str, required=True,
                        help="The path of the module that we want to run relatively to the repository root. E.g. ``training/train.py``")

    return parser.parse_args()


def main():
    args = parse_args()

    repo_name = args.repo_url.strip('.git').split('/')[-1]

    # Path of clone from which we run. We append the commit hash to the clone
    # directory name, such that we create a new clone only when we run from a
    # new commit.
    repo_dst_path = os.path.join(args.copies_base_path, repo_name + f"_{args.commit_hash}")
    print(repo_dst_path)

    # Get repository.
    if os.path.exists(repo_dst_path):
        print("Repository clone already exists.")
        repo = git.Repo(repo_dst_path)
    else:
        print("Repository clone does not exist.")
        print("Cloning ... ")
        repo = git.Repo.clone_from(args.repo_url, to_path=repo_dst_path)

    # Checkout to required commit.
    repo.git.checkout(args.branch_name)
    if args.commit_hash is not None:
        repo.git.checkout(args.commit_hash)

    os.environ['PYTHONPATH'] = repo_dst_path

    print("Running ... ")
    os.system(f"{sys.executable} {os.path.join(repo_dst_path, args.module_path)} --{args.cfg_path_arg_name}={args.cfg_path}")


if __name__ == "__main__":
    main()

```

Example for an .sh file for running an experiment:

```bash
#! /bin/bash

/miniconda/envs/bpcpt/bin/python \
.../train_from_clone_script.py \
	--repo_url=.../BPCPT.git \
	--branch_name="develop" \
	--commit_hash="ba969b243cb1cd59cba1bcb02927243bc93a25ec" \
	--cfg_path="/home/yotampe/Code/bpcpt/config_files/051.yaml" \
	--cfg_path_arg_name="cfg_path" \
	--copies_base_path="/home/yotampe/Code/repo_copies" \
	--module_path="bpcpt/train.py"
```

### Use [YACS](https://github.com/rbgirshick/yacs) configuration

Use YACS with the following two functions:

```python
def initialize_run(argv, phase='train'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, required=True, help='path to cfg file')
    args = parser.parse_args(argv)
    cfg = get_cfg_defaults()

    if args.cfg_path != '':
        yaml_utils.load_from_yaml(args.cfg_path, cfg)

    cfg.PATHS.EXP_DIR = pjoin(cfg.PATHS.RESULTS_DIR, cfg.PATHS.EXP_NAME)

    if phase == 'train':
        sys.stdout = StdoutLogger(cfg.PATHS.EXP_DIR)
        yaml_utils.save_full_config(cfg, pjoin(cfg.PATHS.EXP_DIR, f'{cfg.PATHS.EXP_NAME}.yaml'))
    return cfg

```

and:

```python
import yaml
from yacs.config import CfgNode as CN
import os

def dict_from_yaml(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg_dict

def load_from_yaml(path, yacs_cfg):
    cfg_dict = dict_from_yaml(path)
    yacs_cfg.merge_from_other_cfg(CN(cfg_dict))

def save_full_config(yacs_cfg: CN, path_to_saved_file):
    os.makedirs(os.path.dirname(path_to_saved_file), exist_ok=True)
    with open(path_to_saved_file, 'w+') as f:
        print(yacs_cfg, file=f)  # Python 3.x\
    print(yacs_cfg)
```

Create a default config, for example:

```python
from yacs.config import CfgNode as CN


def get_default_params():
    default_params = dict()

    default_params['PATHS'] = {

            'EXP_NAME'   : 'temp_exp',
            'RESULTS_DIR': 'output',
            'EXP_DIR'    : '',
    }

    default_params['MODEL'] = {

            'N_CH_IN'     : 1,
            'N_CH_OUT'    : 1,
            'ARCHITECTURE': {
                    'N_BLOCKS'        : 2,
                    'N_FEATURES'      : 6,
                    'EXPANSION_FACTOR': 3,
                    'FULL_RESIDUAL'   : True,
            },
            'RES_SCALE'   : 1,
            'USE_BN'      : False,
    }

    default_params['DATA'] = {

            'DATA_DIR'       : '',
            'SHUFFLE'        : True,
            'BATCH_SIZE'     : 1,
            'N_CPU'          : 0,  # number of cpu threads to use during batch generation
            'NORM_MEAN'      : 0.0,
            'NORM_STD'       : 1.0,
            'SIDD_MEAN'      : 0.0,
            'SIDD_STD'       : 1.0,
            'CROP_SIZE'      : (256, 256),
            'H_FLIP_PROB'    : 0.5,
            'V_FLIP_PROB'    : 0.5,
            'NON_RANDOM_CROP': False,
            'GT_TYPE'        : 'GT',
            'LIGHT_MODE'     : '',
    }

    default_params['TRAIN'] = {

            'EXP_DIR'             : '',
            'PRETRAINED_CKPT_PATH': '',
            'START_EPOCH'         : 0,
            'N_EPOCHS'            : 200,  # number of epochs of training
            'IGNORE_CKPT_METADATA': False,

            'OPTIMIZER'           : {'OPTIMIZER'   : 'ADAM',
                                     'LR'          : 0.001,
                                     'WEIGHT_DECAY': 0,
                                     'EPS'         : 10 ** (-8),
                                     'MOMENTUM'    : 0.9,
                                     'BETAS'       : (0.9, 0.99),
                                     'DECAY'       : '100-200',
                                     'GAMMA'       : 0.5,
                                     'SCHEDULER'   : None,
                                     },

            'LOSS'                : [{'name': 'L1', 'weight': 1.0}],
            'TEST_FREQ'           : 1,
            'DECAY_EPOCH'         : 100,
            'CROP_SIZE'           : 256,
            'SAVE_CKPT_FREQ'      : 100,

    }

    default_params['TEST'] = {

            'LOSS': [{'name': 'L1', 'weight': 1.0}],

    }

    default_params['HPO'] = {'USE_HPO': False,
                             'TRAIN'  : {'OPTIMIZER': {'OPTIMIZER': [], 'LR': []}},
                             'DATA'   : {'BATCH_SIZE': [], 'LIGHT_MODE':[]},
                             'MODEL'  : {'ARCHITECTURE': []}}


    return default_params


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return CN(get_default_params()).clone()
```

### Use a logger, let the Loss manage it

Get a logger ([for example this on from OpenAI](https://github.com/openai/spinningup/tree/master/spinup/utils)) and use it to log your training sessions, save it in your experiment directory. Let the logger be a member of the Loss class in the following manner:

```python
from utils import logx
from torch.utils.tensorboard import SummaryWriter

class Loss(nn.Module)
	def __init__(self)
    	self.exp_dir = ...
		self.logger = logx.EpochLogger(self.exp_dir, 'logger')
        self.writer = SummaryWriter(log_dir=os.path.join(exp_dir,'tb'))

```

Once the logger is set up, one still has to keep it's own data-structure with the losses (in the case below it is `self.epoch_losses`)  but it can be overwritten every time `dump_logs`  is called so it will not explode once training becomes long enough.

```python
    def dump_logs(self):
        for phase in ['train', 'test']:
            for loss in self.losses:
                self.logger.log_tabular(f'{phase} {loss["name"]}', self.epoch_losses[phase][loss['name']])
                self.writer.add_scalars('', {f'{phase}_{loss["name"]}': self.epoch_losses[phase][loss['name']]}, self.epoch_n)

        self.logger.log_tabular('Epoch', self.epoch_n)
        self.logger.dump_tabular()
        self.writer.flush()
```

## Batch Analysis

Use the `dataloader` and configs you saved in the experiment config to create an `analysis` script that runs over a set of training sessions and outputs one clear graph  

 



