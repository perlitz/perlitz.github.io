---

title: " Key modules for experiment handling"
layout: post
date: 2021-02-13 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- bash
star: false
category: blog
author: yotam
description: Key modules for experiment handling
---

# Key modules for experiment handling

Each new experiment framework constructed requires a few basics:

1. Config management and documentation
2. stdout saving
3. Source code saving

Below is an example for a `main()` function with all the above taken care of using functions implemented in `acv_utils` (but also display their full implementation here). 

After the example, the sections describe each of the parts in full including implementation.

[TOC]

### Example: `main()` function

```python
from acvutils.logging.copy_source_files import copy_source_files
from acvutils.logging.stdout_logger import StdoutLogger
from os.path import join as pjoin
from acvutils.yacs import yaml_utils
from config.default_config import get_cfg_defaults

def main(argv):
    # Get cfg_path from argparse and load default config from default_config 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, required=True, help='path to config file')
    args = parser.parse_args(argv)
    config = get_cfg_defaults()
    
    # In a config path is passed, run over default config
    if args.cfg_path != '':
        yaml_utils.load_from_yaml(args.cfg_path, config)
    
    # Set exp_dir according to the config_path (same name as config path removing '_cfg')
    config.PATHS.EXP_NAME = args.cfg_path.split('/')[-1].split('.')[0].replace('_cfg', '')
    config.PATHS.EXP_DIR = pjoin(config.PATHS.RESULTS_DIR, config.PATHS.EXP_NAME)

    # Save stdout, sourcefiles and config.
    sys.stdout = StdoutLogger(config.PATHS.EXP_DIR)
    copy_source_files(config.PATHS.EXP_DIR)
    yaml_utils.save_full_config(config, pjoin(config.PATHS.EXP_DIR, 'exp_cfg.yaml'))
    
    # Do whatever you came here to do, in this case, train.
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main(sys.argv[1:])
```

### What you will need

Clone `acv_utils`:

```bash
git clone git@gitlab-srv:SOC_CV_Algorithms/acv-utils.git
```

Install it:

```bash
pip install -e /path/to/where/acv_utils/is/acv_utils/.
```

Install `yacs` and `mpi4py`:

```
pip install yacs mpi4py
```

### Save all Source code

#### HowToUse

To document all source files, in `main()`, use:

```python
from acvutils.logging.copy_source_files import copy_source_files
copy_source_files(exp_dir)
```

To copy all python files to a directory called `sources` in `exp_dir`.

#### Definition

`copy_source_files` looks for all python files in the CWD and copies them so `\sources` dir in the experiment dir, the definition:

```python
import os
from shutil import copyfile
from pathlib import Path

def copy_source_files(exp_path):

    path_to_save_dir = pjoin(exp_path ,'sourcecode')
    sourece_files_dir = os.getcwd()
    os.makedirs(path_to_save_dir, exist_ok=True)
    py_files_paths = [str(path) for path in Path(sourece_files_dir).rglob('*.py') if not str(path).startswith('__init__.py')]
    for file_path_to_copy in py_files_paths:
        copyfile(file_path_to_copy, '/'.join([path_to_save_dir, file_path_to_copy.split('/')[-1]]))
```

### Save stdout

#### HowToUse

To save all the stdout, in main()`, use:

```python
from acvutils.logging.stdout_logger import StdoutLogger
sys.stdout = StdoutLogger(exp_dir)
```

#### Definition

The class is implemented as:

```python
import sys
class StdoutLogger(object):
    def __init__(self, log_dir_path):
        self.terminal = sys.stdout
        self.log_dir_path = log_dir_path

    def write(self, message):
        with open(self.log_dir_path + '/stdout.txt', "a", encoding='utf-8') as self.log:
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
```

### Manage your config with YACS

#### HowToUse

to manage and save your config, in `main()`, after inputting the cfg_path to argparser use:

```python
from os.path import join as pjoin
from acvutils.yacs import yaml_utils
from config.default_config import get_cfg_defaults

config = get_cfg_defaults()
if args.cfg_path != '':
    yaml_utils.load_from_yaml(args.cfg_path, config)
    config.PATHS.EXP_NAME = args.cfg_path.split('/')[-1].split('.')[0].replace('_cfg', '')

config.PATHS.EXP_DIR = pjoin(config.PATHS.RESULTS_DIR, config.PATHS.EXP_NAME)

yaml_utils.save_full_config(config, pjoin(config.PATHS.EXP_DIR, 'exp_cfg.yaml'))

```

You will have to have a `config.default_config.py`, it should look something like:

```python
from yacs.config import CfgNode as CN

def get_default_params():
    default_params = dict()

    default_params['PATHS'] = {

            'DATAROOT'   : '',
            'EXP_NAME'   : 'temp_exp',
            'RESULTS_DIR': 'output',
            'EXP_DIR': '',
    }

    default_params['TRAIN'] = {

            'START_EPOCH': 0,
            'N_EPOCHS'   : 200,  # number of epochs of training
            'BATCH_SIZE' : 1,
            'LR'         : 0.0002,
    }
    
    default_params['VALID'] = {

            'BATCH_SIZE' : 1,
    }

    return default_params

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return CN(get_default_params()).clone()
```

With more `default_params` keys like `PATHS` can be further added and more `key:value` to be added to each.

In `cfg_path` you should have a `.yaml` file that conforms to the structure in config.default_config.py, something like:

```yaml
PATHS:
    DATAROOT: /home/yotampe/Code/cyclegan_slim/datasets/gta2cityscapes
    RESULTS_DIR: /raid/algo/SOCISP_SLOW/ADAS/DA/results/slim

TRAIN:
    BATCH_SIZE: 4
    
VALID:
	BATCH_SIZE: 1
```

Where entries left absent remains in their default.

#### Definition

`yaml_utils` is implemented in `acv_utils` as:

```python
import yaml
from yacs.config import CfgNode as CN

def dict_from_yaml(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg_dict

def load_from_yaml(path, yacs_cfg):
    cfg_dict = dict_from_yaml(path)
    tmp = CN(cfg_dict)
    yacs_cfg.merge_from_other_cfg(tmp)

def save_full_config(yacs_cfg: CN, path_to_saved_file):
    with open(path_to_saved_file, 'w+') as f:
        print(yacs_cfg, file=f)  # Python 3.x\
    print(yacs_cfg)
```



### Log your experiment with logx

First, initialize the logger with:

```python
from acvutils.logging import logx
logger = logx.EpochLogger(exp_dir, 'logger')
```

Add line to the logger with `.log_tabular()` as in:

```python
logger.log_tabular('Epoch', epoch)
logger.log_tabular('G_LR', self.optim_G.param_groups[0]['lr'])
```

and dump once on a while to the file with:

```python
logger.dump_tabular()
```

This will create a file named `logger` in your `exp_dir`, read it later using pandas as a `DataFrame` with:

```python
df = pd.read_csv(log_file_path, sep="\t")
```

