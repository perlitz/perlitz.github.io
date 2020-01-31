# Build a package from your Python project

This explanation ignores the existence of test and package testing.

## Structure

First thing to do is to get your directory in order,  the structure of your python package directory should take the following form:

```
├─ packagename
│  ├─ __init__.py
│  ├─ ...
|  └─ ...
│     └─ ...
│  └─ tests
│     └─ ...
├─ .git
├─ .gitignore
└─ setup.py
```

note that every directory must have an `__init__.py` file in order for it to be considered a module.

Ever wondered what is the `__init__.py` file for? see [this](https://stackoverflow.com/questions/448271/what-is-init-py-for).

**Note!** The best way to move files around is using pycharm that will take care of the imports.

## The `Setup.py` file

Next thing is to create a `setup.py` file which describes all the metadata regarding the package, this file generally has many option, here, I will show the minimum working version (for a longer description of this file visit [here](https://github.com/pypa/sampleproject/blob/master/setup.py) and [here](https://blog.godatadriven.com/setup-py)). My version of `setup.py` is the following:

```
import setuptools
from glob import glob
from os.path import basename, splitext

setuptools.setup(name='packagename',
                 version='0.01',
      			 description='description',
      			 author='Yotam Perlitz',
      			 author_email='yotam.pe@samsung.com',
      			                  packages=setuptools.find_packages(),
                 include_package_data=True,
                 py_module=[splitext(basename(path))[0] for path in 		glob('**/*.py',recursive=True)],
      )
```

The first five fields need no explanation, just keep the package name simple without any '-' of '_' since they may cause problems up ahead, if you find this interesting, check [this](https://packaging.python.org/specifications/core-metadata/#name) out.

All the rest will make sure all files are properly imported.

## Installation

### From local repository 

Once the package is well structured, imports are fixed and `setup.py` is prepared, all that is left if to cd to the location of `setup.py` and hit `pip install -e .` this will install the package in develop mode so that you can change the code, if you do not want to change the code, drop the `-e` and the package will be installed in your virtual environment. 

### From GitLab

Use the command:

```
pip install -e git+http://git@gitlab-srv/repoadress@branch#egg=packagename
```

The `-e` means you can still change the content of the package while it is installed.

If a specific tag is wanted, `@v2.1.0` can be also added for example as a tag

Done forget to:

```bash
export {http,https,ftp}_proxy=http://dlp2-wcg01:8080 && export {HTTP,HTTPS}_PROXY=http://dlp2-wcg01:8080 && conda config --set ssl_verify false
```

before...

## Import issues

In case import issues are coming your way, check out [this](https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html) article which is great, in case you are still having problems [this](https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder/11158224#11158224) might also be of help.