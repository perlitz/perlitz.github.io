---

title: "Build a package from your Python project"
layout: post
date: 2020-01-29 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
<<<<<<< HEAD:_posts/2020-02-01-package-building-in-python.markdown
- python
star: false
category: blog
author: yotam
description:  Build a package from your Python project

---

# Build a package from your Python project

This explanation ignores the existence of test and package testing.
=======
- site building
star: false
category: blog
author: yotam
description: Build a package from your Python project
---

# Build a package from your own Python project
>>>>>>> 088773c39e8daca56422ae98892bf78c6e3cca22:_posts/2020-01-29-package-building-in-python.markdown

## Structure

First thing to do is to get your directory in order,  the structure of your python package directory should take the following form:

```
reponame
	├─ packagename
	│  ├─ __init__.py
	│  ├─ ...
  |   └─ ...
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
<<<<<<< HEAD:_posts/2020-02-01-package-building-in-python.markdown
      			 packages=setuptools.find_packages(),
=======
      			                  packages=setuptools.find_packages(),
                 include_package_data=True,
>>>>>>> 088773c39e8daca56422ae98892bf78c6e3cca22:_posts/2020-01-29-package-building-in-python.markdown
      )
```

The first five fields need no explanation, just keep the package name simple without any '-' of '_' since they may cause problems up ahead, if you find this interesting, check [this](https://packaging.python.org/specifications/core-metadata/#name) out.

`packages=setuptools.find_packages()` will import every directory that has an `__init__.py` file down from the one setup.py is at, if you wish to import packages from different locations, you can use the `where=` argument in find packages to find packages in specific locations..


## Installation

### From local repository 

Once the package is well structured, imports are fixed and `setup.py` is prepared, all that is left if to cd to the location of `setup.py` and hit `pip install -e .` this will install the package in develop mode so that you can change the code, if you do not want to change the code, drop the `-e` and the package will be installed in your virtual environment. 

## Import issues

In case import issues are coming your way, check out [this](https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html) article which is great, in case you are still having problems [this](https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder/11158224#11158224) might also be of help.