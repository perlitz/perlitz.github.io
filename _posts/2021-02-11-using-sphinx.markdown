---

title: "Using Sphinx"
layout: post
date: 2021-02-13 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- bash
star: false
category: blog
author: yotam
description:  Some lessons learned when using sphinx
---

# Using Sphinx for auto-documenting code

### Installation and preparation

Install

```bash
pip install sphinx sphinx_rtd_theme
```

Create a docs directory inside the project 

> Warning: The project directory I am referring to is where the code is, not where the say setup.py file is.

```bash
cd PROJECT_DIR
makedir docs
cd docs
sphinx-quickstart
```

Quickstart, and answer the questions in:

```bash
sphinx-quickstart
```

### Customizing

Add .rst type documentation to the code

edit docs/index.rst to the following:

```rst
.. bpcpt documentation master file, created by
   sphinx-quickstart on sun may  9 06:33:15 2021.
   you can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

welcome to PROJECT documentation!
=================================

.. toctree::
   :maxdepth: 2

   modules

indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

edit docs/conf.py to the following:

```python
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'PROJECT'
copyright = '2021, Yotam Perlitz'
author = 'Yotam Perlitz'

# The full version, including alpha/beta/rc tags
release = '0.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.viewcode',
]

autodoc_typehints = "description"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
```

### Building

go to the docs directory and go:

```bash
sphinx-apidoc -o . ..
```

This will generate a .rst file for all modules in your code, lastly, go to docs and hit:

```bash
make html 
```

that will create `html`s in the `_build` directory.

> In case your modules are not identified, try to append them to your python path
>
> ```bash
> export PYTHONPATH="${PYTHONPATH}:/path/to/project" &&  make html
> ```