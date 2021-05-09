---

title: "Python snippets"
layout: post
date: 2021-02-13 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- bash
star: false
category: blog
author: yotam
description:  Useful python pieces of code 
---

# Python snippets

### Colored print

```python
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")
```

### Boolean and iterable in argparse

```python
parser.add_argument('--eval_epochs', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--debug', action='store_true', default=False)
```



