---
title: "Interacting with the Wayback Machine"
layout: post
date: 2022-01-26 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- NLP
star: false
category: blog
author: yotam
description: Utilities for Interacting with the Wayback Machine
---

```python
import requests
def get_saved_url(orig_url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.5 "
                                 "(KHTML, like Gecko) Chrome/19.0.1084.52 Safari/536.5"}
    wburl = "https://archive.org/wayback/available?url="+orig_url
    response = requests.get(wburl,headers=headers,verify=False)
    data = response.json()
    if data['archived_snapshots']:
        saved_url = data['archived_snapshots']['closest']['url']
    else:
        saved_url = None
    return saved_url
```

```python
def save_url(orig_url):
    s=requests.Session()
    savePage='https://web.archive.org/save/'
    s.get(savePage+orig_url)
```

