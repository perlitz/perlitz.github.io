---

title: "Run parallel inference while training"
layout: post
date: 2021-02-13 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- bash
star: false
category: blog
author: yotam
description: Run parallel training while inference
---

# Run parallel inference while training

```python
infer_part = partial(infer, out_dir_path=cfg.EXP_DIR,model=model_copy.to('cpu'))
multiprocessing.Process(target=infer_part).start()
```

