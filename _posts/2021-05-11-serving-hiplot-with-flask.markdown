---

title: "Serving HiPlot with Flask"
layout: post
date: 2021-02-13 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- bash
star: false
category: blog
author: yotam
description: Serving HiPlot with Flask
---

# Serving HiPlot with Flask

```python
pandas==0.24.2
hiplot==0.1.12
Flask==1.1.1

import hiplot as hip
import pandas as pd
from flask import Flask

app = Flask(__name__)

@app.route('/')
def run_hiplot():
	hiplot_exp = hip.Experiment.from_dataframe(self.df)
    return hiplot_exp.to_html()

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=6006)

```

