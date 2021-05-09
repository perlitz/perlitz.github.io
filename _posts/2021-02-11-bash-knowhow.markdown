---

title: "Bash Knowhow"
layout: post
date: 2021-02-13 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- bash
star: false
category: blog
author: yotam
description:  Bash Knowhow
---

# Bash Knowhow

### Symbolic link

```bash
ln -s {/path/to/file-name} {link-name}

cd />path/to/where/links/will/go
find /where/files/are/linked/from -type f -name "*.png" -exec ln -s '{}' . ';' -maxdepth 3
```

### Copy many files somewhere

Take all files that end with something (in this case it is .xls) and copy it to the target directory.

```bash
find ./ -name '*.xsls' -exec cp -prv '{}' '/path/to/targetDir/' ';'
```

### Find and delete by type

```basg
$ find . -name ".RAW" -delete
```

### WGET from google drive

```bash
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1-O7vWiMa3mDNFXUoYxE3vkKZQpiDXUCf' -O train_imgs.zip
```

### Find CUDA driver

```bash
/usr/local/cuda/bin/nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

### Add postfix

```bash
$ for i in *.png; do mv "$i" "${i%.*}_w300.png"; done 
```

