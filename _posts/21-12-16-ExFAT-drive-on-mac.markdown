---

title: "Serving Files with Python's SimpleHTTPServer Module"
layout: post
date: 2020-01-29 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- python
star: false
category: blog
author: yotam
description:  Serving Files with Python's SimpleHTTPServer Module

---

# Serving Files with Python's SimpleHTTPServer Module

based on: https://apple.stackexchange.com/questions/354363/brand-new-external-hard-drive-formatted-exfat-is-mounted-read-only

1. Type

```bash
diskutil list
```

2. Find the drive, in this case it is `/dev/disk2s1`

   ![image-20211222220011701](21-12-16-ExFAT-drive-on-mac.assets/image-20211222220011701.png)

3. Unmount the drive with `umount /dev/disk2s1 `

4. Now create a directory in in /Volumes by running `sudo mkdir -p /Volumes/<name of your volume>`

5. Mount the hard drive to this directory your created by running `sudo mount_exfat /dev/disk2s1 /Volumes/<name of your volume>`.
