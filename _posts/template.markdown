---

title: "Interacting with the Wayback machine"
layout: post
date: 2020-01-29 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- python
star: false
category: blog
author: yotam
description:  Mounting an ExFAT external drive on MAC

---

# Mounting an ExFAT external drive on MAC

based on [this](https://apple.stackexchange.com/a/378257).

If your exfat drive shows up as read-only, then all you need to do is unmount the drive and then use the "mount_exfat" utility to mount the drive. Once you do this, and right click on the drive → "Get Info" will tell that you have custom access to the drive, instead of "read-only". Here are the steps:

1. Open the terminal on your mac.

2. Type `diskutil list` and you will get a listing like this:

   ```
   /dev/disk0 (internal, physical):    #:                       TYPE NAME SIZE       IDENTIFIER    0:      GUID_partition_scheme                
   *500.3 GB   disk0    1:                        EFI EFI                     209.7 MB   disk0s1    2:                 Apple_APFS Container disk1         500.1 GB   disk0s2
   
   /dev/disk1 (synthesized):    #:                       TYPE NAME        SIZE       IDENTIFIER    0:      APFS Container Scheme -              
   +500.1 GB   disk1
                                Physical Store disk0s2    1:                APFS Volume Untitled - Data         407.2 GB   disk1s1    2:           APFS Volume Preboot                 82.4 MB    disk1s2    3:           APFS Volume Recovery                528.5 MB   disk1s3    4:           APFS Volume VM                      3.2 GB     disk1s4    5:           APFS Volume Untitled                10.7 GB    disk1s5
   
   /dev/disk2 (external, physical):    #:                       TYPE NAME SIZE       IDENTIFIER    0:      GUID_partition_scheme                
   *5.0 TB     disk2    1:                        EFI EFI                     209.7 MB   disk2s1    2:       Microsoft Basic Data Backup Plus             5.0 TB     disk2s2
   ```

3. In the listing above, we notice that `/dev/disk2s2 is our external drive. Unmount the drive by running`sudo umount /dev/disk2s2`.

4. Now create a directory in in /Volumes by running `sudo mkdir -p /Volumes/<name of your volume>`

5. Mount the hard drive to this directory your created by running `sudo mount_exfat /dev/disk2s2 /Volumes/<name of your volume>`.

Once this is done, you should be able to create new folders and write to your drive.
