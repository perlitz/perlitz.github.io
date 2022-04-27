---
title: "Using Selenium to capture images"
layout: post
date: 2022-01-26 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- NLP
star: false
category: blog
author: yotam
description: Using Selenium to capture images
---
# Using Selenium to capture images

When scraping a dataset from the web it is sometimes needed to also capture some visual content as well, in this snippet I present a way to do that.

```python
import time

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


import os
import re
from shutil import copyfile
from selenium import webdriver
from io import BytesIO
from time import sleep

import base64
import pandas as pd
from tqdm.notebook import tqdm

DRIVER_LOCATION = '.chromedriver'
DATASET_PATH = './statista/datast_len_100586_free.csv'
N_STATISTICS_TO_PROCESS = 9999999999
OUT_PATH = './statista/captures_all'

options = Options()
options.add_argument('--disable-notifications')
options.add_argument("--headless")
options.add_argument("--window-size=3072,1920")
options.add_argument("--disable-infobars")
options.add_argument("--disable-extensions")
options.add_argument("--disable-gpu")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")

browser = webdriver.Chrome(DRIVER_LOCATION, chrome_options=options)

dataset = pd.read_csv(DATASET_PATH)     
for url in tqdm(dataset['url'].to_list()[:N_STATISTICS_TO_PROCESS]):

    num = url.split('/')[4]
    
    if f'{num}.png' not in os.listdir(OUT_PATH):     
        
        cleanUrl = url
        browser.get(cleanUrl)
        try:
            
            # take care of cookies button
            try:
                browser.find_element(By.ID,'onetrust-accept-btn-handler').click()
                sleep(1)
            except:
                pass
            
            # take care of "expand statistic button"
            try: 
                browser.find_element(By.XPATH,'//*[@id="statisticContainer"]/div[1]/article/div/div/div[2]/div[2]').click()
                with open(f'{OUT_PATH}/collapsed.txt','a') as f:
                    f.write(f'\n{num}')
            except:
                pass

            # based on https://stackoverflow.com/a/55676925
            element = browser.find_element(By.CLASS_NAME,'highchart-container') 

            # center the figure
            desired_y = element.location['y'] + (element.size['height'] / 2) 
            window_h = browser.execute_script('return window.innerHeight')
            window_y = browser.execute_script('return window.pageYOffset')
            current_y = (window_h / 2) + window_y
            scroll_y_by = desired_y - current_y
            browser.execute_script("window.scrollBy(0, arguments[0]);", scroll_y_by)

            sleep(1.2) # this is how long it takes for the figure animations to load
            screenshot_as_bytes = element.screenshot_as_png

            with open(f'{OUT_PATH}/{num}.png', 'wb') as f:
                f.write(screenshot_as_bytes)

        except Exception as ex:
            print(ex)

browser.quit()
```



