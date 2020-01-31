---
title: "Pandas Tutorial"
layout: post
date: 2019-12-13 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- pandas
star: false
category: blog
author: yotam
description: A simple Pandas tutorial
---

This is a small summary of pandas commands, this is where I keep my pandas snippets for a case of need.

In case a deeper dive in to the subject is wanted, don't hasitate to check out this [much better tutorial](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html). Let's jump in:

Import pandas

```python
import pandas as pd
```

Write/Read DF as .h5:

```python
my_df.to_hdf(path_to_folder+'/my_df_saving_name.h5', 'my_df_saving_name')
my_df = pd.read_hdf(path_to_folder+'/my_df_saving_name.h5')
```

Write/Read DF as .json:

``` python
my_df.to_json(path_or_buf=path_to_folder+'/my_df_saving_name.json')
my_df = pd.to_json(path_or_buf=path_to_folder+'/my_df_saving_name.json')
```

Add a row to DF (note the use of ` ignore_index=True` which tells the DataFrame to set the index as row enumeration, as in a simple list).

```python
my_df = my_df.append({'dir_name': dir_name, 'frame_name': frame_name,'false_positive': num_fp}, ignore_index=True)
```

Add a new column to DF:

```python
my_df['false_negative'] = pd.Series(false_negative_list)

# note that len(false_negative_list) has to be equal to my_df.shape[0]
```

Access a certain column of the DF:

```python
my_series = my_df['dir_name']
my_series = my_df.dir_name
my_series = my_df.loc[:,'dir_name']
```

A [series can be turned in to a list](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_numpy.html#pandas.DataFrame.to_numpy) using:

``` python
nparray = series.to_numpy()
```

Create [iterator](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html) of rows of DF:

```python
df_iterator = my_df.iterrows()
```

[Collapse rows](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) and apply manipulation over duplicates:

```python
my_df = my_df.groupby('dir_name').agg({'false_negative':'mean','false_positive':'mean'})

# note that all coulmn values must be numeric, in case not true (for example for false_positive), can use: my_df['false_positive'] =  pd.to_numeric(my_df['false_positive'])
```

[Drop duplicates](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html) (taking only the first value):

```python
my_df_row_per_dir = my_df.drop_duplicates(subset=['dir_name'],keep='first')
```

[Drop columns](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)

```python
my_df.drop(columns=['false_negative'])
```

Keep certain values of DF:

```python
my_df_zero_fp = my_df[my_df['false_positive']==0]
```

[Count occurrences](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html) in a certain column:

``` python
my_df['false_positive'].value_counts()

# this will produce a table in which each row holds a value (of 'false_positive') and the number of occurences, for example:

false positive value  -  number of occurences
        5    				   1218837
        1  			       	288189
        10   				    167364
        6   				    118085
        17  				    68663
        14  				    47808
        18  				    45225
```

Sort DF by column:

``` python
my_df.sort_values(by='false_positive',ascending=False)
```

View top / bottom:

``` python
my_df.head(3)
my_df.bottom(3)
```

Running over rows of the DF

Sometimes we wish to run over all objects in a specific DF, here is an example of working through

```python
for df_row_num in df.iterrows():
    df_row = df.loc[df_row_num]

    ~do something using row as a one liner DF
```

### Small helpers for saving metadata along with DF to .h5

```python
def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5load(store):
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata

metadata = {'City':'Tel-Aviv'}
h5store(filename_to_save, my_df, **metadata)

with pd.HDFStore(filename) as store:
    data, metadata = h5load(store)
```

