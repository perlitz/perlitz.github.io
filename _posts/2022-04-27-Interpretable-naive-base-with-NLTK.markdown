---
title: "Interpretable Naive Bayes Classification and Feature Selection by information gain with NLTK"
layout: post
date: 2022-04-27 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- NLP
star: false
category: blog
author: yotam
description: Using naive Bayes for interpretable text classification
---

This little tutorial will describe how to use NLTKs Naive Bayes Classifier to solve a simple text classification task, this is generally the first step in addressing a new text classification task from me since (1) if it works well, it is the most lightweight option computationally, (2) It requires little data relative to transformer based LM like RoBERTa (3) it is much more interpretable, as we will show, one can exactly understand what is going on under the hood. Now, as always, we start with the data.

## Dataset

The dataset we will use here is extracted from [Logic2Text](https://arxiv.org/abs/2004.14579), it contained a pairs of (statements, statement-logical-action) where the logical action can be one of the following: [ordinal, superlative, majority, unique, aggregation, comparative, count], their explanation can be found in the paper.

| Sentence                                                     | Action |
| ------------------------------------------------------------ | ------ |
| there were twelve occasions where the length was sixty minutes . | 6      |
| round five was the only time that chris goodwin and andrew kirkaldy had the pole position . | 3      |
| the longest length was when the circuit was dorington park . | 1      |

**Explanation of actions:** The logic2text dataset is build of tables and statements, each statement is classified according to the type of inference action that is needed in order to arrive at the information in the statement, for example: 

1. The first sentence is classified as **count** since is shows that in a count of 12 of the table rows the length was sixty minutes .
2. The second sentence is classified as **unique** since chris goodwin and andrew kirkaldy had the pole position only in the row of round five.
3. The third sentence is classified as **superlative** since it gives the largest value of the table.

We will create a dictionary to map the classes to their corresponding indices.

```python
class2idx = {"ordinal": 0, "superlative": 1, "majority": 2, "unique": 3, "aggregation": 4, "comparative": 5, "count": 6}
idx2class = {value: key for key, value in class2idx.items()}
```

## Preprocessing

Our simple preprocessing will only include the removal of punctuations and appending all remaining words to a single string, there are generally many other preprocessing one could make (that may be beneficial for naive babes) but we try to keep it simple here.

```python
import nltk
import string
from nltk.tokenize import word_tokenize

def sent2bow(sent):
    tokens = word_tokenize(sent)
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in tokens if w!='']
    return words
  
```

Once we have the pre-process function ready, we can load and preprocess the datasets. ==fix loading the data==

```python
train_raw=pd.read_json(train_path)
test_raw =pd.read_json(test_path)

def df2bow(df):
    all_words, dataset = [], []
    for idx, row in df.iterrows():        
        word_list = sent2bow(row.sent)
        all_words.extend(word_list)
				dataset.append((word_list,idx2class[row.action]))
        
    all_words_freq = nltk.FreqDist(w.lower() for w in all_words)
    return dataset, all_words_freq
  
train, all_words_freq = make_bow(train_raw)
test, _               = make_bow(test_raw)
```

Given the list of all words (`all_words`), we can decide which feature to use for the classifier according to their frequency.

```python
n_features = 2000
word_features = list(all_words_freq)[:n_features]
```

Define a feature extractor and process the datasets to pairs of (features, class) 

```python
def features(document, word_features): 
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
  
train_set = [(features(d, word_features), c) for (d,c) in train]
test_set  = [(features(d, word_features), c) for (d,c) in test]
```

## Training

Once the sets are ready, we can create and train the NLTK classifier, and check its accuracy.

```python
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
# => 0.8864468864468864
```

We got a score of 0.886 with about 20sec of training, not bad :) 

## Interpretation

Now that we have a classifier that seems to work, the next step could be seeing what it learned, this could be valuable in order to make sure it didn’t exploit some weakness in the data and since it might make us better understand the data.

For this, we will extract the word count matrix from the classifier: 

```python
from collections import defaultdict

counts = pd.DataFrame(columns=['action'])
counts['action']=pd.Series(list(class2idx.keys()))

for (label, fname), probdist in list(classifier._feature_probdist.items()): 
    p = probdist.prob(True)
    word = fname.split('(')[1].split(')')[0]
    if word != '':
        if word not in dtm.columns:
            counts[word] = 0
        
        counts.at[class2idx[label],word] = probdist.freqdist()[1]

counts.head()
```

The output is the following table, with 2001 columns, each representing how many times each word appear for each class:

| action | the         | of   | in   | was  | a    | season | only | for  | had  | ...  | mary | anaheim | signs | anyway | aguaclara | damon | es   | para | monday | competitor |      |
| ------ | ----------- | ---- | ---- | ---- | ---- | ------ | ---- | ---- | ---- | ---- | ---- | ------- | ----- | ------ | --------- | ----- | ---- | ---- | ------ | ---------- | ---- |
| 0      | ordinal     | 937  | 421  | 529  | 373  | 48     | 198  | 1    | 110  | 167  | ...  | 0       | 2     | 1      | 0         | 0     | 0    | 2    | 1      | 0          |      |
| 1      | superlative | 1083 | 539  | 620  | 422  | 71     | 173  | 13   | 135  | 247  | ...  | 0       | 0     | 1      | 1         | 2     | 0    | 0    | 0      | 0          |      |
| 2      | majority    | 1397 | 1378 | 809  | 125  | 248    | 232  | 5    | 224  | 189  | ...  | 0       | 1     | 1      | 1         | 1     | 1    | 1    | 1      | 2          |      |
| 3      | unique      | 1227 | 363  | 669  | 776  | 250    | 142  | 1255 | 110  | 85   | ...  | 2       | 0     | 0      | 1         | 0     | 3    | 1    | 2      | 3          |      |
| 4      | aggregation | 1010 | 759  | 604  | 432  | 221    | 226  | 0    | 317  | 121  | ...  | 0       | 1     | 1      | 0         | 2     | 1    | 0    | 0      | 0          |      |
| 5      | comparative | 712  | 249  | 441  | 215  | 282    | 84   | 7    | 81   | 209  | ...  | 2       | 0     | 0      | 0         | 0     | 0    | 0    | 0      | 0          |      |
| 6      | count       | 1352 | 1005 | 955  | 198  | 343    | 260  | 52   | 196  | 228  | ...  | 1       | 1     | 1      | 2         | 0     | 0    | 1    | 1      | 0          |      |

To get a grasp an how each of the words impacts classification, we will define a few information-theoretic related quantities:

```python
import numpy as np

def kl(p,q):

    div = np.sum(p*np.log(p/q))
    return div

def get_info_gain(counts, word):
    
    words_table = counts.iloc[:, counts.columns != 'action']
    tot_counts = words_table.sum(axis=1).sum(axis=0)
    word_counts = counts[word].sum(axis=0)
    pwi  = word_counts / tot_counts
    pc   = words_table.sum(axis=1) / tot_counts
    pwic = counts.loc[:,word] / counts.sum(axis=1)
    pcwi = (pwic*pc)/pwi
    info_gain = kl(pcwi, pc) * pwi,
    
    return {
        'word': word,
        'word_count': word_counts,
        'info_gain': info_gain, 
        'pwi': pwi,
        'pwic': list(pwic),
        'kl': kl(pcwi, pc),
        'p(c=0|wi)': pcwi[0],
        'p(c=1|wi)': pcwi[1],
        'p(c=2|wi)': pcwi[2],
        'p(c=3|wi)': pcwi[3],
        'p(c=4|wi)': pcwi[4],
        'p(c=5|wi)': pcwi[5],
        'p(c=6|wi)': pcwi[6],
        'total_counts': tot_counts,
           }

```

For debugging purposes, we expect the word ‘only’ to be have a large information gain and that in case it shows, the probability for the class ‘uniqe’ will be high, let’s check:

```python
get_info_gain(counts, 'only')
```

will give:

```python
{'word': 'only',
 'word_count': 1333,
 'info_gain': 0.020551914070369777,
 'pwi': 0.01304905387017513,
 'pwic': [8.4e-05, 0.001, 0.0003, 0.08, 0.0, 0.0006, 0.003],
 'kl': 1.5749735018983375,
 'p(c=0|wi)': 0.0007501875468867217,
 'p(c=1|wi)': 0.00975243810952738,
 'p(c=2|wi)': 0.0037509377344336087,
 'p(c=3|wi)': 0.9414853713428356,
 'p(c=4|wi)': 0.0,
 'p(c=5|wi)': 0.005251312828207052,
 'p(c=6|wi)': 0.03900975243810952,
 'total_counts': 102153,
}
```

And we see that indeed, `p(c=3|wi)` is indeed higher than the rest with `94%`.

```python
dtm.iloc[:, dtm.columns != 'insight_type'] += 0.5
gains = []
for word in dtm.columns[1:]:
    gains.append(get_info_gain(dtm, word))

gains = pd.DataFrame(gains).sort_values('info_gain', ascending=False)
gains['top_class'] = gains.apply(lambda x: idx2class[x[[f'p(c={i}|wi)' for i in range(6)]].values.argmax()], axis=1)
gains.to_csv(out_path, index=False)
```

And the output table will take the following form:

![image-20220427144620493](/2022-04-27-Interpretable-naive-base-with-NLTK.assets/image-20220427144620493.png)

The table could help us both apply feature extraction by information gain, define rules and just understand how the classifier works. 

