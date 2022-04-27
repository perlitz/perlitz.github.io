---
title: "Zero shot classification"
layout: post
date: 2020-01-29 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- python
star: false
category: blog
author: yotam
description:  Using pertained language models for zero-shot classification
---

# Zero shot classification using large LM

Based in [this](https://nlp.town/blog/zero-shot-classification/) and [this](https://joeddav.github.io/blog/2020/05/29/ZSL.html)

## Zero shot classification as an NLI task

```
# load model pretrained on MNLI
from transformers import BartForSequenceClassification, BartTokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

class_options = 
		[
    'climate',
    'childhood',
    'international commerse',
    'france',
    ]

for class_option in class_options:

  premise = 'The weather is great'
  hypothesis = f'This text is about {class_option}.'

  input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt')
  logits = model(input_ids)[0]

  # we throw away "neutral" (dim 1) and take the probability of
  # "entailment" (2) as the probability of the label being true 
  entail_contradiction_logits = logits[:,[0,2]]
  probs = entail_contradiction_logits.softmax(dim=1)
  true_prob = probs[:,1].item() * 100
  dict_list.append({'premise':premise,'class':sen_class, 'score':true_prob})


```

