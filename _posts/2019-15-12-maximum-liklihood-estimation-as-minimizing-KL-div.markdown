---
title: "Maximum Likelihood Estimation as Minimizing KL-div in Machine learning"
layout: post
date: 2019-12-15 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- machine learning
- math
star: false
category: blog
author: yotam
description: Maximum Likelihood Estimation as Minimizing KL-div in Machine learning
---

Common supervised machine learning algorithms are usually utilized in order to fulfill some well defined task, usually, given some assumed probability distribution $p_{model}(x;\theta)$ the algorithm learns the model's parameters $\theta$ such that $p_{model} (x;\theta)$ is closest to the *real* probability distribution $p_{data}(x)$.

However, even though intuitivally this closeness of models is the goal, maximization of the (log) likelihood is usually the way it is achived, want to know why? read below.

The plan:

1. Introduce KL divergence $D_{KL}$ and it's use as an intuitive measure for modeling probability distributions.
2. Introduce Maximum Likelihood Estimation (MLE) as a common measure for assessing quality of probability distribution modeling.
3. Present equivalence of optimality objective of MLE and $D_{KL}$  for probability distribution modeling.

#### Kullback-Leibler divergence

A common tool to evaluate similarity between probability distributions is the *[Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)* between probability distributions $p_{data}$ and $p_{model}$:

<div>
$$\begin{align} D_{KL}(p_{data}\vert \vert p_{model}) & \equiv\sum_{i}p_{data}(x_i)\log\left[\frac{p_{data}(x_i)}{p_{model}(x_i;\theta)}\right] \\&=\sum_{x\in \mathcal{X}}\log\left[\frac{p_{data}(x)}{p_{model}(x;\theta)}\right]  \end{align}$$
</div>

 In simple terms, $D_{KL}(p||q)$ gives us the amount of information lost when describing the underlying distribution $p$ using $q$ .
 Note the KL divergence is not symmetric $p\leftrightarrow q$ and does not satisfy the triangle inequality as distance metrics.  

#### Maximum likelihood estimation

 Stating the learning machine objective using using $D_{KL}$ is intuitive, one just wants to minimize $D_{KL}(p_{data}\vert \vert p_{model})$ w.r.t $\theta$ in order to learn the closest possible distribution to the real one. However, we usually seem take a different path by maximizing the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function):

$$\begin{align}\text{Likelihood}_{p_{model}}&=p_{model}(X;\theta)\\&=\prod_{x\in \mathcal{X}} p_{model}(x;\theta) \end{align}$$

Assuming x samples are i.i.d.

The process of maximizing the above is termed [*Maximum likelihood estimation*](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation), a process in which the most probable $\theta$ to have produced the observed data is estimated.

$$\theta_{MLE}=\text{argmax}_\theta \prod_{x\in \mathcal{X}} p_{model}(x;\theta)$$

However, minimizing the above deriving a produt is a hassle (and can easily underflow the numerical precision of the computer), instead, we use the monotonically increasing $log$ function to replace the product with a sum and keep the optimization goal the same since $f(x)$ and $\log f(x)$ will arrive at a maximum for the same $x$. A new optimiztion objective is:

$$\theta_{MLE}=\text{argmax}_\theta\ \sum_{x\in\mathcal{X}} \log\left[p_{model}(x;\theta)\right]$$

#### Present equivalence optimality requirements

Going back to our objective formalized using $D_{KL}$:

$$\begin{align}\theta_{D_{KL}}&= \text{argmin}_\theta\  D_{KL}(p_{data}(x)||p_\theta(x))\\&=\text{argmin}_\theta\  \sum_{x\in\mathcal{X}}\log\left[\frac{p_{data}(x)}{p_\theta(x)}\right]&& \text{Definition of } D_{KL}\\&=\sum_{x\in\mathcal{X}}\left(\require{cancel}\cancel{\text{argmin}_\theta\  \log\left[p_{data}(x)\right]}-\text{argmin}_\theta\  \log\left[p_{model}(x;\theta)\right]\right)&& \log(x/y)=\log(x)-\log(y)\\&=\text{argmin}_\theta\ \sum_{x\in\mathcal{X}}\log\left[p_{model}(x;\theta)\right]&& \text { See (#) below}\\&=\text{argmax}_\theta\ \sum_{x\in\mathcal{X}}\log\left[p_{model}(x;\theta)\right]&& -\text{argmin}_\theta(f_\theta)=\text{argmax}_\theta(f_\theta)\\&=\theta_{MLE}\end{align}$$

(#) $\log\left[p_{data}(x)\right]$  is not a function of $\theta$, it does not play the $\text{argmin}_\theta$ game and can be removed.

Indeed we see that optimality of probability distribution modeling under the two apparatuses are in fact equivalent, this (rather trivial) interpretation of MLE as demanding the model to be close to the underling distribution may clear the goal of ML algorithm objectives.  
