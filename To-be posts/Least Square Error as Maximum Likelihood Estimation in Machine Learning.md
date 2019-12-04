# Least square and Probabilistic optimization, how do they relate 

In my [last post](), the MLE (maximum likelihood estimation) was demonstrated to be a natural objective for model learning. However, usually, it is the Ordinary Least Square (OLS) error minimization as an that is taken as an objective in say Linear regression of Autoencoders . 

In the following text, I will focus on the famous [Linear regression](https://en.wikipedia.org/wiki/Linear_regression) and show how MLE demand naturally turns to OLS.  

#### Linear regression in a nut shell

Linear regressions is a linear approach to modeling the relationship between a scalar response and a set of explanatory variables, technically, given a data set $\{y_{i},\,x_{i1},\ldots ,x_{ip}\}_{i=1}^{n}$ is is assumed that there exist some $\boldsymbol{\theta}$ that mediates a linear dependence between any $x_i,y_i$ with some added random noise $\epsilon_i$:
$$
y_i=\bf{x}_i^T\boldsymbol{\theta}+\epsilon_i
$$
for $i=1,...,n$. Fulling all examples in to one equation yields:
$$
\bf{y}=\bf{X}^T\boldsymbol{\theta}+\boldsymbol{\epsilon}\ .
$$

#### In comes Maximum likelihood Estimation (MLE)

Finding the most probable $\boldsymbol{\theta}$ to produce the data can be done via MLE, to put it in mathematical terms, we look for $\boldsymbol{\beta}$ that produces the highest $p(\bf{y},\bf{X}|\boldsymbol{\theta})$, or, simply as derived [before]():
$$
\boldsymbol{\theta}_{MLE}=\text{argmax}_\boldsymbol{\theta}\
 \log\left[p(\bf{y},\bf{X}|\boldsymbol{\theta})\right]\ . demand for an objective, however, there is still a matter of sim
$$
In order to make the above demand a valid objective,  there is still the matter of the type of noise we choose to model the data with.

####  What if we make the noise Gaussian?

