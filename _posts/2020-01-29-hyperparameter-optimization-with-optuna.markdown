---

title: "A 5 min guide to hyper-parameter optimization with Optuna"
layout: post
date: 2020-01-29 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:

- machine learning
- math
star: false
category: blog
author: yotam
description: A 5 min guide to hyper-parameter optimization with Optuna

---

# A 5 min guide to hyper-parameter optimization with Optuna

Hyper-parameter optimization with Optuna

### Finding the best hyper-parameters for your model is now a breeze.

![](https://cdn-images-1.medium.com/max/2000/1*LASFZO4EStM0DlRlECHdcQ.png)

In this post, we will take a simple functioning pytorch neural network training script and enhance it using the [***Optuna](https://optuna.org/)*** package(docs [here](https://optuna.readthedocs.io/en/stable/tutorial/index.html)). This will allow easy assimilation of **smart** **hyper-parameter tuning** and **trial pruning** into your ML workflow with **minimal code modifications.**

Personally, finding the best hyper-parameters to fit my objective has been the worst part of my ML workflow. Up until now, my choices were two: (1) sacrifice my time and sanity and use good old graduate-student-descent or (2) implement a complicated framework to search the parameter space, find the best values, document and visualize the process.

The good news: such a framework already exists, it’s called ***Optuna***, and it’s easy and fun to use.

Our starting point is an MNIST classification script from the Pytorch tutorials. The full script is presented here for completeness, however, since none of the other parts are relevant to our point, I recommend **skimming through all parts leading up to the main function.**

If you want to see the code in action, below is a link to a working google colab notebook.
[**Google Colaboratory**
Optuna MNIST pytorch](https://colab.research.google.com/drive/1TEILbIeyE1wW3daNWOx9rLq0Hc9TXmbV)

Now let’s get down to business.

## Vanilla MNIST Classifier Framework

We begin with imports and data loaders:

<script src="https://gist.github.com/perlitz/04231fb6eb0cfe3b418c60d494dfab17.js"></script>
Next, we implement the network:

<script src="https://gist.github.com/perlitz/efec7fc5abcc940d2981a41e7e184dd5.js"></script>
The train and test methods:
<script src="https://gist.github.com/perlitz/b6551edf9860b9d85ae2a85cf617fc11.js"></script>

And the main function:

<script src="https://gist.github.com/perlitz/21275ae25a77189df60d67ee79ea57bb.js"></script>
Notice that up to this point, nothing interesting really happened, just a simplified pytorch MNIST classifier script. Training the above script with the current randomly chosen hyper-parameters for 5 epochs will give 93% accuracy. Can we do better? Let us see…

## Enhancing the MNIST classifier framework with Optuna

The Optuna framework (installed via pip install optunaand imported as import optuna ) is based upon the ***study** *object. It contains all of the information about the required parameter space, the sampler method and the pruning:

<script src="https://gist.github.com/perlitz/d6a329e42e92250406d33d7411e188b8.js"></script>
Once the study is created, the search space is incorporated via the trial.suggest_ methods. We will embed these into the train_mnist config such that these values:

<script src="https://gist.github.com/perlitz/c9726eb6283032e99fdde4d2681af6fb.js"></script>
will be replaced with these:

<script src="https://gist.github.com/perlitz/c74b2f0227f176888a63e7ac6520e300.js"></script>
In this manner we define the search space to our requirements, once this is done, train_mnist() should get trial as its input and be defined as train_mnist(trial) . Note that a configuration which allows train_mnist to have inputs other than trial exists, check [this](https://optuna.readthedocs.io/en/latest/faq.html) out, if you come across this need.

## Optimization

The final step is to define an objective function, the output of which will be optimized over. In our case we choose train_mnist and its output, the *test error*¹.

Therefore study.optimize will be called, with train_mnist as its parameter:

<script src="https://gist.github.com/perlitz/914c88cbc1476f891ce4358c25e6f972.js"></script>
All in all, main, which was comprised of a single call for train_mnist() , has turned into:

<script src="https://gist.github.com/perlitz/dbc615cc359304cb7366fa09f0c00de2.js"></script>
And that’s it! Once these lines are added to the code, the optimizer will sample the defined parameter space according to the sampler.

After optimization is done, results can be accessed as a dataframe via study.trials_dataframe:

<script src="https://gist.github.com/perlitz/a0942f3df8764de5fcf6c1dd8cc4f8f0.js"></script>
With the following output:

![](https://cdn-images-1.medium.com/max/2000/1*XEJJ_bOcesiWJ40y5KV56A.png)

were one can see all trials and their value. To find the best trial best parameters, study.best_trial and study.best_params can be also used.

Here, we also see how one of the results got a 98.9% test error (~6% improvement) with the same amount of training data and time, this is a major improvement for 3 lines of code.

## Visualization

Other than showing you the best configuration of parameters, Optuna also helps in visualizing the dependence of the objectives on the parameters. Given the study object, all sorts of visualization tools exist in optuna.visualization . You can call plot_parallel_coordinates(study) to view the dependence between the parameters (in this case- lr and momentum) and the objective:

![](https://cdn-images-1.medium.com/max/2000/1*pa-Z4TTtnsGz1LKkqwbr9A.png)

Another way to try to gain some intuition is by using a contour plot. This can be produced by calling plot_contour(study) :

![](https://cdn-images-1.medium.com/max/2000/1*IvLBOIRTPgFuNFXwKk8ynw.png)

To complete the picture, you can also produce a slice plot by calling slice_plot(study) . This can help with the understanding of where the best subspaces are located for each parameter individually.

![](https://cdn-images-1.medium.com/max/2000/1*1Oz4FTV7uaA9a3GzjABgfg.png)

One last visualization option is the study history, produced by plot_optimization_history(study) . This will present the following plot:

![](https://cdn-images-1.medium.com/max/2000/1*6aa_YgxdiGa792WVNvBufA.png)

This shows how Optuna’s study takes place, first by sampling the space evenly, then by focusing in on the most promising areas.

To conclude, I hope you enjoyed this tutorial, I left out several great features like early pruning and different search algorithms, which will have to wait for another time. If I’ve piqued your interest, check out the great [Optuna documentation](https://optuna.readthedocs.io/en/stable/tutorial/index.html), it’s all there.

Enjoy!

[1] Note that in this article I perform a terrible crime for the sake of brevity: one should **never** optimize over the test set, as it will overfit the testing data! A better path would be to split the training set into *train* and *validate*, but since this is not the subject of this post, I‘ve decided to leave it as is.
