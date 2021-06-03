title: "Neural Ordinary Differential Equations Paper Summary"
layout: post
date: 2021-02-13 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:

- bash
star: false
category: blog
author: yotam
description: Neural Ordinary Differential Equations Paper Summary

# A Neural Ordinary Differential Equations[^1] Paper Summary

Citation: [^1]

### Why this paper?

- It won NIPS best paper award winner.
- It has been Cited more than 170 times in less than a year (as of Sep/19).
- It has a few very interesting follow-up works from D. Duvenauds group.
- Its a very cool idea. 

### Our Plan:

This paper presents two main parts

-  Introducing Neural Ordinary Differential Equations (NODE) as a new family of deep neural network models
-  Demonstrating their properties

Starting from the top, we will begin with a recap of Ordinary Differential Equations, we will then introduce NODE as a variant of Residual Neural Network with continuous dynamics.

## Introduction to NODE

An explanation of NODEs requires some basic understanding of differential equations and their solvers, below is a short recap.

### Ordinary Differential Equations Recap

A differential equation describes a state given it's derivatives (the way it changes), as an simple example, for a state $\bf{z}$, the *location*,    $\frac{d\bf{z}(t)}{dt}=v(t)$ will describe it through time $t$ via the *velocity*.  In *Ordinary Differential Equation* (ODE), the state is derived w.r.t just one quantity (e.g time in the above example).

Differential equations are solved via *integration* given initial conditions $\bf{z}_{t_0}$ : 
$$
\bf{z}(t)=\bf{z}_{t_0}+\int_{t_0}^t dt'\ v(t').
$$
In a set of simple cases, these can may solved via analytical methods and the above form is the relevant. In all other cases, these will be solved via some form of numerical integration in a quantized form :  
$$
\bf{z}(t)=\bf{z}_{t_0}+\lim_{\Delta t\rightarrow 0} \Delta t\sum_{t'=t_0}^t \ v(t').
$$
Note that the limit cannot really go to zero and will be an important parameter in sections to come. 

Generalizing this by replacing $\bf{v}(t) \leftrightarrow f(\bf{z}_t , \theta, t)$ with $\theta$ as some parameters of $\bf{f}$ and writing it the way a compiler understands, leaving `step_size` undefined for a minute:

```python
z = z0
for tp in range(t0, t, step_size):
	z = z + f(z, theta, tp)
return z
```

Lets see how numerical integration behaves in the real world, given some vector field defined by equation:
$$
\begin{equation}
\frac{d\mathbf{z}_t}{dt}=f(\mathbf{z}_{t},\theta,t)
\tag{1}
\end{equation}
$$
An exact (analytical) solution to the trajectory of $\bf{z}_t$ is shown in figure 1:

| ![1572246340632](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1572246340632.png) |
| ------------------------------------------------------------ |
| Figure 1: Exact solution of Eq.1, the lines corresponds to the value of $\bf{f}$ along $\bf{z},t$ space, the inclines represents a normalized size of $\bf{f}(z,t)$. credit: D. Duvenaud |

If we instead wish to use a numerical integration, we must account for the step size, the most basic way will be to choose a certain step size and use it along the trajectory, this method is called the [*Euler method*](https://en.wikipedia.org/wiki/Euler_method), it amounts to applying  $\mathbf{z}(t+h)=\mathbf{z}(t)+hf(\mathbf{z},t)$  for $(t-t_0)/h$ times. The solution via an Euler method is shown in fig 2:

| ![1572246365180](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1572246365180.png) |
| ------------------------------------------------------------ |
| Figure 2: Exact and Euler solutions to Eq.1 note that the step size is consistent along all the trajectory. credit: D. Duvenaud |

Note that this example shows rather bad performance of the Euler solution just to stress the point that this is a naïve solution, one could have chosen an arbitrarily small step size $h$ and get a perfect fit at a great calculation cost.

To account for the above problem, adaptive methods (e.g [Runga-Kutta](https://en.wikipedia.org/wiki/Runge–Kutta_methods) came in the picture, in these methods the step size changes according the approximation error such that a tolerable error is defined and step size is chosen according to it. A solution via a kind of adaptive method is shown in fig 3:

| ![1572246387356](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1572246387356.png) |
| ------------------------------------------------------------ |
| Figure 3: All three solutions (Adaptive solution is drawn on top of the exact). Note the varying step sizes in the adaptive solver. To enhance the motivation, note that adaptive solver evaluates  $\bf{f}(z,t)$ less times than the Euler method solution does and achieves much better results. credit: D. Duvenaud |

Where Adaptive solver solution is shown to preform better than the Euler method with less computation.

Pushing forward,  A second concept important to the understanding of NODE are Residual Neural Networks.

### ResNets - Residual Neural Networks

Deep Residual Neural Networks[^5] are a variation the simple *Vanilla* neural network allowing for a solution of the some latter's inherent problems  when expanding the scale of the network.  First, what are Vanilla neural networks?

##### *Vanilla* Neural Networks

A *Vanilla* NN can be thought of as a a stack of functions operating sequentially over some input. As depicted in figure 5, the function $f(z_t,\theta_t)$ is defined every step $t$ by the parameters $\theta_t$  and operates upon the hidden state $z_t$ outputting the hidden state $z_{t+1}$. Putting this to pseudocode, this seems natural as depicted below. 

| ![1571553780407](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1571553780407.png) |
| ------------------------------------------------------------ |
| Figure 4:                                                    |

The above method along with a few structure-preserving improvements (CNNs[^2], RELU activations[^3], Batch normalization[^4] etc) had revolutionized the way machines learn and the way tasks are preformed. As NN performance increased, the ability of these nets was pushed forward by more parameters and more operations aiming to model more complicated data and tasks. However, Around the time that huge networks like GoogleLeNet and VGG began to appear and researchers were able to harness more MACs to their side that some bound on the deeper is better was found (the so called *depth degradation*, see figure 5), and dealt with by the introduction of RNNs, Residual neural networks.

| ![1571555829555](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1571555829555.png) |
| ------------------------------------------------------------ |
| Figure 5: Depth degradation as presented in [^1], Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus test error. |

##### Residual Neural Networks - Incremental representation change

ResNets were introduced to solve the above peculiar result for if there is a better performance for 20-layer network, why wont the 56-layer network solver just learn the 36 last layers as identity mapping? 

following this intuition and the intuition that representation will benefit from a more structured incremental change, ResNets added an extra skip connection constructing the architecture in such a way that a layers has to now learn the *residual* $\Delta \bf{z}_t$ which allows for *incremental* representation change and trivial identity mapping learning (a zero output from the layer).

Given the intuition that representation of the hidden state vary incrementally, this change allows for the layers to expand around zero output (no change) rather than around the unit output as in the vanilla case which makes learning much easier for these kind of architectures.

| ![1571556824177](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1571556824177.png) |
| ------------------------------------------------------------ |
| Figure 6:  Residual networks architecture, skip connections (in orange) were added, now each layer is to learn the residual, or change of representation between stages. |

## From ResNets to ODENets

Fast forward 3 years and ResNets are everywhere. Equipped with their understanding, it is easy to jump to ODENet architecture.

ODENets are a variation on ResNets in which:

1. All layers share the same set of parameters (preform the same operation).
2. Layers receive their time step as input.

These differences are schematically drown in figure 7. In this case, the input vectors that enter each layer are $[\bf{z}_t,t]=(z^1,z^2...,z^N,t)$ instead of  $\bf{z}_t=(z^1,z^2...,z^N)$ and the layers are all identical (parameter input is $\theta$ instead of $\theta [t]$. 

Implementation wise, for a convolutional network, doing this change amounts to using the same layer all through the forward pass and changing some thing like the following:

```python
def forward(self, theta, x):
    return self._layer(x, theta)
```

with:

```python
def forward(self, t, x):
    tt = torch.ones_like([x[:, :1, :, :]]) * t
    ttx = tourch.cat([tt, x], 1)
    return self.layer(ttx)
```

Which is just concatenating a feature map containing just the value of t in all its cells.

| ![1571559344192](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1571559344192.png) |
| ------------------------------------------------------------ |
| Figure 7: A comparison between ResNet and ODENet, two main differences: (1) ODENet has the same parameters for each layer, this in fact means that these are the **same** layer much like in RNNs. (2) ODENet gets an additional input of the time step of the specific layer. |

This change allows for a redefinition of the dynamics of the state as now it is the same function that operates upon it through time along with the addition of time context, these dynamics can be framed as a differential equation. What we have here is just a funny looking form of the simple Euler integration introduced above with $h=1$.

| ![1571560213451](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1571560213451.png) |
| ------------------------------------------------------------ |
| Figure 8: Pseudocode for the implementation of a forward pass of ODENet, the layers are now all identical and, $t$ can be arbitrarily small and the dynamics are continuous. Keeping with the implementation of ResNet will now resemble Euler integration with h=1. |

Now, given that the forward pass is defined as an integration of a differential equation, one may try to solve it using other, more advance solvers rather than the Euler one, as sketched in figure 9:  

| ![1571560499034](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1571560499034.png) |
| ------------------------------------------------------------ |
| Figure 9: Replacing Euler integration in fig 8 by a black-box ODE solver. |

Modern differential equation solvers have many advantages along which are better stability and speed, these solvers are adaptive, thereby they can evaluate the number of evaluations needed to achieve a solution with some defined error. 

A comparison between ResNets and ODENets taken from the original paper is presented in figure 10, there, each line represent a single input and resulting state trajectory, points are evaluations and lines represents manipulations (layer activations) upon the state.  In ODENet context, depth as a measure of function evaluations is defined an adaptive measure and varies depending on the input, once again *depth* varies depending on the input and accuracy required.

Given all the above, we now have a smarter system with which we can do inference through of some general input in an adjustive manner, however, training these kind of networks by naively backpropagating through an ODE may be infeasible due to numerical errors**, **instability or non-differentiability of the solver. In order to solve these problem, the authors suggested running a 2nd ODE solver backwards in time instead of backpropagating through the forward pass this is called the  *Adjoint method*[^6] for the backwards pass, this will both solve the above problems and add a few other benefits to the architecture, lest see how it goes.

| ![1571561222667](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1571561222667.png) |
| ------------------------------------------------------------ |
| Figure 10:                                                   |

## Training ODENets

Let's dive in, deep learning algorithms are generally build out of two steps, a forward and a backward pass, the **forward pass** will produce the output that in turn will be evaluated and produce the loss, in the **backwards pass** the loss will then be used to get the gradients of the loss w.r.t the weights. 

| ![1572436343062](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1572436343062.png) |
| ------------------------------------------------------------ |
| Figure 11:                                                   |

The ODENet framework for learning follows the same lines, it's equations are in fact very similar to these of the ResNet in the limit of continuity, if you are worried about the derivation, just derive these for a ResNet (dropping the BN stages),  breaking this into pieces:

#### Forward pass

The forward pass equation is depicted in fig. 11 showing the state $\bf{z}(t)$ changing with time due to $f(\mathbf{z}_{t'},\theta,t')$, this equation is the same one we showed above as the inference equation of the ODENet and will be solved in our context using some black-box adjustive ODE solver producing $\bf{z}(t_N)$, the output of the network. Once we have $\bf{z}(t_N)$ we can evaluate it's correctness and get both the loss and the sensitivity $a(t_N)$ of the loss to the last state.

| ![1572244414971](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1572244414971.png) |
| ------------------------------------------------------------ |
| Figure 12:                                                   |

#### Backwards pass

This part is where the method deviates from common backpropagation, backprop is a dynamic programming method where the results of subproblems are saved and utilized later, as we said before, saving the activations from the solver along the way will cause instability and errors.

However, since the dynamics are continuous and can be represented using an ODE, we can just replace the direction of time and solve the forward pass equation backwards in time (see fig. 12) using the same solver as in the forward pass, this will both produce the needed activations along the way and will save us from storing all the activations along the way (constant memory in training!). 

| ![1572244472295](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1572244472295.png) |
| ------------------------------------------------------------ |
| Figure 13:                                                   |

A second and third parts of the backwards pass is shown in fig 13, solving for the sensitivity $a(t)$ and the gradients of the loss $\frac{\part L}{\part \theta}$  backwards in time using the same solver as the forward pass will eventually be used to correct the weights.  We note that these three equations may be solved using a single pass of an ODE with all three backward pass equations concatenated together.

| ![1572244513186](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1572244513186.png) |
| ------------------------------------------------------------ |
| Figure 14:                                                   |

Having understood the above method, we should note that there are other architectures (than ODENet) that allow for constant memory training using reversible ResNets [^7], these requires some restrictions on architecture but scales much better. 









## Demonstrating NODEs properties

So far, we've been introducing the concept of neural ODE, however, for a rounder understating, the paper presented a few demonstrations and use cases for this kind of architecture where a neural ODE can replace a regular network and add some value for the application, a first step will be the insertion of an ODENet instead of an RNN for a better time series model generative process.  

### A generative latent function time-series model

In order to construct some intuition of the authors' system, we will need to (very briefly) visit the concepts of an variational autoencoder and of an RNN encoder-decoder.

#### Variational Autoencoder

In order to understand the use case, some basic knowledge of a variational autoencoder is needed, [here](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf) is a link to a nice post by Irhum Shafkat that captures enough of the essence for our explanation. Having assumed the reader now knows something about VAEs, a short intuitive intro to autoencoders and VAE:

An autoencoder is an a system architecture which allows for unsupervised learning. The training goes like this: (1) encoder encodes the input to some latent space, (2) the latent vector is decoded by the decoder to the output and (3) the resemblance between the input and output is evaluated using the reconstruction error in the form of least squares (given that we are using MLE and a gaussian prior) and is incorporated to the loss that is then feeds back to the network to correct the weights of the encoder and decoder.

Given a trained system, that can reliably encode and decode data, one can try to use the decoder to generate data by naively sampling a point in the latent space and and decoding it. Why naïvely?  the way that simple autoencoders are build will not allow a meaningful sampling from the latent space, however, there are some variations of the autoencoder that are build upon regularization of the architecture that allow for a generative model to emerge from this architecture with one example as VAE, the variational autoencoder.  

| ![1572437600772](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1572437600772.png) |
| ------------------------------------------------------------ |
| Figure 6:                                                    |

|           |
| --------- |
| Figure 6: |

|           |
| --------- |
| Figure 6: |

|           |
| --------- |
| Figure 6: |

|           |
| --------- |
| Figure 6: |

|           |
| --------- |
| Figure 6: |































| ![1572248398268](C:\Users\yotampe\website\perlitz.github.io\_posts\2021-05-11-NODE.assets\1572248398268.png) |
| ------------------------------------------------------------ |
| Figure (): Credit: Chen, Tian Qi                             |



[^1]: Chen, Tian Qi et al. “Neural Ordinary Differential Equations.” *NeurIPS* (2018).
[^2]: Krizhevsky, Alex et al. “ImageNet Classification with Deep Convolutional Neural Networks.” *Commun. ACM* 60 (2012): 84-90.
[^3]: Nair, Vinod, and Geoffrey E. Hinton. "Rectified linear units improve restricted boltzmann machines." *Proceedings of the 27th international conference on machine learning (ICML-10)*. 2010.
[^4]: Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *arXiv preprint arXiv:1502.03167* (2015).
[^5]: He, Kaiming et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016)
[^6]: Pontryagin, L.S. et al., “The Mathematical Theory of Optimal Control Processes” 1962
[^7]: Gomez, Aidan N., et al. "The reversible residual network: Backpropagation without storing activations." 2017.