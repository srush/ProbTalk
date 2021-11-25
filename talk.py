# ---
# jupyter:
#   celltoolbar: Slideshow
#   jupytext:
#     cell_metadata_filter: all
#     cell_metadata_json: true
#     formats: md
#     notebook_metadata_filter: all,-language_info,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: light
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   rise:
#     autolaunch: true
#     transition: none
# ---

# + slideshow={"slide_type": "skip"}
from torch import *
from torch.nn.functional import one_hot, pad
import torch
import seaborn
import celluloid
import matplotlib.pyplot as plt
seaborn.set_context("talk")

    
def stdN(means, points):
    I = torch.eye(means.shape[-1])
    return torch.distributions.MultivariateNormal(means, I[None, :, :]).log_prob(points[:, None, :]).exp()

# + [markdown] slideshow={"slide_type": "slide"}
# # Differential Inference: A Criminally Underused Tool
# 
# [@srush](https://twitter.com/srush_nlp) 


# + [markdown] slideshow={"slide_type": "slide"}

# ## Style

# This talk is a live working PyTorch notebook.

# https://github.com/srush/ProbTalk


# + [markdown] slideshow={"slide_type": "slide"}
# ## Preface
# 
# It is bizarre that the main technical contribution of so many papers
# seems to be something that computers can do for us automatically.
# We would be better off just considering autodiff part of the
# optimization procedure, and directly plugging in the objective
# function.  In my opinion, this is actually harmful to the field. - Justin Domke, 2009


# + [markdown] slideshow={"slide_type": "slide"}
# ## Differential Inference
#
# Abuse (auto)differentiation to perform probabilistic inference. 

# + [markdown] slideshow={"slide_type": "slide"}
# ## Disclaimer
# 
# This talk contains no new research:  
#
# * A Differential Approach to Inference in Bayesian Networks (Darwiche, 2000)

# Also Check out

# * Autoconj: Recognizing and Exploiting Conjugacy Without a Domain-Specific Language (Hoffman, Johnson, Tran, 2018)
# * A Tutorial on Probabilistic Circuits  (Antonio Vergari, Robert Peharz, YooJung Choi, and Guy Van den Broeck, AAAI 2020)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Motivation
#
# * I think this stuff is really cool
# * I think it is underused
# * I think elementary probabilty is poorly taught


# + [markdown] slideshow={"slide_type": "slide"}
# # Part 1: Counting the Hard Way

# + [markdown] slideshow={"slide_type": "slide"}
# ## Goal

# I have two coins, how many different ways can I place them?

# TT
# TH
# HT
# HH

# Let's do this the hard way :)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Observed Coins
#
# Let $\lambda^1$ be represent each possibility for Coin 1:
# with $\lambda^1= \delta_0$ as tails and $\lambda^1 = \delta_1$ as heads 


# + slideshow={"slide_type": "slide_fragment"}

def ovar(size, val):
    return one_hot(torch.tensor(val), size).float()

heads = ovar(2, 1)
tails = ovar(2, 0)
plt.imshow(stack([heads, tails]))
__st.pyplot()



# + [markdown] slideshow={"slide_type": "slide"}
# ## Latent Coins
#
# If we do not know the state, we use a $\lambda^1 = \mathbf{1}$.

# + slideshow={"slide_type": "slide_fragment"}

def lvar(size):
    return ones(size, requires_grad=True).float()

l_coin1, l_coin2 = lvar(2), lvar(2)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Counting

# We can use this to count.

# $f(\lambda) = \lambda_0^1  \lambda_0^2 + \lambda_0^1  \lambda_1^2 + \lambda_1^1  \lambda_0^2 + \lambda_1^1  \lambda_1^2$


# + slideshow={"slide_type": "slide"}

def f(l_coin1, l_coin2):
    return (l_coin1[None, :] * l_coin2[:, None]).sum()

f(l_coin1, l_coin2)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Constrained Counting 
#
# We can constrain this count.

# $f(\lambda) = \lambda_0^1  \lambda_0^2 + \lambda_0^1  \lambda_1^2 + \lambda_1^1  \lambda_0^2 + \lambda_1^1  \lambda_1^2$


# Set $\lambda^2 = \delta_0$ then $f(\lambda) = \lambda_0^2 + \lambda_1^2$

# + slideshow={"slide_type": "slide_fragment"}

o_coin2 = ovar(2, 0)
f(l_coin1, o_coin2)


# + [markdown] slideshow={"slide_type": "slide"}
# ## Queries
#

# How do we get the count under *all* constraints?

# + [markdown] slideshow={"slide_type": "slide"}

# ## Differential counting

# $f(\lambda) = \lambda_0^1  \lambda_0^2 + \lambda_0^1  \lambda_1^2 + \lambda_1^1  \lambda_0^2 + \lambda_1^1  \lambda_1^2$



# $f'_{\lambda_0^1}(\lambda) =   \lambda_0^2 +   \lambda_1^2  \sout{ +\lambda_1^1  \lambda_0^2 + \lambda_1^1  \lambda_1^2}$

# + slideshow={"slide_type": "slide_fragment"}

f(l_coin1, l_coin2).backward()
l_coin1.grad[0]

# + slideshow={"slide_type": "slide_fragment"}

# Also gives.
l_coin2.grad


# + [markdown] slideshow={"slide_type": "slide"}
# Can apply both techniques simultaneously. 


# + slideshow={"slide_type": "slide"}

f(l_coin1, o_coin2).backward()
l_coin1.grad[0]

# + [markdown] slideshow={"slide_type": "slide"}
# ## Counting with  Branching 
#
# Place Coin 1.
#
# * If tails, Coin 2 must be heads.
# * If heads, Coin 2 can be either.


# + [markdown] slideshow={"slide_type": "slide"}
# ## Counting Function

# Incorporate branching

# $f(\lambda) = \lambda_0^1  \lambda_1^2 + (\sum_j \lambda_1^1 \lambda_j^2)$

# + slideshow={"slide_type": "slide_fragment"}

def f(l_coin1, l_coin2):
    # If tails, Coin 2 must be heads
    e1 = l_coin1[0] * l_coin2[1]
    
    # If heads, Coin 2 can be either
    e2 = (l_coin1[1] * l_coin2).sum()
    
    return e1 + e2 

# + [markdown] slideshow={"slide_type": "slide"}
# ## Counting

# Number of ways the coins can land. 

# + slideshow={"slide_type": "slide_fragment"}

l_coin1, l_coin2 = lvar(2), lvar(2)
f(l_coin1, l_coin2)


# + [markdown] slideshow={"slide_type": "slide"}

# ## Query

# Number of ways the coins can land. 

# $f'_{\lambda}(\lambda^1_0) =  \lambda_1^2$

# $f'_{\lambda}(\lambda^1_1) =  \sum_j \lambda_j^2$

# + slideshow={"slide_type": "slide_fragment"}

f(l_coin1, l_coin2).backward()
l_coin1.grad




# + [markdown] slideshow={"slide_type": "slide"}

# ## Constrained Query

# Number of ways the coins can land, depending on the first.

# + slideshow={"slide_type": "slide_fragment"}


o_coin1, l_coin2 = ovar(2, 0), lvar(2)
f(o_coin1, l_coin2).backward()
l_coin2.grad


# + [markdown] slideshow={"slide_type": "slide"}
# # Part 2: Probabilistic Inference
#

# + [markdown] slideshow={"slide_type": "slide"}
# ## Differential Inference
#
# This counting trick allows us to derive many
# discrete probability identities directly. 



# + [markdown] slideshow={"slide_type": "slide"}
#
# We specify:
# * Joint - $p(x_1, x_2)$
#
# For observed evidence $e$, we get for free:

# * Marginal - $p(x_2=e)$
# * Constrained Joint - $p(x_1, x_2=e)$
# * Conditional - $p(x_1 | x_2=e)$

# + [markdown] slideshow={"slide_type": "slide"}
# ## What is the benefit?
#
# Declarative generative code with no need for extra code.


# + [markdown] slideshow={"slide_type": "slide"}
#
# ## Coins the Hard Way
#
# Flip two fair coins

# + slideshow={"slide_type": "slide_fragment"}


# + [markdown] slideshow={"slide_type": "slide"}
#
# This function represents the full joint $p(x_1, x_2)$.

# $f(\lambda) = \sum_{i,j} \lambda^1_i \lambda^2_j\  p(x_1=i, x_2=j)$

# + slideshow={"slide_type": "slide_fragment"}

d_coin = torch.ones(2) / 2.
def f(l_coin1, l_coin2):
    # Flip Coin 1 and Coin 2
    flip1 = d_coin * l_coin1
    flip2 = d_coin * l_coin2
    # Sum them up p(x_1, x_2)
    return (flip1[:, None] * flip2[None, :]).sum()


# + [markdown] slideshow={"slide_type": "slide"}
# ## Joint Probability

# Using `ovar` we can get out the joint under observations. 

# $p(x_1=1, x_2=0)$

# + slideshow={"slide_type": "slide_fragment"}
o_coin1, o_coin2 = ovar(2, 1), ovar(2, 0)
f(o_coin1, o_coin2)


# + [markdown] slideshow={"slide_type": "slide"}
# ## Joint Probability

# + [markdown] slideshow={"slide_type": "slide"}
# Using `lvar` we can marginalize out unseen flips. 

# * $p(x_2=0)$

# + [markdown]  slideshow={"slide_type": "slide_fragment"}

# With $\lambda^1 = \mathbf{1}$ ,  $\lambda^2 = \delta_0$
#
# $f(\lambda) = \sum_{i} p(x_1=i, x_2=0) = p(x_2=0)$

# + slideshow={"slide_type": "slide_fragment"}
l_coin1, o_coin2 = lvar(2), ovar(2, 0)
f(l_coin1, o_coin2)


# + [markdown] slideshow={"slide_type": "slide"}
# ## Constrained Joint

# * $p(x_1, x_2=e)$

# Set $\lambda^1 = \mathbf{1}$ and $\lambda^2 = \delta_0$, 
# $f(\lambda) = \sum_{i} \lambda^1_i p(x_1=i, x_2=0)$
#
# therefore
#
# $f'_{\lambda^1_0}(\lambda) =   p(x_1=0, x_2=0)$, 
# $f'_{\lambda^1_1}(\lambda) =   p(x_1=1, x_2=0)$

# + slideshow={"slide_type": "slide"}

# ## Constrained Joint


l_coin1, o_coin2 = lvar(2), ovar(2, 0)
f(l_coin1, o_coin2).backward()
l_coin1.grad[0]



# + [markdown] slideshow={"slide_type": "slide"}
#
# ## Conditional

# * $p(x_1 | x_2=e)$

# With Bayes' Rule:  $p(x_1 | x_2=e) = \frac{p(x_1, x_2=e)}{p(x_2=e)}$

# * Numerator is constrained joint ($f'$)
# * Denominator is marginal ($f$)

#  $f'(\lambda) / f(\lambda)$

# + [markdown] slideshow={"slide_type": "slide"}
#
# ## Conditional

# Use log trick $(\log f)' = f'(\lambda) / f(\lambda)$

# Gives $f'(\lambda) / f(\lambda)$ with $\lambda^2=\delta_e$

# + slideshow={"slide_type": "slide_fragment"}

l_coin1, o_coin2 = lvar(2), ovar(2, 1)
f(l_coin1, o_coin2).log().backward()
l_coin1.grad

# + [markdown] slideshow={"slide_type": "slide"}
#
# ## Punchline 
#
# Just use backprop for discrete inference. 

# + [markdown] slideshow={"slide_type": "slide"}
# ## Part 3: Fancy Coins

# + [markdown] slideshow={"slide_type": "slide"}
#
# ## Example: More Coins
# Flip Coin 1.
#
# * If tails, place Coin 2 as heads.
# * If heads, flip Coin 2.


# + [markdown] slideshow={"slide_type": "slide"}

# ## Generative Process

# + slideshow={"slide_type": "slide_fragment"}

def f(l_coin1, l_coin2):
    # Flip Coin 1
    flip1 = d_coin * l_coin1
    
    # If tails, place Coin 2 as heads.
    e1 = flip1[0] * l_coin2[1]
    
    # If heads, flip Coin 2.
    flip2 = l_coin2 * d_coin
    e2 = (flip1[1] * flip2).sum()
    
    return e1 + e2 

# + [markdown] slideshow={"slide_type": "slide"}

# ## Marginal Inference

# $p(x_1)$ and $p(x_2)$

# + slideshow={"slide_type": "slide_fragment"}

l_coin1, l_coin2 = lvar(2), lvar(2)
f(l_coin1, l_coin2).log().backward()
l_coin1.grad, l_coin2.grad



# + [markdown] slideshow={"slide_type": "slide"}
# ## Example: Coins and Dice
# 
# Geneative Story:
#
# * I flipped a fair coin, if it was heads I rolled a fair die,
# otherwise I rolled a weighted die.

# + slideshow={"slide_type": "slide"}

COIN, DICE = 2, 6
fair_coin = ones(COIN) / 2.0
fair_die = ones(DICE) / 6.0
weighted_die = 0.8 * one_hot(tensor(3), DICE) + 0.2 * fair_die

# + [markdown] slideshow={"slide_type": "slide"}

# ## Generative Story (in code):


# + slideshow={"slide_type": "slide fragment"}

def f(v_flip, v_die):
    # I flipped a fair coin
    x_coin = v_flip * fair_coin
    
    # If it was heads I rolled a fair die.
    roll1 = v_die * fair_die
    e1 = x_coin[1] * roll1

    # If it was tails I rolled a weighted die.
    roll2 = v_die * weighted_die
    e2 = x_coin[0] * roll2
    return (e1 + e2).sum()


v_coin, v_die = ovar(COIN, 0), lvar(DICE)
f(v_coin, v_die).log().backward()
plt.bar(arange(0, DICE)+1, v_die.grad)
__st.pyplot()


# + [markdown] slideshow={"slide_type": "slide"}
# ## Coin from Dice 1

# + slideshow={"slide_type": "slide fragment"}

v_coin, v_die = lvar(COIN), ovar(DICE, 5)
f(v_coin, v_die).log().backward()
plt.bar(["Tails", "Heads"], v_coin.grad)
__st.pyplot()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Coin from Dice 2

# + slideshow={"slide_type": "slide fragment"}

v_coin, v_die = lvar(COIN), ovar(DICE, 3)
f(v_coin, v_die).log().backward()
plt.bar(["Tails", "Heads"], v_coin.grad)
__st.pyplot()


# + [markdown] slideshow={"slide_type": "slide"}
# ## Dice Marginal

# + slideshow={"slide_type": "slide fragment"}

v_coin, v_die = lvar(COIN), lvar(DICE)
f(v_coin, v_die).log().backward()
plt.bar(arange(0, 6)+1, v_die.grad)
__st.pyplot()


# + [markdown] slideshow={"slide_type": "slide"}
# # Example: Summing Up
#
# Let's extend this trick to more complex combinations.  


# + [markdown] slideshow={"slide_type": "slide"}

# A simple 1D convolution for summing random variables.

# + slideshow={"slide_type": "slide fragment"}

def padconv(x, y):
    s = x.shape[0] 
    return x.flip(0) @ pad(y, (s-1, s-1)).unfold(0, s, 1).T


# + [markdown] slideshow={"slide_type": "slide"}

# ## Sum of Variables

# Let `l_count` be the sum of two uniform variables.

# + slideshow={"slide_type": "slide"}

def f(l1, l2, l_count):
    s = l1.shape[0]
    d = ones(s) / s
    e1 = d * l1
    e2 = d * l2
    return (padconv(e1, e2) * l_count).sum()

# + [markdown] slideshow={"slide_type": "slide"}

# ## Sum of Coins

# Let `l_count` be the sum of two uniform variables.

# + slideshow={"slide_type": "slide"}

l_coin1, l_coin2, l_count = lvar(2), lvar(2), lvar(3)
f(l_coin1, l_coin2, l_count).log().backward()
plt.bar(arange(0, 3), l_count.grad)
__st.pyplot()


# + [markdown] slideshow={"slide_type": "slide"}

# ## Sum of Dice

# + slideshow={"slide_type": "slide"}

l_die1, l_die2, l_count = lvar(6), lvar(6), lvar(11)
f(l_die1, l_die2, l_count).log().backward()
l_count.grad
plt.bar(arange(2, 13), l_count.grad)
__st.pyplot()

# + [markdown] slideshow={"slide_type": "slide"}

# ## Dice Conditioned on Sum

# + slideshow={"slide_type": "slide"}

l_die1, l_die2, o_count = lvar(6), lvar(6), ovar(11, 10)
f(l_die1, l_die2, o_count).log().backward()
plt.bar(arange(0,6 )+1, l_die2.grad)
__st.pyplot()


# + [markdown] slideshow={"slide_type": "slide"}
# ## Part 4: Real Models


# + [markdown] slideshow={"slide_type": "slide"}

# ## Example: Graphical Models

# We can apply this method directly to classic graphical models.


# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/SimpleBayesNet.svg/1920px-SimpleBayesNet.svg.png) 

# + [markdown] slideshow={"slide_type": "slide"}

# ## Conditional Probabilities

# + slideshow={"slide_type": "slide"}

def bern(p):
    return tensor([1.0-p, p])

# + slideshow={"slide_type": "slide"}

# p(R)
rain = bern(0.2)

# p(S | R)
sprinkler_rain = stack([bern(0.4), bern(0.01)]).T

# p(W | S, R)
wet = stack([stack([bern(0.0), bern(0.8)]),
             stack([bern(0.9), bern(0.99)])])
wet.permute(2, 0, 1)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Bayes Net


# + slideshow={"slide_type": "slide"}
def f(l_rain, l_sprinkler, l_wet):
    # r ~ P(R)
    e_rain = l_rain * rain
    # s ~ P(S | R=r)
    e_sr = l_sprinkler[:, None] * sprinkler_rain *  e_rain
    # w ~ P(W | S=s, R=r)
    e_wet = l_wet[:, None, None] * wet * e_sr
    return e_wet.sum()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Joint Probability

# + slideshow={"slide_type": "slide"}

o_rain, o_sprinkler, o_wet = ovar(2, 1), ovar(2, 1), ovar(2, 1)
out = f(o_rain, o_sprinkler, o_wet)
out

# + [markdown] slideshow={"slide_type": "slide"}
# ## Marginal Inference

# $P(R)$

# + slideshow={"slide_type": "slide"}

l_rain, l_sprinkler, l_wet = lvar(2), lvar(2), lvar(2)
f(l_rain, l_sprinkler, l_wet).log().backward()
l_rain.grad


# + [markdown] slideshow={"slide_type": "slide"}
# ## Conditional Inference

# $P(R | W=1)$

# + slideshow={"slide_type": "slide"}

l_rain, l_sprinkler, o_wet = lvar(2), lvar(2), ovar(2, 1)
f(l_rain, l_sprinkler, o_wet).log().backward()
l_rain.grad

# + [markdown] slideshow={"slide_type": "slide"}

# ## Example: Gaussian Mixture Model

# + [markdown] slideshow={"slide_type": "slide"}

# ## Data

# + slideshow={"slide_type": "slide"}

BATCH, DIM, CLASSES = 100, 2, 4
I = eye(DIM)
N = torch.distributions.MultivariateNormal
y = randint(0, CLASSES, (BATCH,))
d_means = torch.tensor([[2, 2.], [-2, 2.], [2, -2], [-2, -2.]])
d_prior = ones(CLASSES) / CLASSES
X = N(d_means, I[None, :, :]).sample((BATCH,))[torch.arange(BATCH), y]



# + [markdown] slideshow={"slide_type": "slide"}

# ## Generative Model

# Pick a class, generate point from Gaussian  

# + slideshow={"slide_type": "slide"}

def gmm(X, v_class, d_prior, d_means):
    x_class = v_class * d_prior
    return (stdN(d_means, X) * x_class).sum(-1)

# + [markdown] slideshow={"slide_type": "slide"}

# ## Expectation-Maximization

# Use conditional inference to compute expectation step.

# + slideshow={"slide_type": "slide"}

fig, ax = plt.subplots(nrows=1, ncols=1)
camera = celluloid.Camera(fig)
mu = torch.rand(CLASSES, DIM)

for epoch in arange(0, 10):
    v_class = lvar((X.shape[0], CLASSES))
    gmm(X, v_class, d_prior, mu).log().sum().backward()
    q = v_class.grad

    # Plot
    ax.scatter(X[:, 0], X[:, 1], c=q.argmax(1))
    ax.scatter(mu[:, 0],  mu[:, 1], s= 300, marker="X", color="black")
    camera.snap()
    
    mu = (q[:, :, None] * X[:, None, :]).sum(0) / q.sum(0)[:, None]

__st.write(camera.animate(interval=300, repeat_delay=2000).to_html5_video(), unsafe_allow_html=True)
    
# + [markdown] slideshow={"slide_type": "slide"}
#
# ## Example: Hidden Markov Models

# Hidden markov model 

# + slideshow={"slide_type": "slide"}
def HMM(l_O, l_H, params):
    T, E, P = params
    p = 1.0
    for l in arange(0, l_O.shape[0]):
        P = ((l_H[l] * P)[:, None] * E) @ l_O[l] @ T
        p = p * P.sum()
        P = P / P.sum()
    return (p * P.sum())

# + [markdown] slideshow={"slide_type": "slide"}
#
# Generate simple HMM with circulant transitions

# + slideshow={"slide_type": "slide"}


STATES, OBS = 500, 500
E, T = eye(STATES), zeros(STATES, STATES), 
P = ones(STATES) / STATES
kernel = arange(-6, 7)[:, None]
s = arange(STATES)
T[s, (s + kernel).remainder(STATES)] = 1. / kernel.shape[0]
params = T, E, P

# + [markdown] slideshow={"slide_type": "slide"}
# # ## Posterior Inference
# Posterior inference over states with some known observations

# + slideshow={"slide_type": "slide"}


fig, ax = plt.subplots(nrows=1, ncols=1)
camera = celluloid.Camera(fig)

def ovarN(x, N=OBS): return  one_hot(x, N)[None].float()
def lvarN(s, N=OBS): return  ones(s, N, requires_grad=True)

start = lvarN(1000).detach()
start.requires_grad_(False)
for i in arange(0, 5): 
    start[randint(1000, (1,))[0], :] = ovarN(randint(STATES, (1,))[0])
    states = lvar((start.shape[0], STATES))

    # Run and plot...
    HMM(start, states, params).log().backward()
    ax.imshow(states.grad.transpose(1, 0), vmax=0.02)
    camera.snap()


# HTML(camera.animate(interval=300, repeat_delay=2000).to_jshtml())
__st.write(camera.animate(interval=300, repeat_delay=2000).to_html5_video(), unsafe_allow_html=True)

# + [markdown] slideshow={"slide_type": "slide"}

# ## Conclusion
