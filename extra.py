

# ## OLDOLDOLDOLDOLDOLDOLDOLDOLDOLDOLDOLDOLDOLD


# ## Marginal - $p(x_1=e)= \sum_{j} p(x_1=e, x_2=j)$

def ovar(size, val):
    return one_hot(torch.tensor(val), size).float()




# ## Constrained Join - $p(x_1=e, x_2)$

# With $\lambda^1 = \delta_1$, set $i = 1$
#
# $f(\lambda) = \sum_{j} \lambda^1_1 \lambda^2_j p(x_1=1, x_2=j)$
#
# $f'_{\lambda^2_j}(\lambda) =  \lambda^1_1 p(x_1=1, x_2=j)$

f(o_coin1, l_coin2).backward()
l_coin2.grad



# + [markdown] slideshow={"slide_type": "slide"}
# ## Bayes Rule

# Posterior is constrained joint divided by marginal

# $p(x_1 | x_2=e) = p(x_1, x_2=e) / p(x_2=e)$

# Use log derivative trick

# $(log f)' = f'(\lambda) / f(\lambda)$

# + slideshow={"slide_type": "slide"}

f(l_coin1, l_coin2).log().backward()
l_coin2.grad


# + [markdown] slideshow={"slide_type": "slide"}
# ## Advanced Terms.

# All joints 

# + slideshow={"slide_type": "slide"}

# $f'_{\lambda}(\lambda^1_0) =  \lambda_1^2  $

# $f''_{\lambda}(\lambda^1_0, \lambda_1^1) =  0  $
# $f''_{\lambda}(\lambda^1_0, \lambda_1^2) =  1  $

hessian(coin, (l_coin1, l_coin2))[1][0]



# + [markdown] slideshow={"slide_type": "slide"}
# ## Game Rules
# 
# * Vector operations
# * Gradients
# * Intuitive randomness
# * Counting

# + [markdown] slideshow={"slide_type": "slide"}
# ## Off-Limits
#
# * Bayes' rule
# * Jargon: Prior, posterior, etc.
# * Algorithms such as message passing 
# * Probabilistic programming magic

# + [markdown] slideshow={"slide_type": "slide"}
# ## Approach
#
# * Be declarative
# * Tensorized generative story
# * Aggressive use of autodiff


# + [markdown] slideshow={"slide_type": "slide"}
# ## Example 1: Coin Flips
# 
# Geneative Story:
#
# * I flipped a fair coin, if it was heads I rolled a fair die,
# otherwise I rolled a die waited towards 3.



# + [markdown] slideshow={"slide_type": "slide"}

# ## Modeling Library

# + slideshow={"slide_type": "slide fragment"}

def uniform(size):
    return ones(size) / float(size)

def delta(size, position):
    return one_hot(torch.tensor(position), size)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Example 1: Coin Flips
# 

# + slideshow={"slide_type": "slide fragment"}


COIN, DICE = 2, 6
fair_coin = uniform(COIN)
fair_die = uniform(DICE)
weighted_die = 0.8 * delta(DICE, 4-1) + 0.2 * fair_die


# + [markdown] slideshow={"slide_type": "slide"}
# ## Generative Story (in code):


# + slideshow={"slide_type": "slide fragment"}

def coin_game(v_flip, v_die):
    # I flipped a fair coin
    x_coin = v_flip * fair_coin
    # If it was heads I rolled a fair die.
    x_die = x_coin[1] * fair_die
    # If it was tails I rolled a weighted die.
    x_die += x_coin[0] * weighted_die
    return v_die @ x_die

# + [markdown] slideshow={"slide_type": "slide"}
# ## Inference
#
# To teach inference, we consider asking questions to the
# the underlying model.
#
# To ask a question, we need to specify what is hidden
# and what is observed


# + [markdown] slideshow={"slide_type": "slide"}
# ## Inference Library

# + slideshow={"slide_type": "slide fragment"}

def hidden(size):
    return ones(size, requires_grad=True).float()

def observed(size, val):
    return one_hot(torch.tensor(val), size).float()


# + [markdown] slideshow={"slide_type": "slide"}
# ## Query 1: Warmup
# I saw the coin land on tails.
#
# What is the probability the dice lands on 4?

# + slideshow={"slide_type": "slide fragment"}

v_coin, v_die = observed(COIN, 0), observed(DICE, 4-1)
p = coin_game(v_coin, v_die)
p

# + [markdown] slideshow={"slide_type": "slide"}
# ## Query 2: Warmup
# I saw the coin land on tails.
#
# What are the dice probabilities?

# + slideshow={"slide_type": "slide fragment"}

v_coin, v_die = observed(COIN, 0), hidden(DICE)
coin_game(v_coin, v_die).log().backward()
plt.bar(arange(1, 7), v_die.grad)
__st.pyplot()


# + [markdown] slideshow={"slide_type": "slide"}
# ## Query 3.a
# I saw the dice land on 5.
#
# What are the coin probabilities?

# + slideshow={"slide_type": "slide fragment"}

v_coin, v_die = hidden(COIN), observed(DICE, 5-1)
coin_game(v_coin, v_die).log().backward()
plt.bar(["Tails", "Heads"], v_coin.grad)
__st.pyplot()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Query 3.b
#The die landed on 3.
#
# How did my coin land?

# + slideshow={"slide_type": "slide fragment"}

v_coin, v_die = hidden(COIN), observed(DICE, 3-1)
coin_game(v_coin, v_die).log().backward()
plt.bar(["Tails", "Heads"], v_coin.grad)
__st.pyplot()


# + [markdown] slideshow={"slide_type": "slide"}
# ## Query 4
# I don't see anything.
#
# How did the dice land?

# + slideshow={"slide_type": "slide fragment"}

v_coin, v_die = hidden(COIN), hidden(DICE)
coin_game(v_coin, v_die).log().backward()
plt.bar(arange(1, 7), v_die.grad)
__st.pyplot()


# + [markdown] slideshow={"slide_type": "slide"}
# ## Query 5
# Given any die result.  How did my coin land?

# + slideshow={"slide_type": "slide fragment"}

v_flip, v_die = hidden(2), hidden(6)
out = hessian(coin_game, (v_flip, v_die))[1][0]
out = normalize(out, 1)
plt.imshow(out)
__st.pyplot()


# + [markdown] slideshow={"slide_type": "slide"}
# ## Zoom Out


# + [markdown] slideshow={"slide_type": "slide"}
# ## What is going on?
#
# * Likelihood
# * Conditionals
# * Marginals
# * Joint Marginals

# + [markdown] slideshow={"slide_type": "slide"}
# ## What is going on?
#
# Key idea -> Network Polynomial
#
# * Function from indicators to joints.

# + [markdown] slideshow={"slide_type": "slide"}
# ## Network Polynomial: Known Evidence
#
# $f(\mathbf{\lambda}) = \lambda_0 p(x=e, z=0) + \lambda_1  p(x=e, z=1)$

# $f([0, 1]) = p(x=e, z=1)$



# + [markdown] slideshow={"slide_type": "slide"}
# ## Network Polynomial: Known Evidence
#
# $f(\mathbf{\lambda}) = \lambda_0 p(x=e, z=0) + \lambda_1  p(x=e, z=1)$

# $f(\mathbf{1}) = p(x)$

# $\frac{d \log f}{d\lambda_0}\Bigr|_{(\mathbf{1})} = \frac{p(x, z=0)} {p(x=e)} = p(z=0 | x=e)$

# + [markdown] slideshow={"slide_type": "slide"}
# ## Gradient version?
# $\frac{d \log f}{d\mathbf{\lambda_i^1}} = \frac{p(x=e, z=i)}  {p(x)} = p(z=i | x=e)$

# + [markdown] slideshow={"slide_type": "slide"}
# ## With evidence
#
# $f(\mathbf{\lambda}^1, \mathbf{\lambda}^2) = \lambda^1_0 \lambda^2_0  p(x=e, z=0) + \lambda^1_1 \lambda^2_0  p(x=0, z=1) + \lambda^1_0 \lambda^2_1  p(x=1, z=0) + \lambda^1_1 \lambda^2_1 * p(x=1, z=1)$

# $f(\mathbf{1}, [0, 1]) = p(x=1)$

# $\frac{d \log f}{d\lambda^1_0}\Bigr|_{(\mathbf{1}, [0, 1])} = \frac{p(x=1, z=0)}  {p(x=1)} = p(z=0 | x=1)$









# + [markdown] slideshow={"slide_type": "slide"}
# ## What is going on? 
#
# $f(\mathbf{\lambda}^1, \mathbf{\lambda}^2) = \lambda^1_0 \lambda^2_0  p(x=0, z=0) + \lambda^1_1 \lambda^2_0  p(x=0, z=1) + \lambda^1_0 \lambda^2_1  p(x=1, z=0) + \lambda^1_1 \lambda^2_1  p(x=1, z=1)$


# $\frac{d  f}{d\lambda^1_i d\lambda^2_j} = p(x=i, z =j)$


# + [markdown] slideshow={"slide_type": "slide"}
# ## Generative Story 
# 
# I write a movie review. If I like it, I use
# more words from a positive set. If I don't like it
# I use words from a negative set.


# + slideshow={"slide_type": "slide"}
# ## Generative Story 

V, CLASSES = 1000, 2
d_prior = uniform(CLASSES)
d_pos = 0.5 * uniform(V) + 0.5 * delta(V, 0)
d_neg = 0.5 * uniform(V) + 0.5 * delta(V, 1)


# + slideshow={"slide_type": "slide"}
# ## Generative Story 

def sentiment(v_review, v_words, d_prior=d_prior, d_pos=d_pos, d_neg=d_neg):
    x_sentiment = v_review * d_prior
    return x_sentiment[..., 1] * (v_words * d_pos).sum(-1).prod(-1) + \
           x_sentiment[..., 0] * (v_words * d_neg).sum(-1).prod(-1)






# + slideshow={"slide_type": "slide"}
# ## Generative Story 

v_review, v_words = hidden(CLASSES), observed(V, [0, 2, 2])
sentiment(v_review, v_words).log().backward()
v_review.grad


# + slideshow={"slide_type": "slide"}
# ## Generative Story 

v_review, v_words = hidden(CLASSES), observed(V, [0, 2, 2])
sentiment(v_review, v_words).log().backward()
v_review.grad




# + [markdown] slideshow={"slide_type": "slide"}
# ## What is going on?
#
# $p(x) = \sum_z p(x, z) = \sum_z p(z) \prod_i p(x_i | z)$


# + slideshow={"slide_type": "slide"}
v_review, v_words = observed(CLASSES, [0, 1, 0, 0]), observed(V, [[0, 2, 1], [4, 2, 3], [4, 5, 3], [10, 2, 3]])
c_prior, c_pos, c_neg = hidden(CLASSES), hidden(V), hidden(V)
sentiment(v_review, v_words, c_prior, c_pos, c_neg).log().sum().backward()



# + [markdown] slideshow={"slide_type": "slide"}
# ## What is going on?

# + slideshow={"slide_type": "slide"}
d_prior = normalize(c_prior.grad, 0)
d_pos = normalize(c_pos.grad, 0)
d_neg = normalize(c_neg.grad, 0)


# + [markdown] slideshow={"slide_type": "slide"}
# ## Collecting Statistics
#
# $f(\mathbf{\lambda}, \mathbf{\theta}) = \lambda_0  \theta_0 + \lambda_1   \theta_1$


# $\frac{d \log f}{d\theta_1}\bigr|_{[0, 1], [1, 1]} = \frac{\lambda_1}{\lambda_0  \theta_0 + \lambda_1   \theta_1} = \lambda_1$



# + slideshow={"slide_type": "slide"}


def gmm(X, v_class, d_prior, d_means):
    x_class = v_class * d_prior
    return (stdN(d_means, X) * x_class).sum(-1)

# + slideshow={"slide_type": "slide"}


BATCH, DIM, CLASSES = 100, 2, 4
I = torch.eye(DIM)
N = torch.distributions.MultivariateNormal
y = torch.randint(0, CLASSES, (BATCH,))
d_means = torch.tensor([[2, 2.], [-2, 2.], [2, -2], [-2, -2.]])
d_prior = uniform(CLASSES)

X = N(d_means, I[None, :, :]).sample((BATCH,))[torch.arange(BATCH), y]
mu = torch.rand(CLASSES, DIM)


# + slideshow={"slide_type": "slide"}

for epoch in arange(0, 10):
    # E
    v_class = hidden((X.shape[0], CLASSES))
    gmm(X, v_class, d_prior, d_means).log().sum().backward()

    # M
    q = v_class.grad
    d_means = (q[:, :, None] * X[:, None, :]).sum(0) / q.sum(0)[:, None]

# + slideshow={"slide_type": "slide"}
    
# # Plot
plt.scatter(X[:, 0], X[:, 1], c=q.argmax(1))
plt.scatter(d_means[:, 0],  d_means[:, 1], s= 300, marker="X", color="black")
__st.pyplot()



# END

# + [markdown] slideshow={"slide_type": "slide"}
$\Pr(\mathbf{x}\mid\boldsymbol{\alpha})=\frac{n B\left(\alpha_0,n\right)}
{\prod_{k:x_k>0} x_k B\left(\alpha_k,x_k \right)}$


# + slideshow={"slide_type": "slide"}


def lmbeta(x):
    return lgamma(x).sum() - lgamma(x.sum())

def dirichletmultinomial(v_obs, v_log_prob, d_prior):
    return (v_log_prob * d_prior).sum() - lmbeta(d_prior) + (v_log_prob * v_obs).sum()

# DIRICHLET

v_obs = torch.tensor([1., 10., 1.], requires_grad=True)
d_prior = torch.tensor([5., 4., 1.], requires_grad=True)
v_log_prob = torch.tensor([0.33, 0.33, 0.33]).log()
v_log_prob.requires_grad_(True)
out = dirichletmultinomial(v_obs, v_log_prob, d_prior)
out.logsumexp(0).backward()

out
p = v_log_prob.grad
p


# + slideshow={"slide_type": "slide"}


def lbeta(x, y):
    return lgamma(x) + lgamma(y) - lgamma(x + y)

def dirichletcategorical(v_obs, d_prior):
    n = log(torch.tensor(v_obs.shape[0]).float())
    n_j = v_obs.sum(0)  
    a_0 = d_prior.sum()
    return lgamma(a_0) - lgamma(n + a_0) + (lgamma(n_j + d_prior) - lgamma(d_prior)).sum() 
    # lgramma(
    



    # return log(n) + lbeta(a_0, n) - (total.log() + lbeta(d_prior, total)).sum()


# v_obs, d_prior = observed(3, [0, 1, 1, 0, 1, 2]), torch.tensor([5., 4., 1.], requires_grad=True)
# v_obs2 = hidde(3)
# v_obs3 = vstack([v_obs, v_obs2])
# out = dirichletcategorical(v_obs3,  d_prior)
# out.backward()
# o = v_obs2.grad
# o

# + slideshow={"slide_type": "slide"}
# ## Generative Story 

V, CLASSES = 1000, 2
d_prior = uniform(CLASSES)
d_pos = 0.5 * uniform(V) + 0.5 * delta(V, 0)
d_neg = 0.5 * uniform(V) + 0.5 * delta(V, 1)


# + slideshow={"slide_type": "slide"}
# ## Generative Story 

def sentiment(v_review, v_words, d_prior=d_prior, d_pos=d_pos, d_neg=d_neg):
    x_sentiment = v_review * d_prior
    return x_sentiment[..., 1] * (v_words * d_pos).sum(-1).prod(-1) + \
           x_sentiment[..., 0] * (v_words * d_neg).sum(-1).prod(-1)




# + slideshow={"slide_type": "slide"}
# ## Generative Story 

v_review, v_words = hidden(CLASSES), observed(V, [0, 2, 2])
sentiment(v_review, v_words).log().backward()
v_review.grad


# + slideshow={"slide_type": "slide"}
# ## Generative Story 

v_review, v_words = hidden(CLASSES), observed(V, [0, 2, 2])
sentiment(v_review, v_words).log().backward()
v_review.grad




# + [markdown] slideshow={"slide_type": "slide"}
# ## What is going on?
#
# $p(x) = \sum_z p(x, z) = \sum_z p(z) \prod_i p(x_i | z)$


# + slideshow={"slide_type": "slide"}
v_review, v_words = observed(CLASSES, [0, 1, 0, 0]), observed(V, [[0, 2, 1], [4, 2, 3], [4, 5, 3], [10, 2, 3]])
c_prior, c_pos, c_neg = hidden(CLASSES), hidden(V), hidden(V)
sentiment(v_review, v_words, c_prior, c_pos, c_neg).log().sum().backward()



# + [markdown] slideshow={"slide_type": "slide"}
# ## What is going on?

# + slideshow={"slide_type": "slide"}
d_prior = normalize(c_prior.grad, 0)
d_pos = normalize(c_pos.grad, 0)
d_neg = normalize(c_neg.grad, 0)


# + [markdown] slideshow={"slide_type": "slide"}
# ## Collecting Statistics
#
# $f(\mathbf{\lambda}, \mathbf{\theta}) = \lambda_0  \theta_0 + \lambda_1   \theta_1$


# $\frac{d \log f}{d\theta_1}\bigr|_{[0, 1], [1, 1]} = \frac{\lambda_1}{\lambda_0  \theta_0 + \lambda_1   \theta_1} = \lambda_1$

