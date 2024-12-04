@def sequence = ["loss-functions"]

# Loss Functions for Classification

**Table of Contents**

\toc


# General ML Classification Tasks

In this blog, general machine learning classification task will be covered.

**Some supervised learning basics**
- Linear regression
- Gradient descent algorithms
- Logistic regression
- Classification and softmax regression

The training steps pretty much consist of the below flow:
- Dataset and Dataloader + Model + Loss and Optimizer = Training

Maximizing the log likelihood in the training step.

**A Probabilistic Model**

The dataset is made of $m$ training examples $(x(i), y(i))_{i\in[m]}$, where

$$\mathcal{L}(\theta \mid x) = \log L(\theta)
                             = -m\log (\sigma \sqrt{2\pi}) - \frac{1}{2\sigma^{2}}\sum_{i=1}^{m}(y(i) - \theta^T x(i))^2$$

And the Jacobian vector product or cost function is :
$$ J(\theta) = \frac{1}{2}\sum_{i=1}(y(i) - \theta^Tx(i))^2 $$

giving rise to the ordinary lease squares regression model,

The gradient of the least squares cost function is:

$$\frac{\partial}{\partial\theta} J(\theta) = \sum_{i=0}^{m} (y(i) - \theta^T x(i)) \frac{\partial}{\partial \theta_{j}} (y(i) - \sum_{i=0}^{d} \theta_{k}x_{k}(i)) = \sum_{i=0}^{m} (y(i) - \theta^{T} x(i)) = \sum_{i=0}^{m} (y(i) = \theta^{T} x(i))x_j(i)$$

# Gradient  Descent Algorithms

Batch gradient descent performs the update

$$ \theta_{j} := \theta_{j} + \alpha \sum_{i=0}^{m} (y(i) - \theta^Tx(i)x_{j}(i)) $$

where $\alpha$ is the learning rate,

This method looks at every example in the entire training set

Stochastic gradient descent works very well. The sum above is "replaced" by a loop over the training examples, so that the update becomes:

for $i = 1$ to $m$:
                $$\theta_{j} := \theta_{j} + \alpha (y(i) - \theta_{T}x(i)x_{j}(i))$$


Linear regression: recall that under mild assumptions, the explicit solution for the ordinary least squares can be written explicitly as:
                $$\theta^{*} = (X^TX)^{-1}X^{T}Y$$

where the linear model is written in matrix form $Y = X\theta + \epsilon$, with $Y = (y(1),...y(m)) \in \mathbb{R}^{m \times d}$



In the context of linear regression, the log-likelihood is often associated with the assumption of normally distributed errors. The typical formulation assumes that the response variable follows a normal distribution with a mean determined by the linear regression model. Here's how you can express the log-likelihood for a simple linear regression model:

Assuming the response variable $(y_i)$ for each observation is normally distributed with mean $(\mu_i)$ and constant variance $(\sigma^2)$, the likelihood function for the observed data $(y_i)$ given the linear regression model is:

$$ L(\beta_0, \beta_1, \sigma^2 \mid x_i) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mu_i)^2}{2\sigma^2}\right) $$

where $(\mu_i = \beta_0 + \beta_1 x_i)$ is the mean predicted by the linear regression model for the $(i)$-th observation.

The log-likelihood for the entire dataset $(\{y_1, y_2, \ldots, y_n\})$ is the sum of the log-likelihood contributions from each observation:

$$ \mathcal{L}(\beta_0, \beta_1, \sigma^2 \mid x_i) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mu_i)^2 $$

Here:
- $(\beta_0)$ and $(\beta_1)$ are the coefficients of the linear regression model.
- $(n)$ is the number of observations.

The goal in linear regression is often to maximize this log-likelihood function, which is equivalent to minimizing the sum of squared residuals (ordinary least squares approach).

Note that in practice, maximizing the log-likelihood is often done under the assumption that the errors $(y_i - \mu_i)$ are normally distributed, which allows for the use of maximum likelihood estimation (MLE). This assumption is a key aspect of classical linear regression.


Now let's take a look at logistic regression.

A natural (generalized) linear model for binary classification:

$$ p_{\theta}(y=1 | x) = \sigma(\theta^{T}x)$$
$$ p_{\theta}(y=0 | x) = 1 - \sigma(\theta^{T}x)$$

where $\sigma(z) = \frac{1}{1+\epsilon^{-z}}$ is the sigmoid function (or logistic function).

# insert the graph here TODO

The compact formula is $p_{\theta}(y|x) = \sigma(\theta^{T}x)^y(1 - \theta(\theta^{T}x))^{(1-y)}$

Logistic Regression:

$$ L(\theta) = \prod_{i=1}^{m} \sigma(\theta^{T}x(i)^{y(i)}) $$

There is no closed form formula for $argmax \mathcal(\theta)$ so that we need now to use iterative algorithms. The gradient of the log likelihood is:

$$\frac{\partial}{\partial_{j}} \mathcal{l}(\theta) = \sum_{i=1}^{m}(y(i) - \sigma(\theta^Tx(i))x_{j}(i)$$

where we used the fact that $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

Now we will break down the binary cross entropy loss function the `torch.nn.BCELoss` computes Binary Cross Entropy between the target $y = (y(1),...,y(m)) \in {0, 1}^{m}$ and the output $z = (z(1),...,z(m)) \in [0,1]^{m}$ as follows:

$$ \text{loss}(i) = -[\text{y}(i)\text{log} \text{z}(i) + (1 - \text{y}(i)) \text{log}(1 - \text{z}(i))]$$

$$ \text{BCELoss}(z,y) = \frac{1}{m} \sum_{i=1}^{m} \text{loss}(i)$$

In summary, we get

$$\text{BCEWithLogitsLoss}(z,y) = \text{BCELoss}(\sigma(z),y)$$

The version is more numerically stable.

Note the default $1/m$, where $m$ is typically the size of the batch. This factor will be directly multiplied with the learning rate. Recall the batch gradient descent update:

$$\theta_{j} := \theta_{j} + \alpha \frac{\partial}{\partial \theta_{j}} \text{loss}(\theta) $$

Softmax Regression: now we have $c$ classes and for a training example (x,y), the quantity $\theta_{k}^{T}x$ should be related to the probability for the target $y$ to belong to class $k$. By analogy with the binary case, we assume:

$$ \text{log} p_{\theta}(y = k | x) \approx \theta_{k}^{T}x, \text{for all} k = 1,...,c.$$

as a consequence, we have with $\theta = (\theta_{1},..,\theta_{c}) \in \mathbb{R}^{(dxc)}$:

$$p_{\theta}(y = k | x) = \frac{e^{\theta_{k}^{T}x}}{\sum_{l} e^{\theta_{l}^{T}}x}$$

and we can write it in vector form:

$$(p_{\theta}(y = k | x)_{k=1,...,c} = softmax(\theta_{1}^{T}x),...,\theta_{c}^{T}x)$$

where the sigmoid function is applied componentwise.

For the logistic regression, we had only one parameter $\theta$ whereas here, for two classes we have two parameters: $\theta_{1}$ and $\theta_{2}$.

For 2 classes, we recover the logistic regression:

$$ p_{\theta} ( y = 1 | x) = \frac{e_{1}^{T}x}{e_{1}^{T}x + e_{0}^{T}x} $$
$$                         = \frac{1}{1 + e^{\theta_{0}^{T} - \theta_{1}^{T}x}}$$


Classification and softmax regression:

For the softmax regression, the log-likelihood can be written as:

$$ \mathcal{l}(\theta) = \sum_{i=1}^{m} \sum_{k=1}^{c}(y(i)=k)\text{log}\left( \frac{e^{\theta_{k}^{T}}x(i)}{\sum_{l} e^{\theta_{l}^{T}}x(i)} \right)$$

$$ = \sum_{i=1}^{T} \text{log softmax}_{y(i)}(\theta_{1}^{T} ,..., \theta_{c}^{T}x(i)) $$

In PyTorch, if the last layer of your network is a LogSoftmax() function, then you can od a softmax regression with the `torch.nn.NLLoss()`.

<!-- {{yt_tsp 0 0 Recap}}
{{yt_tsp 145 0 How to choose your loss?}}
{{yt_tsp 198 0 A probabilistic model for linear regression}}
{{yt_tsp 470 0 Gradient descent, learning rate, SGD}}
{{yt_tsp 690 0 Pytorch code for gradient descent}}
{{yt_tsp 915 0 A probabilistic model for logistic regression}}
{{yt_tsp 1047 0 Notations (information theory)}}
{{yt_tsp 1258 0 Likelihood for logistic regression}}
{{yt_tsp 1363 0 BCELoss}}
{{yt_tsp 1421 0 BCEWithLogitsLoss}}
{{yt_tsp 1537 0 Beware of the reduction parameter}}
{{yt_tsp 1647 0 Softmax regression}}
{{yt_tsp 1852 0 NLLLoss}}
{{yt_tsp 2088 0 Classification in pytorch}}
{{yt_tsp 2196 0 Why maximizing accuracy directly is hard?}}
{{yt_tsp 2304 0 Classification in deep learning}}
{{yt_tsp 2450 0 Regression without knowing the underlying model}}
{{yt_tsp 2578 0 Overfitting in polynomial regression}}
{{yt_tsp 2720 0 Validation set}}
{{yt_tsp 2935 0 Notion of risk and hypothesis space}}
{{yt_tsp 3280 0 estimation error and approximation error}} -->

<!-- ## Slides and Notebook

- [slides](https://dataflowr.github.io/slides/module3.html)
- [notebook](https://github.com/dataflowr/notebooks/blob/master/Module3/03_polynomial_regression.ipynb) in [colab](https://colab.research.google.com/github/dataflowr/notebooks/blob/master/Module3/03_polynomial_regression.ipynb) An explanation of underfitting and overfitting with polynomial regression. -->

## Minimal Working Examples

### [`BCELoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss)
```python
import torch.nn as nn
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3,4,5)
target = torch.randn(3,4,5)
loss(m(input), target)
```

### [`NLLLoss`](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) and [`CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
```python
import torch.nn as nn
m = nn.LogSoftmax(dim=1)
loss1 = nn.NLLLoss()
loss2 = nn.CrossEntropyLoss()
C = 8
input = torch.randn(3,C,4,5)
target = torch.empty(3,4,5 dtype=torch.long).random_(0,C)
assert loss1(m(input),target) == loss2(input,target)
```

<!-- ## Quiz

To check you know your loss, you can do the [quizzes](https://dataflowr.github.io/quiz/module3.html)
 -->
