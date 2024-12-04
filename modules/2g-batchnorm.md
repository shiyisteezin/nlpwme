@def sequence = ["batchnorm"]

# Batchnorm or Batch Renormalization


**Table of Contents**

\toc


## Batchnorm

Machine learning enthusiasts are looking for smarter ways to craft [weight initialization]() techniques. Batch re-normalization was introduced to force the activation statistics during the forward pass by re-normalizing the initial weights.

To quote,

"Training Deep Neural Network is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization..."

### Data Normalization

Weight initialization strategies aim to preserve the activation variance constant across layers, under the initial assumption that the input feature variances are the same.

Normalization, specifically standardizing input data feature-wise, is a common preprocessing step in machine learning to ensure that different features have comparable scales. This process helps algorithms converge faster and makes models less sensitive to the scale of input features. Let's break down the steps involved in standardizing input data feature-wise:

### Step 1: Understand the Need for Normalization

Different features in your dataset might have different scales, which can lead to issues when training machine learning models. For example, features with larger scales might dominate the learning process, making it difficult for the model to effectively use information from features with smaller scales. Normalization addresses this by scaling all features to have similar ranges.

### Step 2: Compute Mean and Standard Deviation for Each Feature

For each feature, calculate its mean )$\mu$) and standard deviation )$\sigma$) from the entire dataset. These values will be used to standardize the data.

$$ \mu_i = \frac{1}{m} \sum_{j=1}^{m} x_{ij} $$
$$ \sigma_i = \sqrt{\frac{1}{m} \sum_{j=1}^{m} (x_{ij} - \mu_i)^2} $$

Here, $(x_{ij})$ represents the value of feature $(i)$ for the $(j)$-th data point, and $(m)$ is the number of data points.

### Step 3: Standardize Each Feature

For each feature $(i)$ and each data point $(j($, apply the standardization formula:

$$ x'_{ij} = \frac{x_{ij} - \mu_i}{\sigma_i} $$

Here, $(x'_{ij})$ is the standardized value of feature $(i)$ for the $(j)$-th data point.

### Step 4: Understand the Result

After normalization, each feature will have a mean of 0 and a standard deviation of 1. This ensures that all features are centered around zero and have a comparable scale.

### Step 5: Apply Normalization During Training and Inference

When training a machine learning model, normalize the input features using the computed mean and standard deviation from the training set. During inference, use the same mean and standard deviation values for normalization.

### Example in Python (Using Scikit-Learn)

```python
from sklearn.preprocessing import StandardScaler

# Assuming X is your input data matrix
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
```

In this example, `X_normalized` will contain the standardized version of your input data.

Normalization is an essential preprocessing step, especially when using algorithms that are sensitive to the scale of input features, such as gradient-based optimization methods in neural networks or support vector machines.

Or in a more simplified notation, we see in visualization format that,

![](../extras/batch_norm/stndrd_scaling.png)

### Batch Normalization Process

 Below explains the batch normalization process used in machine learning, specifically in the context of neural networks. Here's a step-by-step breakdown:

- **Batch Normalization**: A technique to improve the speed, performance, and stability of artificial neural networks.

- **Mini-batch**: A subset of the training data, denoted by $\mathbf{u}_b \in \mathbb{R}^D$ for $b = 1, \ldots, B$, where $B$ is the batch size and $D$ is the number of features.

- **Mean and Variance Calculation**:
  - Mean: $\hat{\mu}_{\text{batch}} = \frac{1}{B} \sum_{b=1}^B \mathbf{u}_b$
  - Variance: $\hat{\sigma}^2_{\text{batch}} = \frac{1}{B} \sum_{b=1}^B (\mathbf{u}_b - \hat{\mu}_{\text{batch}})^2$

- **Component-wise Normalization**: The input $\mathbf{u}_b$ is normalized to $\hat{\mathbf{u}}_b$ using the computed mean and variance:
  - $\hat{\mathbf{u}}_b = \frac{(\mathbf{u}_b - \hat{\mu}_{\text{batch}})}{\sqrt{\hat{\sigma}^2_{\text{batch}} + \epsilon}}$

- **Standardization**: The normalized value $\hat{\mathbf{u}}_b$ is then linearly transformed to $y_b$:
  - $y_b = \gamma \odot \hat{\mathbf{u}}_b + \beta$

- **Parameters**: $\gamma$ and $\beta$ are learnable parameters of the model, and $y_b$ is the output after applying batch normalization.

This process is used to normalize the inputs of each layer so that they have a mean of zero and a variance of one, which helps to stabilize the learning process and reduce the number of training epochs required. The $\epsilon$ is a small constant added for numerical stability to avoid division by zero.

### Performing Back-Propagation for Batch Normalization at Inference/Runtime


Performing backpropagation with batch normalization involves computing gradients with respect to the input, scale (gamma), and shift (beta) parameters. Let's break down the steps for backpropagation through a component-wise affine transformation, considering batch normalization.

Assume you have input $( x )$, scale parameter $( \gamma )$, shift parameter $( \beta )$, normalized input $( \hat{x} )$, and batch statistics $( \mu )$ (mean) and $( \sigma^2 )$ (variance).

### Forward Pass

#### Input Transformation
   \[ \text{Affine Transformation } z = \gamma \hat{x} + \beta \]
   - $( z )$ is the output of the component-wise affine transformation.

### Backward Pass

#### Gradients with Respect to $( z )$
\[ \frac{\partial L}{\partial z} \]
- Compute the gradient of the loss $( L )$ with respect to $( z )$.

#### Gradients with Respect to $( \gamma )$ and $( \beta )$
\[ \frac{\partial L}{\partial \gamma} = \sum \frac{\partial L}{\partial z} \cdot \hat{x} \]
\[ \frac{\partial L}{\partial \beta} = \sum \frac{\partial L}{\partial z} \]

#### Gradients with Respect to $( \hat{x} )$
\[ \frac{\partial L}{\partial \hat{x}} = \frac{\partial L}{\partial z} \cdot \gamma \]

#### Gradients with Respect to $( \sigma^2 )$ and $( \mu )$
\[ \frac{\partial L}{\partial \sigma^2} = \sum \frac{\partial L}{\partial z} \cdot (\hat{x} - \mu) \cdot \frac{-1}{2} \cdot (\sigma^2 + \epsilon)^{-\frac{3}{2}} \]
\[ \frac{\partial L}{\partial \mu} = \sum \frac{\partial L}{\partial z} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\sum -2 (\hat{x} - \mu)}{m} \]

#### Gradients with Respect to $( x )$
\[ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial \hat{x}} \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{2 (\hat{x} - \mu)}{m} + \frac{\partial L}{\partial \mu} \cdot \frac{1}{m} \]

### Update Parameters
Update the parameters using the computed gradients during backpropagation.

```python
# Assuming gamma, beta, x, mean, variance, and gradient_loss are known
# epsilon is a small constant for numerical stability

# Gradients
dL_dz = gradient_loss
dL_dgamma = np.sum(dL_dz * normalized_x, axis=0)
dL_dbeta = np.sum(dL_dz, axis=0)
dL_dnormalized_x = dL_dz * gamma

# Gradients for variance and mean
dL_dvariance = np.sum(dL_dz * (x - mean) * (-1 / 2) * (variance + epsilon)**(-3 / 2), axis=0)
dL_dmean = np.sum(dL_dz * (-1 / np.sqrt(variance + epsilon)), axis=0) + dL_dvariance * np.sum(-2 * (x - mean)) / len(x)

# Gradient for x
dL_dx = dL_dnormalized_x / np.sqrt(variance + epsilon) + dL_dvariance * 2 * (x - mean) / len(x) + dL_dmean / len(x)

# Update gamma and beta
gamma -= learning_rate * dL_dgamma
beta -= learning_rate * dL_dbeta

# Update other parameters as needed (e.g., in the case of an optimizer)
```

Note: This is a simplified explanation, and actual implementations might involve additional considerations, such as the choice of optimizer, learning rate scheduling, and the specific architecture of your neural network.

### In The Context of A Batch Normalized Set of Weights


Let's break down the components of this expression:

- $( y )$: The output of the batch normalization layer.
- $( \gamma )$: Scale parameter.
- $( u )$: The input to the batch normalization layer.
- $( \mu_{\text{stat}} )$: Batch mean.
- $( \sigma_{\text{stat}}^2 )$: Batch variance.
- $( \epsilon )$: A small constant for numerical stability.
- $( \beta )$: Shift parameter.

Here's the breakdown:

\[ y = \gamma \odot \left( \frac{u - \mu_{\text{stat}}}{\sqrt{\sigma_{\text{stat}}^2 + \epsilon}} \right) + \beta \]

- $( \odot )$ represents element-wise (component-wise) multiplication.
- $( \frac{u - \mu_{\text{stat}}}{\sqrt{\sigma_{\text{stat}}^2 + \epsilon}} )$: Normalizing the input $( u )$ by subtracting the batch mean $( \mu_{\text{stat}} )$ and dividing by the square root of the batch variance $( \sigma_{\text{stat}}^2 + \epsilon )$ for numerical stability.
- $( \gamma \odot )$: Scaling the normalized input by the learnable scale parameter $( \gamma )$.
- $( + \beta )$: Shifting the scaled and normalized input by the learnable shift parameter $( \beta )$.

This formulation is characteristic of batch normalization, a technique used to improve the training of deep neural networks by normalizing the input of each layer. The scale and shift parameters $( \gamma )$ and $( \beta )$ are learnable parameters that allow the model to adjust the normalization based on the data.

During training, $( \mu_{\text{stat}} )$ and $( \sigma_{\text{stat}}^2 )$ are computed based on the statistics of the current mini-batch, and during inference, running averages of these statistics are typically used.


## The Comparison Between These Two Scenarios

The expression you've provided is actually very similar to the general form of the batch normalization layer that I explained earlier. Let's break down the similarities and differences:

Your expression:
\[ y = \gamma \odot \left( \frac{u - \mu_{\text{stat}}}{\sqrt{\sigma_{\text{stat}}^2 + \epsilon}} \right) + \beta \]

General batch normalization expression:
\[ z = \gamma \hat{x} + \beta \]

Here, $( y )$ and $( z )$ play the same role as the output of the batch normalization layer. Both expressions involve scaling $( \gamma )$ and shifting $( \beta )$ the normalized input. Let's break down the terms:

- $( \frac{u - \mu_{\text{stat}}}{\sqrt{\sigma_{\text{stat}}^2 + \epsilon}} )$: This is the normalized input, which is the result of subtracting the batch mean $( \mu_{\text{stat}} )$ and dividing by the square root of the batch variance $( \sigma_{\text{stat}}^2 + \epsilon )$.

- $( \gamma )$: This is the scale parameter, which is multiplied element-wise with the normalized input.

- $( \beta )$: This is the shift parameter, which is added element-wise to the scaled and normalized input.

The key differences are in the notation used, but conceptually, they represent the same idea of normalizing the input, scaling it, and then shifting it. Both formulations are part of the batch normalization process in neural networks, where the goal is to stabilize and speed up training by normalizing the input of each layer. The specific notation and parameter names might vary, but the underlying principles are consistent.



<!--
{{youtube_placeholder batchnorm}} -->


<!-- ## Slides and Notebook

- [slides](https://abursuc.github.io/slides/polytechnique/14-04-batchnorm.html#1)
- [notebook](https://github.com/dataflowr/notebooks/blob/master/Module16/16_batchnorm_simple.ipynb)  -->
