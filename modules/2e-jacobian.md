@def sequence = ["jax"]

# Automatic Differentiation: More on Linear Mapping and Jacobian Matrices

**Table of Contents**

\toc

# Autodiff and Backpropagation


## The Jacobian Matrix

Below is the Jacobian matrix of the vector-valued function $( \mathbf{f}(\mathbf{x}) )$ expressed in terms of partial derivatives and gradients. Let's break down each part:

1. **Jacobian Matrix Definition:**
   - The Jacobian matrix $( J_{\mathbf{f}}(\mathbf{x}) )$ is defined as an $( m \times n )$ matrix, where each entry represents the partial derivative of a component of $( \mathbf{f} )$ with respect to a variable $( x_i )$.

\[\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = J_{\mathbf{f}}(\mathbf{x}) = \left( \begin{array}{ccc}
\frac{\partial f_1}{\partial x_1}&\dots& \frac{\partial f_1}{\partial x_n}\\
\vdots&&\vdots\\
\frac{\partial f_m}{\partial x_1}&\dots& \frac{\partial f_m}{\partial x_n}
\end{array}\right)\]

2. **Vectorized Form:**
   - The Jacobian matrix can also be expressed in a vectorized form by stacking the gradients of the individual components of $( \mathbf{f} )$.

\[\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = J_{\mathbf{f}}(\mathbf{x}) = \left( \frac{\partial \mathbf{f}}{\partial x_1}, \dots, \frac{\partial \mathbf{f}}{\partial x_n} \right)\]

3. **Gradient Representation:**
   - Each row of the Jacobian matrix corresponds to the transpose of the gradient of a component of $( \mathbf{f} )$ with respect to $( \mathbf{x} )$.

\[\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = J_{\mathbf{f}}(\mathbf{x}) = \left( \begin{array}{c}
\nabla f_1(\mathbf{x})^T\\
\vdots\\
\nabla f_m(\mathbf{x})^T
\end{array}\right)\]

In summary, the Jacobian matrix provides a linear approximation of the function $( \mathbf{f}(\mathbf{x}) )$ near the point $( \mathbf{x} )$. It captures the sensitivity of each component of $( \mathbf{f} )$ to small changes in the variables $( x_i )$. The vectorized form and the representation using gradients offer different perspectives on the same mathematical concept.

## Taylor Expansion

The above expressions involve concepts that are related to the Taylor expansion. Let me clarify the connection between the provided expression and the Taylor expansion.

The Taylor expansion of a function $( \mathbf{f}(\mathbf{x}) )$ around a point $( \mathbf{x}_0 )$ is given by:

\[ \mathbf{f}(\mathbf{x}_0 + \mathbf{v}) = \mathbf{f}(\mathbf{x}_0) + J_{\mathbf{f}}(\mathbf{x}_0)\mathbf{v} + o(\|\mathbf{v}\|) \]

Now, let's relate this to the provided expression:

\[ J_{\mathbf{f}}(\mathbf{x}) = \left( \frac{\partial \mathbf{f}}{\partial x_1}, \dots, \frac{\partial \mathbf{f}}{\partial x_n} \right) \]

This expression represents the Jacobian matrix $( J_{\mathbf{f}}(\mathbf{x}) )$ as a row vector of partial derivatives. The Taylor expansion involves the Jacobian matrix $( J_{\mathbf{f}}(\mathbf{x}_0) )$ (similar to $( J_{\mathbf{f}}(\mathbf{x}) )$), the perturbation vector $( \mathbf{v} )$ (similar to $( \Delta \mathbf{x} )$), and the higher-order terms $( o(\|\mathbf{v}\|) )$ (similar to $( o(h) )$).

To show the connection more explicitly, consider a small perturbation $( \Delta \mathbf{x} )$ around the point $( \mathbf{x}_0 )$:

\[ \mathbf{f}(\mathbf{x}_0 + \Delta \mathbf{x}) = \mathbf{f}(\mathbf{x}_0) + J_{\mathbf{f}}(\mathbf{x}_0)\Delta \mathbf{x} + o(\|\Delta \mathbf{x}\|) \]

Here, $( J_{\mathbf{f}}(\mathbf{x}_0) )$ is the Jacobian matrix evaluated at $( \mathbf{x}_0 )$, and $( \Delta \mathbf{x} )$ is the perturbation vector. This is a form of the Taylor expansion, where the first term is the function value at $( \mathbf{x}_0 )$, the second term is the linear approximation (Jacobian matrix multiplied by the perturbation), and the third term represents higher-order terms.

So, while the provided expression itself is not the Taylor expansion, it involves the concept of partial derivatives and gradients, which are fundamental to understanding and deriving the Taylor expansion of a multivariate function.

Hence the Jacobian $J_{\mathbf{f}}(\mathbf{x})\in \mathbb{R}^{m\times n}$ is a linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$ such that for $\mathbf{x},\mathbf{v} \in \mathbb{R}^n$ and $h\in \mathbb{R}$:
\begin{align*}
\mathbf{f}(\mathbf{x}+h\mathbf{v}) = \mathbf{f}(\mathbf{x}) + hJ_{\mathbf{f}}(\mathbf{x})\mathbf{v} +o(h).
\end{align*}
The term $J_{\mathbf{f}}(\mathbf{x})\mathbf{v}\in \mathbb{R}^m$ is a Jacobian Vector Product (**JVP**), corresponding to the interpretation where the Jacobian is the linear map: $J_{\mathbf{f}}(\mathbf{x}):\mathbb{R}^n \to \mathbb{R}^m$, where $J_{\mathbf{f}}(\mathbf{x})(\mathbf{v})=J_{\mathbf{f}}(\mathbf{x})\mathbf{v}$.

The last part is emphasizing the interpretation of the Jacobian matrix $(J_{\mathbf{f}}(\mathbf{x}))$ as a linear map that transforms vectors. The expression $(J_{\mathbf{f}}(\mathbf{x})\mathbf{v})$ represents the result of applying this linear map to the vector $(\mathbf{v})$.

Here's a breakdown:

- $(J_{\mathbf{f}}(\mathbf{x}))$ is the Jacobian matrix, which contains the partial derivatives of each component of the vector-valued function $(\mathbf{f}(\mathbf{x}))$ with respect to each element of $(\mathbf{x})$. It has dimensions $(m \times n)$.

- $(\mathbf{v})$ is a vector in $(\mathbb{R}^n)$.

- The product $(J_{\mathbf{f}}(\mathbf{x})\mathbf{v})$ represents the result of multiplying the Jacobian matrix by the vector $(\mathbf{v})$.

- The result is a vector in $(\mathbb{R}^m)$, and it can be interpreted as the change in the output of the function $(\mathbf{f}(\mathbf{x}))$ due to a small change $(\mathbf{v})$ in the input.

So, in summary, $(J_{\mathbf{f}}(\mathbf{x})\mathbf{v})$ is a Jacobian vector product (JVP) that quantifies the linear transformation of the input vector $(\mathbf{v})$ by the Jacobian matrix $(J_{\mathbf{f}}(\mathbf{x}))$. This interpretation is in line with the understanding of the Jacobian as a linear map from $(\mathbb{R}^n)$ to $(\mathbb{R}^m)$.

Above explains a mathematical representation of the first-order Taylor expansion of the function $( \mathbf{f}(\mathbf{x}) )$ around the point $( \mathbf{x} )$. Let's break down the key components and understand why this expansion is valid:

1. **Taylor Expansion:**
   - The Taylor expansion of a function $( \mathbf{f}(\mathbf{x}) )$ around a point $( \mathbf{x}_0 )$ is given by:
     \[ \mathbf{f}(\mathbf{x}_0 + \mathbf{v}) = \mathbf{f}(\mathbf{x}_0) + J_{\mathbf{f}}(\mathbf{x}_0)\mathbf{v} + o(\|\mathbf{v}\|) \]
   - Here, $( J_{\mathbf{f}}(\mathbf{x}_0) )$ is the Jacobian matrix of $( \mathbf{f} )$ at the point $( \mathbf{x}_0 )$, and $( o(\|\mathbf{v}\|) )$ represents a term that goes to zero faster than $( \|\mathbf{v}\| )$ as $( \|\mathbf{v}\| )$ approaches zero.

2. **Jacobian Matrix $( J_{\mathbf{f}}(\mathbf{x}) )$:**
   - The Jacobian matrix $( J_{\mathbf{f}}(\mathbf{x}) )$ represents the linearization of the function $( \mathbf{f} )$ around the point $( \mathbf{x} )$. It contains the partial derivatives of each component of $( \mathbf{f} )$ with respect to each variable $( x_i )$.

3. **Linear Map Interpretation:**
   - The Jacobian matrix $( J_{\mathbf{f}}(\mathbf{x}) )$ can be viewed as a linear map that transforms small changes in the input vector $( \mathbf{v} )$ to small changes in the output vector $( \mathbf{f} )$.
   - The linear approximation $( J_{\mathbf{f}}(\mathbf{x})\mathbf{v} )$ represents the change in $( \mathbf{f} )$ resulting from a small change $( \mathbf{v} )$ in the input.

4. **Higher-Order Terms:**
   - The term $( o(\|\mathbf{v}\|) )$ represents the higher-order terms in the Taylor expansion that capture the behavior beyond the linear approximation. As $( \|\mathbf{v}\| )$ approaches zero, these terms become negligible compared to the linear term.

In summary, the expression you provided is a way to approximate the function $( \mathbf{f} )$ near the point $( \mathbf{x} )$ using a linear map (the Jacobian) and accounting for higher-order terms that become negligible as the input perturbation $( \mathbf{v} )$ becomes small. This is a fundamental concept in calculus and optimization, providing a local linear approximation to a function.

## Chain composition

The gradient of the loss function with respect to the parameters is computed in machine learning. In particular, if the parameters are high-dimensional, the loss is a real number. Hence, consider a real-valued function $\mathbf{f}:\mathbb{R}^n\stackrel{\mathbf{g}_1}{\to}\mathbb{R}^m \stackrel{\mathbf{g}_2}{\to}\mathbb{R}^d\stackrel{h}{\to}\mathbb{R}$, so that $\mathbf{f}(\mathbf{x}) = h(\mathbf{g}_2(\mathbf{g}_1(\mathbf{x})))\in \mathbb{R}$. We have
\begin{align*}
\underbrace{\nabla\mathbf{f}(\mathbf{x})}_{n\times 1}=\underbrace{J_{\mathbf{g}_1}(\mathbf{x})^T}_{n\times m}\underbrace{J_{\mathbf{g}_2}(\mathbf{g}_1(\mathbf{x}))^T}_{m\times d}\underbrace{\nabla h(\mathbf{g}_2(\mathbf{g}_1(\mathbf{x})))}_{d\times 1}.
\end{align*}

In order to perform this computation, if we begin at the right and work our way down to a vector (of size $m$), we will need to perform $O(nm+md)$ operations to create another matrix times a vector. $O(nmd+nd)$ operations result from performing matrix-matrix multiplication from the left. Thus, it is evident that beginning from the right is far more efficient as soon as $m\approx d$. It should be noted, however, that in order to do the computation from right to left, the values of $\mathbf{g}_1(\mathbf{x})\in\mathbb{R}^m$ and $\mathbf{x}\in\mathbb{R}^n$ must be kept in memory.

An effective method for calculating the gradient "from the right to the left," or backward, is **backpropagation**. Specifically, we will have to calculate amounts in the following format: This may be expressed as $\mathbf{u}^T J_{\mathbf{f}}(\mathbf{x})$, which is a Vector Jacobian Product (**VJP**), corresponding to the interpretation where the Jacobian is the linear map: $J_{\mathbf{f}}(\mathbf{x})^T\mathbf{u} \in \mathbb{R}^n$ with $\mathbf{u} \in\mathbb{R}^m$. Composed with the linear map $J_{\mathbf{f}}(\mathbf{x}):\mathbb{R}^n \to \mathbb{R}^m$ In order for $\mathbf{u}^TJ_{\mathbf{f}}(\mathbf{x}) = \mathbf{u} \circ J_{\mathbf{f}}(\mathbf{x})$ to be true, $\mathbf{u}:\mathbb{R}^m\to \mathbb{R}$.


### Jacobian Vector Product

The Jacobian-vector product (JVP) is a concept in calculus and linear algebra, particularly relevant in the context of automatic differentiation and optimization. It involves computing the product of the Jacobian matrix of a function with a given vector. The Jacobian matrix represents the partial derivatives of a vector-valued function with respect to its input variables, and the JVP allows you to efficiently compute the effect of a small perturbation in the input space on the function's output.

For a function $( f: \mathbb{R}^n \rightarrow \mathbb{R}^m )$, the Jacobian matrix $( J )$ is an $( m \times n )$ matrix where each entry $( J_{ij} )$ is the partial derivative of the $( i )$-th output with respect to the $( j )$-th input.

The Jacobian-vector product is computed as follows: Given a vector $( v )$ in the output space $( \mathbb{R}^m )$, the JVP of $( f )$ with respect to $( v )$ at a point $( x )$ in the input space is denoted as $( J(v) )$ and is computed as:

\[ J(v) = J \cdot v \]

Mathematically, the JVP is the matrix-vector product of the Jacobian matrix $( J )$ and the vector $( v )$. The resulting vector $( J(v) )$ represents the directional derivative of the function $( f )$ at the point $( x )$ in the direction of the vector $( v )$.

In the context of automatic differentiation libraries like JAX, which supports JVP, this concept is essential for efficiently calculating gradients and optimizing functions. JVPs are particularly useful when dealing with vectorized operations and optimization algorithms that require information about the direction and magnitude of changes in the function's output concerning changes in the input.

**Example:** let $\mathbf{f}(\mathbf{x}, W) = \mathbf{x} W\in \mathbb{R}^b$ where $W\in \mathbb{R}^{a\times b}$ and $\mathbf{x}\in \mathbb{R}^a$. We clearly have
$$
J_{\mathbf{f}}(\mathbf{x}) = W^T.
$$
Note that here, we are slightly abusing notations and considering the partial function $\mathbf{x}\mapsto \mathbf{f}(\mathbf{x}, W)$. To see this, we can write $f_j = \sum_{i}x_iW_{ij}$ so that
$$
\frac{\partial \mathbf{f}}{\partial x_i}= \left( W_{i1}\dots W_{ib}\right)^T
$$
Then recall from definitions that
$$
J_{\mathbf{f}}(\mathbf{x}) = \left( \frac{\partial \mathbf{f}}{\partial x_1},\dots, \frac{\partial \mathbf{f}}{\partial x_n}\right)=W^T.
$$
Now we clearly have
$$
J_{\mathbf{f}}(W) = \begin{bmatrix} \mathbf{x} \\ \vdots \\ \mathbf{x} \end{bmatrix}  $$
$$ \text{ since, } \mathbf{f}(\mathbf{x}, W+\Delta W) = \mathbf{f}(\mathbf{x}, W) + \mathbf{x} \Delta W.
$$
Note that multiplying $\mathbf{x}$ on the left is actually convenient when using broadcasting, i.e. we can take a batch of input vectors of shape $\text{bs}\times a$ without modifying the math above.

In short, what the above did was ,

explaining the concept of computing the gradient of a loss function with respect to parameters in machine learning. The text describes the mathematical process of calculating the gradient of a composite function, which is an essential part of training machine learning models, particularly neural networks. The process involves applying the chain rule to compute the derivatives of nested functions.

Key points from the above:

1. The function $f(x)$ is defined as the composition of three functions, $f: \mathbb{R}^n \xrightarrow{g_1} \mathbb{R}^m \xrightarrow{g_2} \mathbb{R}^d \xrightarrow{h} \mathbb{R}$.

2. The gradient $\nabla f(x)$ is computed using the chain rule as $\nabla f(x) = J_{g_1}(x)^TJ_{g_2}(g_1(x))^T\nabla h(g_2(g_1(x)))$, where $J_{g_1}(x)$ and $J_{g_2}(g_1(x))$ are Jacobian matrices of the functions $g_1$ and $g_2$ at respective points.

3. It mentions that computing the gradient from right to left, also known as backpropagation, is more efficient, especially when the dimensions $m$ and $d$ are approximately equal.

4. The concept of Vector Jacobian Product (VJP) is introduced, which is expressed as $u^TJ_f(x)$ and allows for efficient computation of the gradient.

5. An example is provided with a function $f(x, W) = xW$ for vectors $x \in \mathbb{R}^a$ and matrices $W \in \mathbb{R}^{a \times b}$, where the Jacobian of $f$ with respect to $x$ is $J_f(x) = W^T$.

6. The text also explains a slight abuse of notation when considering the partial function $x \mapsto f(x, W)$, where the gradient of each component of the function with respect to $x_i$ is represented by the column vector of $W$.

7. Finally, it points out the convenience of multiplying by $x$ on the left when using broadcasting, which is a technique used in programming to perform operations on arrays of different shapes.


## Implementation

`torch.autograd` in PyTorch offers functions and classes that implement automated differentiation of any arbitrary scalar-valued function. Use the `forward()` and `backward()` static methods to construct a custom [autograd.Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function) by subclassing this class. Here's an illustration:

```python
class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result
# Use it by calling the apply method:
output = Exp.apply(input)
```
### Backprop The Functional Way

Here we will implement in `numpy` a different approach mimicking the functional approach of [JAX](https://jax.readthedocs.io/en/latest/index.html) see [The Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#).

Two arguments will be required for each function: the parameters {w} and the input {x}. In order to get $J_{\mathbf{f}}(\mathbf{x})$ and $J_{\mathbf{f}}(\mathbf{w})$, we construct two **vjp** functions for each function, each of which takes a gradient $\mathbf{u}$ as an argument. These functions then return $J_{\mathbf{f}}(\mathbf{x})^T \mathbf{u}$ and $J_{\mathbf{f}}(\mathbf{w})^T \mathbf{u}$, respectively. In summary, for $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{w} \in \mathbb{R}^d$, and $\mathbf{f}(\mathbf{x},\mathbf{w}) \in \mathbb{R}^m$,
\begin{align*}
{\bf vjp}_\mathbf{x}(\mathbf{u}) &= J_{\mathbf{f}}(\mathbf{x})^T \mathbf{u}, \text{ with } J_{\mathbf{f}}(\mathbf{x})\in\mathbb{R}^{m\times n}, \mathbf{u}\in \mathbb{R}^m\\
{\bf vjp}_\mathbf{w}(\mathbf{u}) &= J_{\mathbf{f}}(\mathbf{w})^T \mathbf{u}, \text{ with } J_{\mathbf{f}}(\mathbf{w})\in\mathbb{R}^{m\times d}, \mathbf{u}\in \mathbb{R}^m
\end{align*}
Then backpropagation is simply done by first computing the gradient of the loss and then composing the **vjp** functions in the right order.

The expressions and functions described in the provided approach using `numpy` mimic the functionality of JAX's automatic differentiation. Let's break down the key components and their meanings:

1. **Function $( \mathbf{f}(\mathbf{x}, \mathbf{w}) )$:**
   - This is the target function that takes two sets of parameters, $(\mathbf{x})$ and $(\mathbf{w})$, and produces an output.

2. **Jacobian-Vector Product Functions $({\bf vjp}_\mathbf{x}(\mathbf{u}))$ and $({\bf vjp}_\mathbf{w}(\mathbf{u}))$:**
   - These functions compute the Jacobian-vector product of the function $( \mathbf{f} )$ with respect to $(\mathbf{x})$ and $(\mathbf{w})$, respectively.
   - $({\bf vjp}_\mathbf{x}(\mathbf{u}))$ computes $(J_{\mathbf{f}}(\mathbf{x})^T \mathbf{u})$, where $(J_{\mathbf{f}}(\mathbf{x}))$ is the Jacobian matrix of $( \mathbf{f} )$ with respect to $(\mathbf{x})$.
   - $({\bf vjp}_\mathbf{w}(\mathbf{u}))$ computes $(J_{\mathbf{f}}(\mathbf{w})^T \mathbf{u})$, where $(J_{\mathbf{f}}(\mathbf{w}))$ is the Jacobian matrix of $( \mathbf{f} )$ with respect to $(\mathbf{w})$.

3. **Loss Function $( \text{loss\_function}(\mathbf{x}, \mathbf{w}) )$:**
   - This is a function that takes $(\mathbf{x})$ and $(\mathbf{w})$ as inputs and computes a scalar loss. The example provided uses a simple quadratic loss.

4. **Compute Gradients Function:**
   - The `compute_gradients` function is responsible for computing the gradients of the loss with respect to $(\mathbf{x})$ and $(\mathbf{w})$ using the Jacobian-vector product functions.
   - It initializes a gradient ($(\mathbf{u}_{\text{loss}})$) for the loss and then computes the Jacobian-vector products $({\bf vjp}_\mathbf{x}(\mathbf{u}_{\text{loss}}))$ and $({\bf vjp}_\mathbf{w}(\mathbf{u}_{\text{loss}}))$.

5. **Example Usage:**
   - The example usage demonstrates how to compute gradients for a specific $(\mathbf{x})$ and $(\mathbf{w})$ using the `compute_gradients` function.

In summary, this approach allows for the computation of gradients using the concept of Jacobian-vector products. By defining functions that mimic the behavior of JAX's `vjp` functions, you can perform backpropagation to compute gradients efficiently. The provided example is a simplified illustration, and in practice, this methodology can be extended to more complex functions and scenarios.


To implement the described approach using `numpy`, we can define functions that compute the Jacobian-vector products ($({\bf vjp}_\mathbf{x}(\mathbf{u}))$ and $({\bf vjp}_\mathbf{w}(\mathbf{u}))$) for a given function $(\mathbf{f}(\mathbf{x}, \mathbf{w}))$. Then, we can use these functions to perform backpropagation for a given loss.

Here's a simple example:

```python
import numpy as np

# Define the function f(x, w)
def my_function(x, w):
    return np.dot(x, w)

# Define the vjp functions
def vjp_x(u, x, w):
    return np.outer(u, w)

def vjp_w(u, x, w):
    return np.outer(u, x)

# Define a loss function
def loss_function(x, w):
    y = my_function(x, w)
    return np.sum(y**2)

# Compute the gradients using vjp functions
def compute_gradients(x, w):
    u_loss = np.ones_like(my_function(x, w))  # Gradient of the loss w.r.t. my_function(x, w)
    vjp_x_result = vjp_x(u_loss, x, w)
    vjp_w_result = vjp_w(u_loss, x, w)
    return vjp_x_result, vjp_w_result

# Example usage
x_val = np.array([1.0, 2.0, 3.0])
w_val = np.array([0.1, 0.2, 0.3])

grad_x, grad_w = compute_gradients(x_val, w_val)

print("Gradient with respect to x:", grad_x)
print("Gradient with respect to w:", grad_w)
```

In this example, `my_function` is a simple linear function, and `loss_function` is a quadratic loss. The `vjp_x` and `vjp_w` functions compute the Jacobian-vector products with respect to `x` and `w`, respectively. The `compute_gradients` function then uses these vjp functions to compute the gradients of the loss.

This is a basic illustration, and in practice, you might want to generalize these functions for more complex scenarios and functions.


<!-- ## Practice

- intro to JAX: autodiff the functional way [autodiff\_functional\_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/autodiff_functional_empty.ipynb) and its solution [autodiff\_functional\_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/autodiff_functional_sol.ipynb)
- Linear regression in JAX [linear\_regression\_jax.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/linear_regression_jax.ipynb) -->
