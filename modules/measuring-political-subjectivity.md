## Measuring Political Subjectivity with Variational Encoding Methods 

\toc


In socio-politics, quantified approaches and modeling techniques are applied in supporting and facilitating political analyses. Individuals, parties, committees and other political entities come together and try to push forward campaigns in hope to receive appropriate patrionization and support for their political agenda. 

The Political Action Committees (PACs or Super PACs) amass funding resources that could benefit the elections. These type of fundings could be from other individuals, or political entities. For the sole of purpose of understanding what the processes of fund raising activities like these really are, this part of the project explores the 2021-2022 PACs financial data.

This part of the project will first present the receipts, disbursements, and other expenditures in terms of propagating political actions in visualization format grounded in states; for example, how many different political action committees there are by US states. 

This part of the project will also break down all the candidates of 2022 their basic information as mentioned above including their basic demographics, political party affiliation, election cycle, and incumbency.

All info is retrievable through the Federal Election Commission's directory. This project seeks to conduct the research with full transparency and abide to relevant code of conduct.

## The Motivation behind This Project 

Measuring political sentiment and polarization is a common practice in the realm of social science research. However, it may also be applicable to solving business problems, like providing more information about a certain candidate to voters to fill the information gap and facilitate voting processes. 

This project tries to help someone who is interested in voting activities understand the political leaning of a candidate for federal elections. 

In this blog, the structure and construct of the model will be explained. Please check out this [repo](https://github.com/shiyis/c4fe-tbip) for a more comprehensive demo of the project and other complementary analysis. 

draws inspiration from website like [OpenSecrets](https://www.opensecrets.org/) and [this paper](https://arxiv.org/abs/2005.04232), where it strives to uncover information of a politician's agenda and activities (campaign-related or financial).

helps the general population who is interested in partaking in political activities understand a politician (or anyone who authors political content)'s leaning/stance by extracting crucial information from relevant political text. 

Website like [OpenSecrets](https://www.opensecrets.org/) provides valuable statistics and educational information to start. This project tries to top it off by retrieving organic information (Tweets) of said candidates and conduct analysis accordingly. 

## Model Implementation with NumPyro

```python
%%capture
%pip install numpyro==0.10.1
%pip install optax
```


```python
from scipy import sparse
import jax
import jax.numpy as jnp
import numpy as np

dataPath = "tbip/data/senate-speeches-114/clean/"

# Load data
author_indices = jax.device_put(
    jnp.load(dataPath + "author_indices.npy"), jax.devices("gpu")[0]
)

counts = sparse.load_npz(dataPath + "counts.npz")

with open(dataPath + "vocabulary.txt", "r") as f:
    vocabulary = f.readlines()

with open(dataPath + "author_map.txt", "r") as f:
    author_map = f.readlines()

author_map = np.array(author_map)

num_authors = int(author_indices.max() + 1)
num_documents, num_words = counts.shape
```


```python

pre_initialize_parameters = True
```

```python

# Fit NMF to be used as initialization for TBIP
from sklearn.decomposition import NMF

if pre_initialize_parameters:
    nmf_model = NMF(
        n_components=num_topics, init="random", random_state=0, max_iter=500
    )
    # Define initialization arrays
    initial_document_loc = jnp.log(
        jnp.array(np.float32(nmf_model.fit_transform(counts) + 1e-2))
    )
    initial_objective_topic_loc = jnp.log(
        jnp.array(np.float32(nmf_model.components_ + 1e-2))
    )
else:
    rng1, rng2 = random.split(rng_seed, 2)
    initial_document_loc = random.normal(rng1, 
                                        shape=(num_documents, num_topics))
    initial_objective_topic_loc = random.normal(rng2, 
                                        shape=(num_topics, num_words))
```


```python

# Fit NMF to be used as initialization for TBIP
from sklearn.decomposition import NMF

if pre_initialize_parameters:
    nmf_model = NMF(
        n_components=num_topics, init="random", random_state=0, max_iter=500
    )
    # Define initialization arrays
    initial_document_loc = jnp.log(
        jnp.array(np.float32(nmf_model.fit_transform(counts) + 1e-2))
    )
    initial_objective_topic_loc = jnp.log(
        jnp.array(np.float32(nmf_model.components_ + 1e-2))
    )
else:
    rng1, rng2 = random.split(rng_seed, 2)
    initial_document_loc = random.normal(rng1, shape=(num_documents, num_topics))
    initial_objective_topic_loc = random.normal(rng2, shape=(num_topics, num_words))

```

The results are inferred using variational inference with reparameterization gradients. 

It is intractable to evaluate the posterior distribution, so we approximate the posterior with a distribution. How do we set the values? We want to minimize the KL-Divergence between and the posterior, which is equivalent to maximizing the ELBO:

Sure, here is the LaTeX representation of the Evidence Lower Bound (ELBO) and the Kullback-Leibler (KL) divergence:

$$
\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x, z) - \log q(z|x)]
$$

$$
\text{KL}(q(z|x)||p(z)) = \mathbb{E}_{q(z|x)}[\log q(z|x) - \log p(z)]
$$

In these equations:
- $q(z|x)$ represents the approximate posterior distribution over latent variables $z$ given input data $x$ 
- $p(x, z)$ is the joint distribution of the observed data $x$ and the latent variables $z$ - it's essentially the likelihood of generating the observed documents given the latent variables - it quantifies how likely it is to see a particular set of documents along with their associated latent representations.
- $p(z)$ is the prior distribution over latent variables - in the context of document clustering, it can represent the prior distribution of topics over documents, capturing assumptions about the distribution of topics in the dataset.
- The expectation $\mathbb{E}_{q(z|x)}[\cdot]$ is taken with respect to the approximate posterior $q(z|x)$.
- The ELBO is the lower bound on the log-likelihood of the observed data $x$, and maximizing it is equivalent to minimizing the KL divergence between the approximate posterior $q(z|x)$ and the true prior $p(z)$.


We set the variational family to be the mean-field family, meaning the latent variables factorize over documents, topics ,and authors :

$$ q_\phi(\theta, \beta, \eta, x) = \prod_{d,k,s} q(\theta_d)q(\beta_k)q(\eta_k)q(x_s) $$


We use lognormal factors for the positive variables and Gaussian factors for the real variables:

$$q(\theta_d) = \text{LogNormal}_K(\mu_{\theta_d}\sigma^2_{\theta_d})$$

$$q(\beta_k) = \text{LogNormal}_V(\mu_{\beta_k}, \sigma^2_{\beta_k})$$

$$q(\eta_k) = \mathcal{N}_V(\mu_{\eta_k}, \sigma^2_{\eta_k})$$

$$q(x_s) = \mathcal{N}(\mu_{x_s}, \sigma^2_{x_s}).$$


Thus, our goal is to maximize the ELBO with respect to  $$\phi = \{\mu_\theta, \sigma_\theta, \mu_\beta, \sigma_\beta,\mu_\eta, \sigma_\eta, \mu_x, \sigma_x\}$$


```python

In the cell below, we define the model and the variational family (guide).

from numpyro import plate, sample, param
import numpyro.distributions as dist
from numpyro.distributions import constraints

# Define the model and variational family


class TBIP:
    def __init__(self, N, D, K, V, batch_size, init_mu_theta=None, init_mu_beta=None):
        self.N = N  # number of people
        self.D = D  # number of documents
        self.K = K  # number of topics
        self.V = V  # number of words in vocabulary
        self.batch_size = batch_size  # number of documents in a batch

        if init_mu_theta is None:
            init_mu_theta = jnp.zeros([D, K])
        else:
            self.init_mu_theta = init_mu_theta

        if init_mu_beta is None:
            init_mu_beta = jnp.zeros([K, V])
        else:
            self.init_mu_beta = init_mu_beta

    def model(self, Y_batch, d_batch, i_batch):
        with plate("i", self.N):
            # Sample the per-unit latent variables (ideal points)
            x = sample("x", dist.Normal())

        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                beta = sample("beta", dist.Gamma(0.3, 0.3))
                eta = sample("eta", dist.Normal())

        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                # Sample document-level latent variables (topic intensities)
                theta = sample("theta", dist.Gamma(0.3, 0.3))

            # Compute Poisson rates for each word
            P = jnp.sum(
                jnp.expand_dims(theta, 2)
                * jnp.expand_dims(beta, 0)
                * jnp.exp(
                    jnp.expand_dims(x[i_batch], (1, 2)) * 
                    jnp.expand_dims(eta, 0)
                ),
                1,
            )

        with plate("v", size=self.V, dim=-1):
            # Sample observed words
            sample("Y_batch", dist.Poisson(P), obs=Y_batch)

    def guide(self, Y_batch, d_batch, i_batch):
        # This defines variational family. Notice that each of the latent variables
        # defined in the sample statements in the model above has a corresponding
        # sample statement in the guide. The guide is responsible for providing
        # variational parameters for each of these latent variables.

        # Also notice it is required that model and the guide have the same call.

        mu_x = param(
            "mu_x", init_value=-1 + 2 * random.uniform(random.PRNGKey(1), (self.N,))
        )
        sigma_x = param(
            "sigma_y", init_value=jnp.ones([self.N]), constraint=constraints.positive
        )

        mu_eta = param(
            "mu_eta", init_value=random.normal(random.PRNGKey(2), (self.K, self.V))
        )
        sigma_eta = param(
            "sigma_eta",
            init_value=jnp.ones([self.K, self.V]),
            constraint=constraints.positive,
        )

        mu_theta = param("mu_theta", init_value=self.init_mu_theta)
        sigma_theta = param(
            "sigma_theta",
            init_value=jnp.ones([self.D, self.K]),
            constraint=constraints.positive,
        )

        mu_beta = param("mu_beta", init_value=self.init_mu_beta)
        sigma_beta = param(
            "sigma_beta",
            init_value=jnp.ones([self.K, self.V]),
            constraint=constraints.positive,
        )

        with plate("i", self.N):
            sample("x", dist.Normal(mu_x, sigma_x))

        with plate("k", size=self.K, dim=-2):
            with plate("k_v", size=self.V, dim=-1):
                sample("beta", dist.LogNormal(mu_beta, sigma_beta))
                sample("eta", dist.Normal(mu_eta, sigma_eta))

        with plate("d", size=self.D, subsample_size=self.batch_size, dim=-2):
            with plate("d_k", size=self.K, dim=-1):
                sample("theta", dist.LogNormal(mu_theta[d_batch], sigma_theta[d_batch]))

    def get_batch(self, rng, Y, author_indices):
        # Helper functions to obtain a batch of data, convert from scipy.sparse
        # to jax.numpy.array and move to gpu

        D_batch = random.choice(rng, jnp.arange(self.D), shape=(self.batch_size,))
        Y_batch = jax.device_put(jnp.array(Y[D_batch].toarray()), jax.devices("gpu")[0])
        D_batch = jax.device_put(D_batch, jax.devices("gpu")[0])
        I_batch = author_indices[D_batch]
        return Y_batch, I_batch, D_batch
```

```python 

# Initialize the model
from optax import adam, exponential_decay
from numpyro.infer import SVI, TraceMeanField_ELBO
from jax import jit

num_steps = 50000
batch_size = 512  # Large batches are recommended
learning_rate = 0.01
decay_rate = 0.01

tbip = TBIP(
    N=num_authors,
    D=num_documents,
    K=num_topics,
    V=num_words,
    batch_size=batch_size,
    init_mu_theta=initial_document_loc,
    init_mu_beta=initial_objective_topic_loc,
)

svi_batch = SVI(
    model=tbip.model,
    guide=tbip.guide,
    optim=adam(exponential_decay(learning_rate, num_steps, decay_rate)),
    loss=TraceMeanField_ELBO(),
)

# Compile update function for faster training
svi_batch_update = jit(svi_batch.update)

# Get initial batch. This informs the dimension of arrays and ensures they are
# consistent with dimensions (N, D, K, V) defined above.
Y_batch, I_batch, D_batch = tbip.get_batch(random.PRNGKey(1), counts, author_indices)

# Initialize the parameters using initial batch
svi_state = svi_batch.init(
    random.PRNGKey(0), Y_batch=Y_batch, d_batch=D_batch, i_batch=I_batch
)
```


```python 

# @title Run this cell to create helper function for printing topics


def get_topics(
    neutral_mean, negative_mean, positive_mean, vocabulary, print_to_terminal=True
):
    num_topics, num_words = neutral_mean.shape
    words_per_topic = 10
    top_neutral_words = np.argsort(-neutral_mean, axis=1)
    top_negative_words = np.argsort(-negative_mean, axis=1)
    top_positive_words = np.argsort(-positive_mean, axis=1)
    topic_strings = []
    for topic_idx in range(num_topics):
        neutral_start_string = "Neutral  {}:".format(topic_idx)
        neutral_row = [
            vocabulary[word] for word in top_neutral_words[topic_idx, :words_per_topic]
        ]
        neutral_row_string = ", ".join(neutral_row)
        neutral_string = " ".join([neutral_start_string, neutral_row_string])

        positive_start_string = "Positive {}:".format(topic_idx)
        positive_row = [
            vocabulary[word] for word in top_positive_words[topic_idx, :words_per_topic]
        ]
        positive_row_string = ", ".join(positive_row)
        positive_string = " ".join([positive_start_string, positive_row_string])

        negative_start_string = "Negative {}:".format(topic_idx)
        negative_row = [
            vocabulary[word] for word in top_negative_words[topic_idx, :words_per_topic]
        ]
        negative_row_string = ", ".join(negative_row)
        negative_string = " ".join([negative_start_string, negative_row_string])

        if print_to_terminal:
            topic_strings.append(negative_string)
            topic_strings.append(neutral_string)
            topic_strings.append(positive_string)
            topic_strings.append("==========")
        else:
            topic_strings.append(
                "  \n".join([negative_string, neutral_string, positive_string])
            )

    if print_to_terminal:
        all_topics = "{}\n".format(np.array(topic_strings))
    else:
        all_topics = np.array(topic_strings)
    return all_topics
```
```python

# Run SVI
from tqdm import tqdm
import pandas as pd

print_steps = 100
print_intermediate_results = False

rngs = random.split(random.PRNGKey(2), num_steps)
losses = []
pbar = tqdm(range(num_steps))


for step in pbar:
    Y_batch, I_batch, D_batch = tbip.get_batch(rngs[step], counts, author_indices)
    svi_state, loss = svi_batch_update(
        svi_state, Y_batch=Y_batch, d_batch=D_batch, i_batch=I_batch
    )

    loss = loss / counts.shape[0]
    losses.append(loss)
    if step % print_steps == 0 or step == num_steps - 1:
        pbar.set_description(
            "Init loss: "
            + "{:10.4f}".format(jnp.array(losses[0]))
            + f"; Avg loss (last {print_steps} iter): "
            + "{:10.4f}".format(jnp.array(losses[-100:]).mean())
        )

    if (step + 1) % 2500 == 0 or step == num_steps - 1:
        # Save intermediate results
        estimated_params = svi_batch.get_params(svi_state)

        neutral_mean = (
            estimated_params["mu_beta"] + estimated_params["sigma_beta"] ** 2 / 2
        )

        positive_mean = (
            estimated_params["mu_beta"]
            + estimated_params["mu_eta"]
            + (estimated_params["sigma_beta"] ** 2 + estimated_params["sigma_eta"] ** 2)
            / 2
        )

        negative_mean = (
            estimated_params["mu_beta"]
            - estimated_params["mu_eta"]
            + (estimated_params["sigma_beta"] ** 2 + estimated_params["sigma_eta"] ** 2)
            / 2
        )

        np.save("neutral_topic_mean.npy", neutral_mean)
        np.save("negative_topic_mean.npy", positive_mean)
        np.save("positive_topic_mean.npy", negative_mean)

        topics = get_topics(neutral_mean, positive_mean, negative_mean, vocabulary)

        with open("topics.txt", "w") as f:
            print(topics, file=f)

        authors = pd.DataFrame(
            {"name": author_map, "ideal_point": np.array(estimated_params["mu_x"])}
        )
        authors.to_csv("authors.csv")

        if print_intermediate_results:
            print(f"Results after {step} steps.")
            print(topics)
            sorted_authors = "Authors sorted by their ideal points: " + ",".join(
                list(authors.sort_values("ideal_point")["name"])
            )
            print(sorted_authors.replace("\n", " "))
```

```python

import os
import matplotlib.pyplot as plt
import seaborn as sns

neutral_topic_mean = np.load("neutral_topic_mean.npy")
negative_topic_mean = np.load("negative_topic_mean.npy")
positive_topic_mean = np.load("positive_topic_mean.npy")
authors = pd.read_csv("authors.csv")
authors["name"] = authors["name"].str.replace("\n", "")

```


```python 

selected_authors = np.array(
    [
        "Dean Heller (R)",
        "Bernard Sanders (I)",
        "Elizabeth Warren (D)",
        "Charles Schumer (D)",
        "Susan Collins (R)",
        "Marco Rubio (R)",
        "John Mccain (R)",
        "Ted Cruz (R)",
    ]
)

sns.set(style="whitegrid")
fig = plt.figure(figsize=(12, 1))
ax = plt.axes([0, 0, 1, 1], frameon=False)
for index in range(authors.shape[0]):
    ax.scatter(authors["ideal_point"][index], 0, c="black", s=20)
    if authors["name"][index] in selected_authors:
        ax.annotate(
            author_map[index],
            xy=(authors["ideal_point"][index], 0.0),
            xytext=(authors["ideal_point"][index], 0),
            rotation=30,
            size=14,
        )
ax.set_yticks([])
plt.show()

```

```python

from numpyro.infer.autoguide import AutoNormal


def create_svi_object(guide):
    svi_object = SVI(
        model=tbip.model,
        guide=guide,
        optim=adam(exponential_decay(learning_rate, num_steps, decay_rate)),
        loss=TraceMeanField_ELBO(),
    )

    Y_batch, I_batch, D_batch = tbip.get_batch(
        random.PRNGKey(1), counts, author_indices
    )

    svi_state = svi_batch.init(
        random.PRNGKey(0), Y_batch=Y_batch, d_batch=D_batch, i_batch=I_batch
    )

    return svi_state


# This state uses the guide defined manually above
svi_state_manualguide = create_svi_object(guide=tbip.guide)

# Now let's create this object but using AutoNormal guide. We just need to ensure that
# parameters are initialized as above.
autoguide = AutoNormal(
    model=tbip.model,
    init_loc_fn={"beta": initial_objective_topic_loc, "theta": initial_document_loc},
)
svi_state_autoguide = create_svi_object(guide=autoguide)


# Assert that the keys in the optimizer states are identical
assert svi_state_manualguide[0][1][0].keys() == svi_state_autoguide[0][1][0].keys()

# Assert that the values in the optimizer states are identical
for key in svi_state_manualguide[0][1][0].keys():
    assert jnp.all(
        svi_state_manualguide[0][1][0][key] == svi_state_autoguide[0][1][0][key])

```

