

# The Importance of Understanding Context in Natural Language Processing

**Table of Contents**

\toc

The old school of language theories before the whole Connectionism came out was boggled by the difficulty of incorporating context and nuances. Again as we have talked in the previous sections on [the historical development of NLP](https://shiyis.github.io/nlpwme/modules/1-phil-of-mind/), Connectionism emerged as a response to perceived limitations in the traditional computational theory of mind, particularly in its ability to handle context and capture the complexity of cognitive processes. The computational theory of mind, influenced by classical artificial intelligence (AI), often relied on symbolic representations and rule-based systems.

In the subsequent section of this blog, how the later researches in AI have been able to have a break through in terms of solving the problem will be explained.

### References

[Word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method](https://arxiv.org/abs/1402.3722) by Yoav Goldberg and Omer Levy

### What Is Word2Vec?

As per definition, Word2Vec is a technique in natural language processing (NLP) that represents words as vectors in a continuous vector space, capturing semantic relationships between words. As an alternative to the simpler one hot encoding method, one of the reasons was that it could not accurately capture the similarity between different words as the cosine similarity could. 

One tool to address the aforementioned issue is Word2vec. It uses a fixed-length vector to represent each word and makes use of these vectors to more clearly show the linkages of analogies and similarity between various words. The Word2vec tool has two models: the continuous bag of words [CBOW] (effective estimate of word representations in vector space) and the skip-gram [distributed representations of words and phrases and their compositionality]. We will next examine the two models and how they were trained.

For the vectors $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d$, the math formula of cosine similarity is below, 


$$\frac{\boldsymbol{x}^\top \boldsymbol{y}}{\|\boldsymbol{x}\| \|\boldsymbol{y}\|} \in [-1, 1].$$

As we have talked about in a different blog, recent researches strive to represent mental processes in a more distributed way and also be able to capture their compositionality; therefore, two different methods were introduced: Skip-gram and CBOW. The first one is to capture the context within a sentence, and the second one smoothly samples words through an algorithm called sliding window, which we will go into details in subsequent sections. 

### Skip-gram Model

The Skip-gram model in Word2Vec is a type of word embedding model designed to learn distributed representations (embeddings) for words in a continuous vector space. The primary objective of Skip-gram is to predict the context words given a target word. Let's break down the key components of the Skip-gram model:

> **Objective Function**: The training objective is to maximize the probability of the context words given the target word. Mathematically, it involves maximizing the conditional probability of the context words given the target word.

> **Input-Output Pairs**: For each occurrence of a word in the training data, the Skip-gram model generates multiple training examples. Each training example consists of a target word and one of its context words. The context words are sampled from a fixed-size window around the target word.

> **Architecture**: The model is typically a neural network with a single hidden layer. The input layer represents the target word, and the output layer represents the context words. The hidden layer contains the word embeddings (continuous vector representations) for each word in the vocabulary.

>**Softmax Activation**: The output layer uses a softmax activation function, which converts the raw output scores into probabilities. These probabilities represent the likelihood of each word being a context word given the target word.

>**Training**: During training, the model adjusts its parameters (word embeddings and weights) to improve the prediction accuracy of context words for each target word. The training process involves backpropagation and gradient descent to minimize the negative log-likelihood of the observed context words.

>**Word Embeddings**: Once trained, the hidden layer's weights serve as the word embeddings. These embeddings capture semantic relationships between words based on their co-occurrence patterns.

>**Applications**: The learned word embeddings can be used for various natural language processing tasks, such as similarity analysis, language modeling, and as input representations for downstream machine learning tasks.

In summary, the Skip-gram model learns word embeddings by training on the task of predicting context words given a target word. It captures the distributional semantics of words, representing them as vectors in a continuous vector space.


### CBOW or Continuous Bag of Words Model

Continuous Bag of Words (CBOW) model in Word2Vec involves a `sliding window` during its training process. The sliding window is a mechanism used to define the context of a target word.

Here's how it typically works:

> **Context Window**: CBOW considers a fixed-size context window around each target word. This window defines the neighboring words that are used as input to predict the target word.

> **Sliding Window**: The sliding window moves through the training text, and at each position, it considers the words within the window as the context for the current target word.

> **Training Examples**: For each target word in the text, the CBOW model is trained to predict the target word based on the words within its context window.

> **Parameter**: The size of the context window is a parameter that can be set during the training process. It determines how many words on either side of the target word are considered as context.

For example, if the context window size is set to 5, the CBOW model will use the five words to the left and five words to the right of the target word as the context for training at each step. 

This sliding window mechanism allows the model to capture the local context and syntactic information around each word, helping to learn meaningful word embeddings based on the words that tend to co-occur.



The skip-gram model assumes that a word can be used to generate the words that surround it in a text sequence. For example, we assume that the text sequence is `the`, `man`, `loves`, `his`, and `son`. We use `loves` as the central target word and set the context window size to 2. As shown below, given the central target word `loves`, the skip-gram model is concerned with the conditional probability for generating the context words, `the`, `man`, `his` and `son`, that are within a distance of no more than 2 words, which is,

$$\mathbb{P}(\textrm{the},\textrm{man},\textrm{his},\textrm{son}\mid\textrm{loves}).$$

We assume that, given the central target word, the context words are generated independently of each other. In this case, the formula above can be rewritten as

$$\mathbb{P}(\textrm{the}\mid\textrm{loves})\cdot\mathbb{P}(\textrm{man}\mid\textrm{loves})\cdot\mathbb{P}(\textrm{his}\mid\textrm{loves})\cdot\mathbb{P}(\textrm{son}\mid\textrm{loves}) $$
    

![alt=skip-gram-demo](https://www.di.ens.fr/~lelarge/skip-gram.svg)

    
In the skip-gram model, each word is represented as two $d$-dimension vectors, which are used to compute the conditional probability. We assume that the word is indexed as $i$ in the dictionary, its vector is represented as $\boldsymbol{v}_i\in\mathbb{R}^d$ when it is the central target word, and $\boldsymbol{u}_i\in\mathbb{R}^d$ when it is a context word.  Let the central target word $w_c$ and context word $w_o$ be indexed as $c$ and $o$ respectively in the dictionary. The conditional probability of generating the context word for the given central target word can be obtained by performing a softmax operation on the vector inner product:


$$\mathbb{P}(w_o \mid w_c) = \frac{\text{exp}(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}$$


where vocabulary index set $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$. Assume that a text sequence of length $T$ is given, where the word at time step $t$ is denoted as $w^{(t)}$. Assume that context words are independently generated given center words. When context window size is $m$, the likelihood function of the skip-gram model is the joint probability of generating all the context words given any center word

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} \mathbb{P}(w^{(t+j)} \mid w^{(t)})$$


After the training, for any word in the dictionary with index $i$, we are going to get its two word vector sets $\boldsymbol{v}_i$ and $\boldsymbol{u}_i$.  In applications of natural language processing (NLP), the central target word vector in the skip-gram model is generally used as the representation vector of a word.

### A Comparison Between These Two Models

The Skip-gram model in Word2Vec does not use a fixed sliding window in the same way as the Continuous Bag of Words (CBOW) model. Instead, the Skip-gram model considers each word-context pair in the training data separately.

Here's a brief comparison of the two models:

**CBOW**:

CBOW predicts a target word based on its surrounding context (a fixed-size window of neighboring words).
It sums up the embeddings of the context words and uses them to predict the target word.
Skip-gram:


**Skip-gram**:

On the other hand, takes a target word as input and aims to predict the context words within a certain range.
It treats each word-context pair in the training data as a separate training example.
In the Skip-gram model, there is no fixed sliding window that moves through the text. Instead, each word is considered in isolation, and the model is trained to predict the words that are likely to appear in its context. The context words can be selected from a fixed-size window around the target word, but it's not constrained by a fixed window during the entire training process.

In summary, while `CBOW` uses a sliding window to define the context for each target word, `Skip-gram` treats each word-context pair independently without a fixed sliding window. Both of them are important models for building the distributed and compositional attributes of an utterance. 

For a full implementation of Word2Vec, please check out this [notebook](https://github.com/dataflowr/notebooks/blob/master/Module8/08_Word2vec_pytorch_empty.ipynb).