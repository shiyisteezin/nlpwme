
@def sequence = ["noisy-channel"]


# How Everything Started


Table of Contents

\toc



Hi there, if you have come across this blog and have the patience to go through it with me, by the end of it you will get to know and understand what noisy channel model that Claude E. Shannon first presented and experimented with back in the 40s really is. And you will also get to understand why it is important for our discussion. 



{{youtube_placeholder noisy-channel}}


### What Exactly Is A Noisy Channel Model?

It's a system designed to capture how information gets transmitted.  It's also a mathematical or probabilistic model developed to capture the way signals get transmitted. 

### One of The Most Important Master's Theses In The 20th Century

A Symbolic Analysis of Relay and Switching Circuits. Shannon's paper examines symbolic logic of 19th century English mathematician George Boole, and presents how the boolean logic could have a profound impact on electronic circuit design. Which is the entropy equation 

$$\text{H} = -\sum \text{p}_{i} \text{log}_{i}  \text{p}_{i}$$

The equation we have discussed in another blog defines the information source in terms of the probability distribution of symbols being produced by that information source. 

This equation marks the fundamental mechanism behind it, which has to do with the level of uncertainty in information. 

### The Morse Code

The model is created by having a message written in English encoded in Morse code. The message then gets trasmitted over telegraph line. There's the noise coming from the transmission line. At the receiving end, another telegraph line receives the message and decodes the message. The decoded message then gets stored as human readable form. The final message may contain errors. 

Let's break down the components of the noisy channel model,

- The model begins with an information source that's written in human language. 
- The message is encoded by the transmitter from English in to Morse code then sent from the channel to the telegraph line. 
- The message then gets sent to the receiver to the receiver for decoding. 
- Because of the noise in the channel, the final decoded message may contain errors.


### Another Important Equation

Here I will follow the video and jot down the important equation that represents the transmission of a message through the noisy channel. 

Below is the equation. So, here `e` and/or `f` represents respectively the message that gets encoded and sent in the channel and the output message that gets decoded back to human readable language. 

Here as we are looking at the equation, given the output message `f`, we want to reconstruct the message back to `e`, by finding the `e` that maximizes the probability of `f` given `e`. Whatever `e` hat that satisfies our conditions will be the best hypothesis. 

$$\hat{e} = \underset{e}{\arg\max} \ p(\text{e|f})$$

### Bayes To The Rescue
To calculate the probability, we are going to use an important mathematical law, the Bayes equation. 

$$p(\text{e|f}) = \frac{p(\textbf{f|e})p(\textbf{e})}{\textbf{f}} $$


We will be talking about this God of an equation independently in another blog, but here we could understand that we are able to calculate the above noisy channel model through using the Bayes' law. And the three components available in the equation will allow us to do this. 

The posterior function which is intuitively the outcome function $p(\text{e|f})$ is derived from the likelihood function which is the probability of `e` given `f` or $p(\text{f|e})$ and the prior distributions $p(e)$ together divided by the prior distribution of $p(f)$.

Given this definition, we could redefine our previous definition in terms of the channel model and the language models.

First we do it by taking the posterior solution $p(\text{e|f})$ and replace it with the right-hand side of the Bayes' Law. According to the tutorial video, we could simplify the equation a little bit. 


Recall performing an arg max operation over English messages `e`, this means we are looking for the English message `e`, which maximizes the total value of the equation. 

Since `e` doesn't show up in the denominator, so no matter what value `e` has, won't be affected which leaves us with the final equation,

$$\hat{e} = \underset{e}{\arg\max} (\text{f|e}) \ p(\text{e})$$



