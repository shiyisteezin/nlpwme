
# What is Mutual Information? 

A non-negative metric known as mutual information (MI) is used to assess how closely two random variables depend on one another. The reciprocal information measures how much we can learn about a second variable by looking at the values of the first.

Because it can assess non-linear relationships as well as linear ones, the mutual information is a useful substitute for Pearson's correlation coefficient. In contrast to [Pearson's correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient), it is also appropriate for both continuous and discrete variables. Entropy and MI are closely connected concepts. As a result, this blog will provide a summary of entropy and discuss how the two are related.

### What’s Pointwise Mutual Information? 


Pointwise mutual information (PMI), sometimes known as point mutual information, is a measure of association used in statistics, probability theory, and information theory. It contrasts the likelihood of two events happening simultaneously with the likelihood of the same events occurring independently.

PMI, particularly in its positive pointwise mutual information variant, has been referred to as `one of the most important concepts in NLP` because it draws on the intuition that

> the best way to weigh the association between two words is to ask how much more the two words co-occur in a corpus than we would have expected them to appear by chance.


Robert Fano first proposed the idea in 1961 under the label "mutual information," however the phrase is now more commonly used to refer to a related measure of reliance between random variables: The average PMI of all potential events is referred to as the mutual information (MI) of two discrete random variables.


#### Why Is It Important To Our Discussion? 

When discussing information theory in relation to language processing. It is a method of quantifying communication or message transmission through mathematics. Associations between messages are therefore crucial.


#### Mutual Information and Pointwise Mutual Information 

Mutual information is theoretically plagued by two issues: In contrast to conventional assessment measures that do not differentiate between long and short texts, it assumes independent word variables and gives longer documents more weights in the estimate of the feature scores. 

a different version of mutual information is provided that gets over both issues: Weighted Average Pointwise Mutual Information (WAPMI). We offer both strong theoretical and empirical support for WAPMI. 

Additionally, it's demonstrated that WAPMI possesses a useful quality that other feature metrics do not, more specifically _the ability to automatically choose the appropriate feature set size by maximizing an objective function_. This can be accomplished using a straightforward heuristic rather than requiring expensive techniques like EM and model selection.

#### The Formula of MI

$$I(W:C) = \sum_{t=1}^{|V|} \sum_{j=1}^{|C|} \text{p}(\text{w}_{t}, \text{c}_{j}) \text{log} \frac{\text{p}(\text{w}_{t}|\text{c}_{j})} {\text{p} (\text{w}_{t})}$$



This could be written as a weighted sum of Kullback-Leibler or KL divergences, because this is the measure of information gain between two probability disitributions. _p_ and _q_ is defined as $D(p || q) = \sum_{x} p(x) \text{log} \frac{p(x)}{q(x)}$. 

Therefore, this can be written as the weighted average KL-divergence between the class-conditional distribution of words and the global (unconditioned) distribution in the entire corpus: 



$$I(W:C) = \sum_{j=1}^{|C|} \text{p}({c}_{j}) \text{D} (\text{p}(\text{W}|\text{c}_{j})|| {\text{p} (\text{W})})$$

Now we just need a binary feature to indicate whether the next word in a document is $w_{t}$, namely $p(W_{t} = 1) = p(W = w_{t})$

$$ MI(w_{t}) := I(W_{t}; C) = \sum_{j=1}^{|C|} \sum_{x=0,1} p(W_{t} = x, c_{j}) log \frac{p(W_{t} = x | c_{j})}{p(W_{t} = x)} $$

However, the problem with above formula is that contrary to its assumption of $w_{t}$ as an independent random variable, in fact $\sum_{t=1}^{|V|} p(W_{t}=1)=1$, so to avoid this problem point-wise mutual information is introduced where the formula (2) sums over word scores instead; demonstrated as follows,

$$PMI(w_{t}) := \sum_{j=1}^{|C|} p(w_{t}, c_{j}) \text{log} \frac{p(w_{t} | c_{t})}{p(w_{t})}$$

Another problem arises where all training documents in one class is treated according to class-conditional probabilities as one big document, so the formula is impacted by individual document length especially the larger ones. To resolve this problem, instead using class-conditional distribution, the document-conditional probabilities ($p(w_{t}, c_{j}) = n(w_{t},d_{i})/|d_{i}|$)  
are in leu used. Together as a whole, 


$$ WAPMI(w_{t}) := \sum_{j=1}^{|C|} \sum_{d_{i} \in c_{j}} \alpha_{i} p(w_{t} | d_{i}) log \frac{p(w_{t}|c_{j})}{p(w_{t})}$$

where the weight coefficient $\alpha_{i}$ could be calibrated to account for 

- $\alpha_{i} = p(c_{j}) · |d_{i}|/\sum{d_{i} \in c_{j}} |d_{i}|$. This gives each document a weight proportional to its lengths. 

- $\alpha_{i} = 1/ \sum_{j=1}^{|C|} |c_{j}|$. This gives equal weight to all documents. This corresponds to an evaluation measure that counts each misclassified document as the same error.

- $a_{i} = 1/(|c_{j}| · |C|)$ where $d_{i} \in c_{j}$. This gives equal weight to the classes by normalizing for class size, i.e. documents from small categories receive higher weights.


Here is an example of the [Naive Bayes Mutual Information Classifier](https://towardsdatascience.com/multinomial-na%C3%AFve-bayes-classifier-using-pointwise-mutual-information-9ade011fcbd0)