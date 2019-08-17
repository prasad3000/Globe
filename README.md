# Globe word-to-vec

Establish the relation between multiple english words using pytorch

## Getting Started
When working with words, dealing with the huge but sparse domain of language can be challenging. Even for a small corpus, your neural network (or any type of model) needs to support many thousands of discrete inputs and outputs. Besides the raw number words, the standard technique of representing words as one-hot vectors (e.g. "the" = `[0 0 0 1 0 0 0 0 ...]`) does not capture any information about relationships between words. Word vectors address this problem by representing words in a multi-dimensional vector space. This can bring the dimensionality of the problem from hundreds-of-thousands to just hundreds. Plus, the vector space is able to capture semantic relationships between words in terms of distance and vector arithmetic. 
How is GloVe different from word2vec?(https://www.quora.com/How-is-GloVe-different-from-word2vec) on Quora for some better explanations.
Here we use some pretrained vector to make things more reliable.
For more details follow https://blog.acolyer.org/2016/04/22/glove-global-vectors-for-word-representation/



### Dependency

#pytorch
#torchtext

### Implimentation

The 'GloVe' object includes three attributes:
1. `stoi` _string-to-index_ returns a dictionary of words to indexes
2. `itos` _index-to-string_ returns an array of words by index
3. `vectors` returns the actual vectors. To get a word vector get the index to get the vector:
