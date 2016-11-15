# Hybrid computing using a neural network with dynamic external memory
### by Google DeepMind

Paper available here: http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html 

This paper, published in Nature 2016, develops the idea of a differentiable neural computer (DNC). This paper is based heavily on the work Alex Graves and Greg Wayne previously did on "Neural Turing Machines" (link: https://arxiv.org/abs/1410.5401). I'd highly recommend that anybody attempting to really understand DNCs read and understand his paper first.

DNCs can be viewed as a more general type of LSTM (http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) where the network learns how to use memory to understand data rather than attempting to learn the raw sequence relationships. This allows the network to be trained on a small amount of data and generalize to large amount of data without training, as well as handling inputs that were not necessarily seen during training --- a huge divergence from what was previously possible.

These findings are facilitated by the author's novel framework of memory which is fully differentiable (thus the name). Because of this property, the memory structure is able to be coupled with a neural network and trained by gradient descent or any other optimization method.
