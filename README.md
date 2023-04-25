# Attention-Matrix
The input sequences are translated to the output sequences in the attention-based model using a technique that gives attention weights
to every element in the input sequence. These weights specify the relative importance of each element in the input sequence, which the model
will use to generate the appropriate output sequence. Based on the similarity between the input and output sequences at each time step, 
the attention weights are determined. These weights are represented by the attention matrix for each output sequence time step.
Each element of the output sequence in an attention-based model is formed by examining the whole input sequence, but with varying weights 
assigned to each element based on their relevance to the current output element. These relevance ratings are calculated using a compatibility function, 
which compares the similarity of the current output element to each element in the input sequence. The function is usually implemented as a dot product
or a concatenation, followed by a neural network layer.
The attention mechanism then uses a softmax function to these relevance scores to build a set of weights that total to one, signifying the relative 
importance of each input sequence element for the current output element. These weights are then calculated.

Contrarily, convolutional models produce the equivalent output sequence by applying a fixed filter to the input sequence at each time step.
Typically, the filter is fixed during testing and learned during training. Which attributes of the input are determined by the filter.
