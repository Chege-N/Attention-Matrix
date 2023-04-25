import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
import hw6_utils as utils
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConvNet(nn.Module):
    """
    A model with one 1d conv layer and one 1d transpose conv layer.
    It generates the embeddings with the conv layer with the given 
    kernel size and the number of output channels. Then uses ReLU 
    activation on the generated embedding and passes it to a 
    transpose conv layer that has the same kernel size as the conv
    layer and one output channel.

    Arguments:
        kernel_size: size of the kernel for 1d convolution.
        length: length of the sequence (10 in the given data)
        out_chan: number of output channels for the conv layer
            and number of input layers in the trasnpose conv layer.
        bias: binary flag for using bias.

    Returns: 
        the predicted mapping (size: n x length)
    """
    def __init__(self, kernel_size=3, length=10, out_chan=32, bias=True):
        super(ConvNet, self).__init__()

        self.conv = nn.Conv1d(1, out_chan, kernel_size)
        self.relu = nn.ReLU()
        self.trans_conv = nn.ConvTranspose1d(out_chan, 1, kernel_size)
        self.input_length = length

    def forward(self, x):
        pass
        # Reshape input vector to [batch_size, 1, input_length]
        x = torch.reshape(x, (-1, 1, self.input_length))

        # Apply convolutional layer
        x = self.conv(x)

        # Apply ReLU activation function
        x = self.relu(x)

        # Apply transposed convolutional layer
        x = self.trans_conv(x)

        # Reshape output vector to [batch_size, input_length]
        x = torch.reshape(x, (-1, self.input_length))

        # Apply sigmoid activation function to obtain binary output vector
        x = torch.sigmoid(x)

        return x

class Attention(nn.Module):
    """
    An attention-based model with one single head. It uses linear layer
    without bias to generate query key and value for the embedding of 
    each element of the input vector. 

    Arguments:
        length: length of the sequence (10 in the given data)
        embedding_dim: the embedding dimension for each element
            of the sequence.
        positional_encoding: a booliean flag which turns on the 
            positional encoding when set to True.

    Returns: 
        the predicted mapping (size: n x length)
    """
    def __init__(self, length=10, embedding_dim=16, positional_encoding=True):
        super().__init__()

        self.embedding = nn.Embedding(2, embedding_dim)

        # TODO: Add 3 linear layers with no bias for generating 
        # the query, key, and values
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.out = nn.Linear(embedding_dim, 1)

        self.attention = np.zeros((length,length))
        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.pos_encode = utils.PositionalEncoding(d_model=embedding_dim, max_len=length)

    def compute_new_values(self, q, k, v):
        """
        Computes the attention matrix and the new values:
        
        Arguments:
            q: query
            k: key
            v: value

        Returns:
            values: the new values computed using the 
                attention matrix.
            attentions: attention matrix
        """
        attentions = F.softmax(
            torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32)), dim=-1)

        # Apply attention weights to values
        values = torch.bmm(attentions, v)

        return values, attentions.detach()

    def attention_mat(self):
        return np.mean(self.attention, axis=0)

    def forward(self, x):
        x = self.embedding(x.long())

        if self.positional_encoding:
            x_ = x.permute(1, 0, 2)
            x_ = self.pos_encode(x_)
            x = x_.permute(1, 0, 2)

        # TODO: compute the query, key, and value representations.
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        values, attention = self.compute_new_values(query, key, value)

        self.attention = attention.numpy()
        values = self.out(values)
        return values.view(x.shape[0],-1)


def train(model, epoch, optimizer, criterion, trainloader, log=True):
    model.train()
    train_loss = 0.0
    total_seen = 0
    correct = 0.0
    for batch_idx, inputs in enumerate(trainloader):
        inputs = inputs.float().to(device)
        seq_len = inputs.shape[-1]//2
        X = inputs[:,:seq_len]
        Y = inputs[:,seq_len:]

        model.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        predictions = torch.clip(outputs,0,1)
        predictions = (predictions>0.5).float()
        total_seen += Y.size(0)
        train_loss += loss.item()
        correct += predictions.eq(Y).sum().item()

    accuracy = 100.*correct/(seq_len*total_seen) 
    if log:
        print('Epoch: %d  Train Loss: %.3f | Train Acc: %.3f' % (epoch, train_loss/(batch_idx+1), accuracy))

    return accuracy


def test(model, testloader, log=True):
    model.eval()
    correct = 0.0
    predictions = None
    for batch_idx, inputs in enumerate(testloader):
        inputs = inputs.float().to(device)
        seq_len = inputs.shape[-1]//2
        X = inputs[:,:seq_len]
        Y = inputs[:,seq_len:]
        outputs = model(X)
        predictions = torch.clip(outputs,0,1)
        predictions = (predictions>0.5).float()
        correct += torch.prod(predictions.eq(Y).float(), dim=1).item()

    if log:
        print('Test Acc: %.3f' % (100.*correct/len(testloader)))
    return outputs.detach(), correct



def run_model(model_type, positional_encoding, kernel_size):
    seed_val = 1
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    length = 10
    X = utils.load_data()
    n = X.shape[0]

    train_X = X[:int(0.8*n)]
    test_X = X[int(0.8*n):]

    trainloader = torch.utils.data.DataLoader(train_X, shuffle=True, batch_size=64, num_workers=1)
    testloader = torch.utils.data.DataLoader(test_X, shuffle=False, batch_size=1, num_workers=1)

    if model_type == 'Convolution':
        model = ConvNet(length=length, kernel_size=kernel_size).to(device)
    elif model_type == 'Attention':
        model = Attention(length=length, positional_encoding=positional_encoding).to(device)
    else:
        print('not a valid model!')
        exit(0)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss(reduction="none")
    epoch_train_accuracy = []
    for epoch in range(25):
        epoch_train_accuracy.append(train(model, epoch, optimizer, criterion, trainloader))
        if epoch % 10 == 0 and epoch > 0:
            test(model, testloader)

    if model_type == "Attention":
        print(model.attention)
        # plt.figure(figsize=(10, 5))
        # plt.title("Attention matrix")
        # plt.matshow(model.attention)
        # plt.legend()
        # plt.show()
    return epoch_train_accuracy


# define inputs and outputs
inputs = tf.keras.layers.Input(shape=(seq_length, input_dim))
outputs = tf.keras.layers.Dense(output_dim, activation='softmax')(inputs)

# define attention mechanism
attention = tf.keras.layers.Attention()([inputs, outputs])
context_vector = tf.keras.layers.GlobalAveragePooling1D()(attention)

# concatenate attention vector with output
concatenated = tf.keras.layers.Concatenate()([outputs, context_vector])

# define final output layer
output = tf.keras.layers.Dense(output_dim, activation='softmax')(concatenated)

# define model
model = tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_val, y_val))

"""
This is the Attention Matrix Plot
"""

plt.imshow(attention_weights, cmap='hot', interpolation='nearest')
plt.xlabel('Input sequence')
plt.ylabel('Output sequence')
plt.title('Attention Matrix')
plt.show()


"""
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
Typically, the filter is fixed during testing and learned during training. Which attributes of the input are determined by the filter

"""

if __name__ == "__main__":
    # seed_val = 1
    # torch.manual_seed(seed_val)
    # torch.cuda.manual_seed_all(seed_val)
    # np.random.seed(seed_val)
    # random.seed(seed_val)
    #
    # length = 10
    # kernel_size = 10
    # X = utils.load_data()
    # n = X.shape[0]
    # model_type = 'Attention'
    #
    # train_X = X[:int(0.8*n)]
    # test_X = X[int(0.8*n):]
    #
    # trainloader = torch.utils.data.DataLoader(train_X, shuffle=True, batch_size=64, num_workers=1)
    # testloader = torch.utils.data.DataLoader(test_X, shuffle=False, batch_size=1, num_workers=1)
    #
    # if model_type == 'Convolution':
    #     model = ConvNet(length=length, kernel_size=kernel_size).to(device)
    # elif model_type == 'Attention':
    #     model = Attention(length=length, positional_encoding=False).to(device)
    # else:
    #     print('not a valid model!')
    #     exit(0)
    #
    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    # criterion = nn.MSELoss(reduction="none")
    # epoch_train_accuracy = []
    # for epoch in range(25):
    #     epoch_train_accuracy.append(train(model, epoch, optimizer, criterion, trainloader))
    #     if epoch % 10 == 0 and epoch > 0:
    #         test(model, testloader)
    model_type = 'Attention'
    positional_encoding = True
    kernel_size = 10
    epoch_train_accuracy1  = run_model(model_type, positional_encoding, kernel_size)
    positional_encoding = False
    epoch_train_accuracy2 = run_model(model_type, positional_encoding, kernel_size)
    plt.figure(figsize=(10, 5))
    plt.title("Training Accuracy vs Epoch")
    plt.plot(epoch_train_accuracy1,label=f"{model_type} train accuracy with positional encoding")
    plt.plot(epoch_train_accuracy2,label=f"{model_type} train accuracy without positional encoding ")
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.legend()
    plt.show()

    model_type = 'Convolution'
    positional_encoding = True
    kernel_size = 10
    epoch_train_accuracy1 = run_model(model_type, positional_encoding, kernel_size)
    kernel_size = 3
    epoch_train_accuracy2 = run_model(model_type, positional_encoding, kernel_size)
    plt.figure(figsize=(10, 5))
    plt.title("Training Accuracy vs Epoch")
    plt.plot(epoch_train_accuracy1,label=f"{model_type} train accuracy with Kernel size 10")
    plt.plot(epoch_train_accuracy2,label=f"{model_type} train accuracy with Kernel size 3")
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.legend()
    plt.show()

    

    
