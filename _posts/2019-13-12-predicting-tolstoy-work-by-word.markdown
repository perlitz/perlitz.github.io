---
title: "Predicting Tolstoy"
layout: post
date: 2019-12-13 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- RNN
- PyTorch
- NLP
  star: false
  category: blog
  author: yotam
  description: A simple character prediction excercise 
---
  Begin with the imports:

```python
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
```

Loading the data:

```python
with open('/content/drive/My Drive/Colab_Data/PT_udacity/anna.txt', 'r') as f:
    text = f.read()
```

Tokanize the letters:

```python
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

encoded = np.array([char2int[ch] for ch in text])
```

And one-hot encode them:

```python
def one_hot_encode(arr, n_labels):
    
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot
```

Next, mini-batching:

```python
def get_batches(arr, batch_size, seq_length):

    n_batches = int(np.floor(len(arr)/(batch_size*seq_length)))
    print(n_batches)

    arr = arr[0:(n_batches*batch_size*seq_length)]
    
    arr = arr.reshape(batch_size, seq_length*n_batches)
    
    for n in range(0, arr.shape[1], seq_length):
        x = arr[:,n:n+seq_length]
        y = np.zeros_like(x)

        try:
          y[:, :-1], y[:, -1] = x[:,1:], arr[:,n+seq_length]
        except IndexError:
          y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
          
        yield x, y
```

Varifying if a gpu is present:

```python
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')
```

And defining the network:

```python
class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        self.lstm = nn.LSTM(input_size=len(self.chars), 
                            hidden_size=n_hidden, 
                            num_layers=n_layers, 
                            dropout=drop_prob, 
                            batch_first=True)
        
        self.fc = nn.Linear(in_features=n_hidden, 
                            out_features=len(self.chars))
        
        self.drop = nn.Dropout(drop_prob)

    
    def forward(self, x, hidden):
                

        out, hidden = self.lstm(x, hidden)
        out = out.contiguous().view(-1, 
        out = self.drop(out)
        out = self.fc(out)

       
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        
        self.batch_size = batch_size
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden
        
```

Train:

```python
def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
  
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if(train_on_gpu):
        net.cuda()
    
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):

        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            

            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()


            h = tuple([each.data for each in h])


            net.zero_grad()

            output, h = net(inputs, h)

            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):

                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                
                    val_losses.append(val_loss.item())
                
                net.train()
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
```

Instantiating:

```python
n_hidden=512
n_layers=2

net = CharRNN(chars, n_hidden, n_layers)
print(net)
```

And training:

```python
batch_size = 128
seq_length = 200
n_epochs =  20 

train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)
```

Once training is done, one may save the checkpoint:

```python
# change the name, for saving multiple files
model_name = '/content/drive/My Drive/Colab_Data/PT_udacity/rnn_20_epoch.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)
```

Predict:

```python
def predict(net, char, h=None, top_k=None):

        
        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x)
        
        if(train_on_gpu):
            inputs = inputs.cuda()
        
        h = tuple([each.data for each in h])
        out, h = net(inputs, h)

        p = F.softmax(out, dim=1).data
        if(train_on_gpu):
            p = p.cpu() 
            
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
       
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        return net.int2char[char], h
```

And sample:

```python
def sample(net, size, prime='The', top_k=None):
        
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() 
    
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)
```

Lets try this:

```python
print(sample(net, 1000, prime='Anna', top_k=5))
```

Outputs

```
Anna, that
she was not being at her,. He was the profes which
said, and the same stark, all this came to him, and
his stuck of stations and her walte as he was all the
whole armance was the
party where he had been despirated than a little attertion
to tell about a law and the delaction.

"Well, thy wanted, that' docto a last,
and I do now went all her in she had been saying, and an ind the
world all talk all without all and took it with that it, was in the morth
that she had to get to be a law to me a class of table, and
he said to the party to the churcester."

"It's a glad is a man shand of it.... Why'ne
with him. And we have been assorute you, this is a sudden thit the promist to the men. He could be, and they
have to do to me, and I should be a sudmen of marraing
this since any making or the face of the
mothings of the meanines..

At which were way, thinking about an official their angrownation
and head. He do be to see her hears. But
the propesty of tho country announce to me that he
c
```

Or:

```python
print(sample(net, 2000, top_k=5, prime="And Levin said"))
```

And:

```
And Levin said the
room, and the study had thought of the cross, she will have some
sisters and the people he
considered horsoly to him. The desiling wither they was a cornical that always
world in the sudes of the place of hours all the people
were to borely the cheater as this so to the
country and the steps out of it as
to be to bring the point of tears, and still all the
wailing often that the
parent has because the clum with them, and the man
said it, becare in the marshal, and
were alroading the portrait of
the meadout of her fingers, and
a stand of a signist as he had a passion and the subject. The more
sawing to hers about his wife towands in the monthan, was standing horsed and as so that it walked to
another at the same time of to see the
priscions of the solt, as she was at the province all she was
a presting of his wife. He seemed to him that he had need
that a money were thinking of him and to see a children of a smell on his
hat in, he was thinking of his wife, and should have
been attemptiage with the stronger and a particulous son of
the stury, and he could not certain the subject.

"What is so see that you walk out with him. He's a son to me all the mind,
though, and I see, as he can a speaking about the care of her. As the sort of
the carriage, is
that is there as in the meaning," answered Varenka
sitting out to all about the fact, that he did not his hand.

"I've always see her, I've sort to me. And him as it all a might be
a man, become. You
know a silence and the might been
a chief clusiously friend."

She was so such a singing what were surprised at his way, he was already. And
so that without her share
with a country's past of sume women, and a get alone and hope and the subject
was, but at the chief state.
 "You country..." said Sergozy

an a conversation at her.

"I don't see."

"I am going to mo alone, I delichted your."

She saw that it was not talking at the person of homity.

"It's that is the stinglined of the country of the meating of
the position. H
```



