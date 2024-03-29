import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from pprint  import pprint

with open('Data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

type(text)

text[:1000]

all_characters = set(text)
len(all_characters)

for pair in enumerate(all_characters):
    print(pair)

# decoder  num --> letter
decoder = dict(enumerate(all_characters))

encoder = {char: index for index, char in decoder.items()}

pprint(encoder)

# encoder - letter --> num
encoded_text = np.array([encoder[char] for char in text])
encoded_text[:500]

decoder[53]

def one_hot_encoder(encoded_text, num_uni_char):
    # encoded_text  --> batch of encoded text
    # num_uni_char  --> len(set(text))

    one_hot = np.zeros((encoded_text.size, num_uni_char))

    one_hot = one_hot.astype(np.float32)

    one_hot[np.arange(one_hot.shape[0]), encoded_text.flatten()] = 1.0

    one_hot = one_hot.reshape((*encoded_text.shape, num_uni_char))

    return one_hot

arr = np.array([1, 2, 0])
one_hot_encoder(arr, 3)

## 92 - Generating Training Batches
example_text = np.arange(10)
example_text.reshape

def generate_batches(encoded_text, samp_per_batch=10, seq_len=50):
    # X: encoded text of length seqP_len
    # Y: encoded text shifted by one

    # how many chars per batch?
    char_per_batch = samp_per_batch * seq_len

    # how many batches can we make, given the len of encoded text
    num_batches_avail = int(len(encoded_text)/char_per_batch)

    # cut off the end of the encoded text, that won't fit evenly into a batch
    encoded_text = encoded_text[:num_batches_avail*char_per_batch]

    encoded_text = encoded_text.reshape((samp_per_batch, -1))

    for n in range(0, encoded_text.shape[1], seq_len):
        x = encoded_text[:, n:n+seq_len]
        # zeros array to the same shape as x
        y = np.zeros_like(x)
        
        try:
            y[:,:-1] = x[:,1:]
            y[:,-1] = encoded_text[:, n+seq_len]

        except:
            y[:,:-1] = x[:,1:]
            y[:,-1] = encoded_text[:,0]

        yield x, y


sample_text = encoded_text[:20]

batche_generator = generate_batches(sample_text, samp_per_batch=2, seq_len=5)
x, y = next(batche_generator)

# # class 93 - LSTM Model

class CharModel(nn.Module):
    
    def __init__(self, all_chars, num_hidden=256, num_layers=4, drop_prob=.5, use_gpu=False):
        
        super().__init__()

        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.use_gpu = use_gpu

        self.all_chars = all_chars
        self.decoder = dict(enumerate(all_chars))
        self.encoder = {char:ind for ind, char in decoder.items()}

        self.lstm = nn.LSTM(len(self.all_chars), num_hidden, num_layers, dropout=drop_prob, batch_first=True)
          