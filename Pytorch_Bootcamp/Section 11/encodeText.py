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

# encoder - letter --> num
encoder = {char: index for index, char in decoder.items()}

pprint(encoder)

encoded_text = np.array([encoder[char] for char in text])
encoded_text[:500]

decoder[27]