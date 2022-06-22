#  ===================
# Example 1-1

# from sklearn.feature_extraction.text import CountVectorizer 
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt

# corpus = ['Time flies flies like an arrow.', 'Fruit flies like a banana.']
# one_hot_vectorizer = CountVectorizer(binary=True) 
# one_hot = one_hot_vectorizer.fit_transform(corpus).toarray() 

# one_hot

# sns.heatmap(one_hot, annot=True, 
#             cbar=False, 
#             yticklabels=['Sentence 2'])
# plt.show()


#  ===================
# Example 1-2

# from sklearn.feature_extraction.text import TfidfVectorizer 
# import seaborn as sns

# tfidf_vectorizer = TfidfVectorizer() 
# tfidf = tfidf_vectorizer.fit_transform(corpus).toarray() 
# sns.heatmap(tfidf, annot=True, 
#             cbar=False, yticklabels= ['Sentence 1', 'Sentence 2'])
# plt.show()


#  ===================
# Example 1-15 - Creating tensors for Gradient bookkeeping
def describe(x): 
    print("Type: {}".format(x.type())) 
    print("Shape/size: {}".format(x.shape)) 
    print("Values: \n{}".format(x))

import torch
x = torch.ones(2, 2, requires_grad=True)
describe(x)

print(x.grad is None)
# ---

y = (x + 2) * (x + 5) + 3
describe(y)
print(x.grad is None)
# ---

z = y.mean()
describe(z)
z.backward()
print(x.grad is None)


#  ===================
# Example 1-16 - Creating CUDA tensors
print(torch.cuda.is_available())


#  ===================
# Exercises
a = torch.rand(3, 3)
a.unsqueeze(0)

a.squeeze(0)

3 + torch.rand(5, 3) * (7-3)

a = torch.rand(3, 3)
a.normal_(3)