import nltk
nltk.download('punkt_tab')
import numpy as np
import re

'''
Basically:
    Bag of words 
Bag of Words
    Using NLTK
    1. Tokenize the text into words
'''

text = "I love programming in Python. Python is great for data science. Coding is fun!"
# Tokenize the text into words
dataset = nltk.sent_tokenize(text)
    
dataset = nltk.sent_tokenize(text)
for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W', ' ', dataset[i])
    dataset[i] = re.sub(r'\s+', ' ', dataset[i])

print(dataset)

word2count = {}
for sentence in dataset:
    for word in nltk.word_tokenize(sentence):
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
print(word2count)
    