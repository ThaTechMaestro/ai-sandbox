import nltk
nltk.download('punkt_tab')
import numpy as np

text = "I love programming in Python. Python is great for data science."
# Tokenize the text into words
tokens = nltk.word_tokenize(text)
print("Tokens:", tokens)