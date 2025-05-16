import nltk
nltk.download('punkt')
import string

"""
Bag of Words (BoW) Model

- Purpose: Convert text (sentence/paragraph) into a frequency-based numeric format.
- Process:
    1. Tokenization (using NLTK):
        - Split text into sentences
        - Split each sentence into words.
    
    2. Word Vocabulary Creation:
        - Create a dictionary to store word counts.
        - Key: word
        - Value: word count

    3. Counting Words:
        - For each tokenized word:
            - If the word is not in the dictionary, add it with a count of 1.
            - If the word is already in the dictionary, increment its count by 1.
    
- Output: A dictionary representing word frequencies in the text.
"""


# Input Text
text = "Python is great for data science. Coding is fun!"

# Tokenize text into sentences
dataset = nltk.sent_tokenize(text)
print("Tokenized Sentences:", dataset)

# Initialize dictionary for word counting
word2count = {}

# Tokenize words in each sentence and count
for sentence in dataset:
    for word in nltk.word_tokenize(sentence):
        # Lowercase and remove punctuation
        word = word.lower().strip(string.punctuation) 
        if word:  # Ignore empty words
            word2count[word] = word2count.get(word, 0) + 1

# Display the final Bag of Words (word frequency)
print("Word Frequency (Bag of Words):")
print(word2count)

