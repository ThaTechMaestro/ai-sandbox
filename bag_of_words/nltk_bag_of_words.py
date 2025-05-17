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

# Step 1: Build Vocabulary from Initial Text
def build_vocabulary(text):
    dataset = nltk.sent_tokenize(text)
    vocabulary = set()

    for sentence in dataset:
        for word in nltk.word_tokenize(sentence):
            word = word.lower().strip(string.punctuation)
            if word:
                vocabulary.add(word)
    
    # Sorted for consistent vector order
    return sorted(list(vocabulary)) 

# Step 2: Convert New Text to Bag of Words 
# Vector Using the Fixed Vocabulary
def text_to_bag_of_words(text, vocabulary):
    dataset = nltk.sent_tokenize(text)
    # Initialize all counts to 0
    word2count = dict.fromkeys(vocabulary, 0) 

    for sentence in dataset:
        for word in nltk.word_tokenize(sentence):
            word = word.lower().strip(string.punctuation)
            
            # Only count words in the fixed vocabulary
            if word in word2count: 
                word2count[word] += 1

    # Generate vector string
    vector = [str(word2count[word]) for word in vocabulary]
    return vector

# Initial Text (Training Text)
initial_text = "Python is great for data science. Coding is fun!"
vocabulary = build_vocabulary(initial_text)

print("Vocabulary (Fixed):", vocabulary)

# Using the Fixed Vocabulary with New Text
# generate Bag of Words Vector
new_text = "Python is amazing. Data science is evolving."
vector = text_to_bag_of_words(new_text, vocabulary)

print("Bag of Words Vector for New Text:")
print(f"[{', '.join(vector)}]")