## Bag of Words (BoW)

* **Core Purpose:** Convert text (words or sentences) into numerical representations.
* **Problem It Solves:** Transforms text (non-numeric) into a format usable by machine learning models.

### **What It Solves:**

1. **Represents Words as Numbers:** Each unique word in the text becomes a column in a vector.
2. **Represents Sentences as Numbers:** Each sentence is transformed into a vector of word counts, maintaining a fixed-size numerical format.

### **Pros:**

* **Simple and Intuitive:** Easy to understand and implement.
* **Fast for Small Datasets:** Efficient for basic text processing tasks.
* **Compatible with Basic Models:** Works well with Naive Bayes, SVM, Logistic Regression.

### **Cons:**

* **Ignores Context:** Word order and sentence meaning are lost.
* **High Dimensionality:** Large vocabulary leads to sparse vectors.
* **No Semantic Understanding:** Treats words independently.


Why the Name Bag of Words?
It just not considers the frequency of words
But also if the words being checked for exists
Similar to a physical, where things are placed in without care 
    for arrangement