import csv
import pandas as pd
from collections import Counter

# Read the csv file
df = pd.read_csv('keywords.csv')

# Initialize an empty dictionary to store word frequencies per article
word_freq_per_article = {}

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    # Get the current article
    article = row['article']

    # Get the words in the current row's 'top_words' field
    words = row['top_words'].split(", ")

    # If this article is not yet in the dictionary, add it
    if article not in word_freq_per_article:
        word_freq_per_article[article] = Counter()

    # Update the word frequencies for this article
    word_freq_per_article[article].update(words)

# Now word_freq_per_article is a dictionary where the keys are article ids
# and the values are Counter objects with word frequencies

# For each article, print the 5 most common words
for article in word_freq_per_article:
    print(f"Article: {article}")
    for word, freq in word_freq_per_article[article].most_common(5):
        print(f"{word}")
    print()
