# Text preprocessing

# $pip install nltk

# 1. Tokenization and Text Cleaning

# White space tokenization
sentence = " I was born in Romania in 2000."
sentence.split()
sentence = " I was born in Romania in 2000, I am 24 years old"
sentence.split(',')

# NTLK tokenizer
import nltk.tokenize
from nltk.tokenize import (word_tokenize, sent_tokenize, TreebankWordTokenizer, wordpunct_tokenize, TweetTokenizer, MWETokenizer)

text1 = "Hope, is the only thing stronger than fear! #Hope #Amal.M"

# word tokenization
nltk.download()
print(word_tokenize(text1, language='english'))

# sentence tokenization
print(sent_tokenize(text1))

# word tokenization with punctuation
print(wordpunct_tokenize(text1))

# Treebank word tokenzation
text2 = "What you don't want to be done to yourself, don't do to others..."
tokenizer = TreebankWordTokenizer()
print(tokenizer.tokenize(text2))

# Tweet tokenization
tweet = "Don't take cryptocurrency advice from people on Twitter #advice @crypto"
tokenizer = TweetTokenizer()
print(tokenizer.tokenize(tweet))

# MWE tokenization (multi-word expression tokenizer)
text3 = "Hope, is the only thing stronger than fear! Hunger Games #Hope"
tokenizer = MWETokenizer()
print(tokenizer.tokenize(word_tokenize(text3)))

tokenizer = MWETokenizer()
tokenizer.add_mwe(('Hunger', 'Games'))
print(tokenizer.tokenize(word_tokenize(text3)))

# NLTK word tokenization with cleaning
text4 = "NLP is amazing! Let's explore its wonders."
tokens = nltk.word_tokenize(text4)
cleaned_tokens = [word.lower() for word in tokens if word.isalpha()]
print(cleaned_tokens)

# $pip install textblob
# BlobText tokenization
from textblob import TextBlob

text = " But I'm glad you'll see me as I am. Above all, I wouldn't want people to think that I want to prove anything. I do not want to prove anything, I just want to live to cause no evil to anyone but myself. I have that right, haven't I? Lev Tolstoi."
blob_object = TextBlob(text)

# Word tokenization of the text
text_words = blob_object.words

# See all tokens
print(text_words)
# To count the number of tokens
print(len(text_words))

# $pip install spacy
# spacy tokenization
import spacy
# $ python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')
text = "All happy families are alike; each unhappy family is unhappy in its own way!!! #Lev Tolstoy."
doc = nlp(text)
for token in doc:
    print(token, token.idx)

# $pip install gensim
# gensim tokenization
from gensim.utils import tokenize

list(tokenize(text))

# keras tokenization
# $pip install tensorflow
# $pip install Keras-Preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

ntoken = Tokenizer(num_words=20)
text = "All happy families are alike; each unhappy family is unhappy in its own way!!! #Lev Tolstoy."
ntoken.fit_on_texts(text)
list_words = text_to_word_sequence(text)
print(list_words)

# 2. Stop words removing

# NLTK
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
filtered_sentence = [word for word in cleaned_tokens if word not in stop_words]
print(filtered_sentence)

# spacy
# Load the core English language model
nlp = spacy.load("en_core_web_sm")

# Process the text
text = "This is a sample sentence with some stop words."
doc = nlp(text)

# Remove stop words
filtered_tokens = [token.text for token in doc if not token.is_stop]

# Print the text excluding stop words
print(filtered_tokens)

# Create a set of stop words
stop_words = spacy.lang.en.stop_words.STOP_WORDS


# Define a function to remove stop words from a sentence
def remove_stop_words(sentence):
    # Parse the sentence using spaCy
    doc = nlp(sentence)

    # Use a list comprehension to remove stop words
    filtered_tokens = [token for token in doc if not token.is_stop]

    # Join the filtered tokens back into a sentence
    return ' '.join([token.text for token in filtered_tokens])


sentence = "This is an example sentence with stop words."

filtered_sentence2 = remove_stop_words(sentence)
print(filtered_sentence2)

# gensim
from gensim.parsing.preprocessing import remove_stopwords

# Define a list of words
words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

# Remove the stop words
filtered_words = remove_stopwords(' '.join(words))
# Print the filtered list of words
print(filtered_words)
# Output: ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']


# Specific/Custom stop words removal
stop_words = ["a", "an", "the", "and", "but", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
              "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
              "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
              "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
              "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
              "will", "just"]

# Define a string containing some text
text = "The quick brown fox jumps over the lazy dog."

# Split the string into a list of words
words = text.split()

# Create a new list to hold the filtered words
filtered_words = []

# Iterate over the list of words
for word in words:
    # If the word is not in the stop word list, add it to the filtered list
    if word not in stop_words:
        filtered_words.append(word)

    # Print the filtered list of words
print(filtered_words)
# Output: ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog.']


# 3. Stemming and lematization
# NLTK
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_sentence]
print(stemmed_words)

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language='english')
tokens = ['compute', 'computer', 'computed', 'computing']

for token in tokens:
    print(token + ' --> ' + stemmer.stem(token))

# Lemmatization

# NLTK
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_sentence]
print(lemmatized_words)

# spaCy

sentence6 = nlp('compute computer computed computing')
for word in sentence6:
    print(word.text, word.lemma_)

sentence7 = nlp('A letter has been written, asking him to be released')

for word in sentence7:
    print(word.text + '  ===>', word.lemma_)

# 4. POS tagging
# NLTK
from nltk import pos_tag

pos_tags = nltk.pos_tag(filtered_sentence)
print(pos_tags)

# spacy
# create a spacy document
sentence = nlp('Manchester United is looking to sign a forward for $90 million')
for word in sentence:
    print(word.text)

for word in sentence:
    print(word.text, word.pos_)

for word in sentence:
    print(word.text, word.pos_, word.dep_)

# 5. Named Entity Recognition (NER)
#  NLTK
from nltk import ne_chunk

ner_tags = ne_chunk(pos_tags)
print(ner_tags)

# spacy
# create a spacy document
sentence5 = nlp('Manchester United is looking to sign Harry Kane for $90 million')
for word in sentence5:
    print(word.text)

for entity in sentence.ents:
    print(entity.text + ' - ' + entity.label_ + ' - ' + str(spacy.explain(entity.label_)))

for noun in sentence5.noun_chunks:
    print(noun.text)

# 6. An example: cleaning a tweet
import re


def clean_tweet(tweet):
    tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#\w+', '', tweet)  # Remove hashtags
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    return tweet


tweet = "Loving the new #iPhone! Best phone ever! @Apple https://apple.com"
clean_tweet(tweet)
