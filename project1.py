import nltk
import spacy
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
# python -m spacy download en_core_web_lg

# Load the CNN/DailyMail dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')

# Load Spacy model
nlp = spacy.load('en_core_web_lg')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    # Tokenize words and remove stop words
    word_frequencies = Counter(
        word.lower()
        for sentence in sentences
        for word in word_tokenize(sentence)
        if word.lower() not in stop_words and word.isalpha()
    )
    # Calculate sentence scores
    sentence_scores = {
        sentence: sum(word_frequencies[word.lower()] for word in word_tokenize(sentence))
        for sentence in sentences
    }
    return sentences, sentence_scores


def summarize_text(text, num_sentences=3):
    sentences, sentence_scores = preprocess_text(text)
    # Select the top sentences
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = ' '.join(summary_sentences)
    return summary


# Function to evaluate the summarizer
def evaluate_summarizer(dataset, num_samples=100, num_sentences=3):
    references = []
    predictions = []

    for i in range(num_samples):
        article = dataset['validation'][i]['article']
        reference_summary = dataset['validation'][i]['highlights']

        # Generate summary
        generated_summary = summarize_text(article, num_sentences)

        references.append(reference_summary)
        predictions.append(generated_summary)

    return references, predictions


# Evaluate the summarizer on 100 samples
references, predictions = evaluate_summarizer(dataset)

# Display the first 5 reference summaries and the corresponding generated summaries
for i in range(5):
    print(f"Reference Summary {i + 1}: {references[i]}")
    print(f"Generated Summary {i + 1}: {predictions[i]}\n")