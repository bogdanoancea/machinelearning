import pandas as pd
import numpy as np
import nltk
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import fasttext
from transformers import BertTokenizer, BertModel
import torch
import tensorflow as tf

nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('pricerunner_aggregate.csv')

# Initialize Spacy
nlp = spacy.load('en_core_web_lg')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Tokenize and remove stop words
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Lemmatize tokens using Spacy
    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]

    return lemmatized_tokens


# Apply preprocessing to the review text
df['cleaned_text'] = df['Product Title'].apply(preprocess_text)

# Display the first few rows with cleaned text
print(df.head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df[' Category ID'], test_size=0.2, random_state=42)

# Initialize the CountVectorizer
count_vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Prepare data for Word2Vec
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
X_train_seq_padded = pad_sequences(X_train_seq, padding='post')
X_test_seq_padded = pad_sequences(X_test_seq, padding='post', maxlen=X_train_seq_padded.shape[1])

# Define and train the Word2Vec model using TensorFlow
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=X_train_seq_padded.shape[1])
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train_seq_padded, np.zeros((X_train_seq_padded.shape[0], embedding_dim)), epochs=1, verbose=1)

# Extract word vectors
word2vec_weights = model.get_weights()[0]


def get_word2vec_vectors(texts):
    vectors = []
    for text in texts:
        word_vectors = [word2vec_weights[tokenizer.word_index[word]] for word in text if word in tokenizer.word_index]
        vectors.append(np.mean(word_vectors, axis=0) if word_vectors else np.zeros(embedding_dim))
    return np.array(vectors)


X_train_w2v = get_word2vec_vectors(X_train)
X_test_w2v = get_word2vec_vectors(X_test)

# Prepare data for FastText
with open("fasttext_train.txt", "w") as f:
    for text in X_train:
        f.write(" ".join(text) + "\n")

# Train FastText model
ft_modelSK = fasttext.train_unsupervised("fasttext_train.txt", model='skipgram', dim=100)


def get_fasttext_vectors(texts):
    vectors = []
    for text in texts:
        word_vectors = [ft_modelSK[word] for word in text]
        vectors.append(np.mean(word_vectors, axis=0))
    return np.array(vectors)


X_train_ft = get_fasttext_vectors(X_train)
X_test_ft = get_fasttext_vectors(X_test)

# Load GloVe embeddings
glove_vectors = {}
with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_vectors[word] = vector


def get_glove_vectors(texts):
    vectors = []
    for text in texts:
        word_vectors = [glove_vectors[word] for word in text if word in glove_vectors]
        vectors.append(np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100))
    return np.array(vectors)


# Vectorize the text
X_train_glove = get_glove_vectors(X_train)
X_test_glove = get_glove_vectors(X_test)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_vectors(texts):
    vectors = []
    for text in texts:
        inputs = tokenizer(" ".join(text), return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        vectors.append(outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten())
    return np.array(vectors)


# Vectorize the text
X_train_bert = get_bert_vectors(X_train)
X_test_bert = get_bert_vectors(X_test)

# Combine all vectors into a single array for visualization
X_combined = np.concatenate(
    [X_train_counts.toarray(), X_train_tfidf.toarray(), X_train_w2v, X_train_ft, X_train_glove, X_train_bert], axis=1)

X_combined = np.concatenate([X_train_counts.toarray(), X_train_tfidf.toarray()], axis=1)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_combined)

# Visualize the result
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_train, palette='viridis')
plt.title("t-SNE Visualization of Product Reviews")
plt.show()


# Train the Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print(classification_report(y_test, y_pred))