from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
corpus = [
'It is Monday today.',
'It was Sunday yesterday.',
'It will be Tuesday tomorrow.',
'Which day of the week will it be on the day after tomorrow?',]

X = vectorizer.fit_transform(corpus)
X_array=X.toarray()
X
X_array
vectorizer.get_feature_names_out()
vectorizer.vocabulary_.get('monday')

#something new
vectorizer.transform(['Something completely new.']).toarray()


ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))
Y = ngram_vectorizer.fit_transform(corpus)
Y_array=Y.toarray()
Y
Y_array
ngram_vectorizer.get_feature_names_out()
ngram_vectorizer.vocabulary_.get('monday')

analyze = ngram_vectorizer.build_analyzer()
analyze('Bi-grams are cool!') == (['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])


from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
count_matrix = [[3, 0, 1],
[2, 0, 0],
[3, 0, 0],
[4, 0, 0],
[3, 2, 0],
[3, 0, 2]]
tfidf = transformer.fit_transform(count_matrix)
tfidf.toarray()


# Word2Vec
# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action='ignore')


# Reads ‘alice.txt’ file
sample = open("alice.txt")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
	temp = []

	# tokenize the sentence into words
	for j in word_tokenize(i):
		temp.append(j.lower())

	data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count=1,
								vector_size=100, window=5)

# Print results
print("Cosine similarity between 'alice' " +
	"and 'wonderland' - CBOW : ",
	model1.wv.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " +
	"and 'machines' - CBOW : ",
	model1.wv.similarity('alice', 'machines'))

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100,
								window=5, sg=1)

# Print results
print("Cosine similarity between 'alice' " +
	"and 'wonderland' - Skip Gram : ",
	model2.wv.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " +
	"and 'machines' - Skip Gram : ",
	model2.wv.similarity('alice', 'machines'))
