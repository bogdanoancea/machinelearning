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

ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))
Y = ngram_vectorizer.fit_transform(corpus)
Y_array=Y.toarray()
Y
Y_array
ngram_vectorizer.get_feature_names_out()
ngram_vectorizer.vocabulary_.get('monday')

