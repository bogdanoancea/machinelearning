# content based recommender

import pandas as pd

# Load the dataset
metadata = pd.read_csv('/Users/bogdanoancea/OneDrive/Scoala/BazeleInformaticii/MachineLearning/Saptamana9/movies_metadata.csv', low_memory=False)


# Create a content-based recommender
def content_based_recommender(title, metadata):
    # Select features
    features = ['title', 'genres', 'tagline', 'overview']

    # Create a DataFrame with selected features
    content = metadata[features]

    # Drop rows with missing values
    content = content.dropna()

    # Lowercase strings and remove spaces
    content['title'] = content['title'].str.lower()
    content['genres'] = content['genres'].str.lower()
    content['tagline'] = content['tagline'].str.lower()
    content['overview'] = content['overview'].str.lower()

    # Combine features into a single string
    content['combined'] = content['genres'] + ' ' + content['tagline'] + ' ' + content['overview']

    # Import TfIdfVectorizer from scikit-learn
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Define a TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Construct the TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(content['combined'])

    # Import linear_kernel to compute the dot product
    from sklearn.metrics.pairwise import linear_kernel

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Get the index of the movie that matches the title
    indices = pd.Series(content.index, index=content['title']).drop_duplicates()
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return content['title'].iloc[movie_indices]


# Get recommendations for a movie
recommended_movies = content_based_recommender('avatar', metadata)
print(recommended_movies)