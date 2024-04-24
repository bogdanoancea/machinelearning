from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import PredictionImpossible

# Load the MovieLens 100k dataset
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)

# Configure the algorithm
sim_options = {
    'name': 'cosine',
    'user_based': True  # Compute similarities between users
}
algo = KNNBasic(sim_options=sim_options)

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Calculate and print the accuracy measures
accuracy.rmse(predictions)
accuracy.mae(predictions)

# Function to get top-N recommendations for a given user
def get_top_n_recommendations(predictions, user_id, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid == user_id:
            if uid not in top_n:
                top_n[uid] = []
            top_n[uid].append((iid, est))

    # Sort the predictions for each user and retrieve the highest rated items
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n[user_id]

# Assume we want recommendations for user '196' and we want the top 10 recommendations
user_id = '196'
top_recommendations = get_top_n_recommendations(predictions, user_id, n=10)
print(f"Top 10 recommendations for User {user_id}:")
for movie_id, predicted_rating in top_recommendations:
    print(f"Movie ID: {movie_id}, Predicted Rating: {predicted_rating}")

import os


def load_movie_titles(file_path):
    """
    Loads movie titles from the u.item file from MovieLens 100k dataset.

    Args:
        file_path (str): The path to the u.item file.

    Returns:
        dict: A dictionary mapping movie IDs to movie titles.
    """
    movie_titles = {}
    with open(file_path, 'r', encoding='ISO-8859-1') as file:  # Using ISO-8859-1 encoding
        for line in file:
            parts = line.split('|')  # Split the line into parts
            movie_id = parts[0]
            title = parts[1]
            movie_titles[movie_id] = title
    return movie_titles


# Path to the u.item file (modify this path according to your local setup)
item_file_path = 'path_to_ml_100k/u.item'

# Load movie titles
movie_titles = load_movie_titles(item_file_path)


# Function to get a movie title by movie ID
def get_movie_title(movie_id, movie_titles):
    """
    Retrieves the movie title for a given movie ID.

    Args:
        movie_id (str): The movie ID as a string.
        movie_titles (dict): A dictionary of movie IDs to titles.

    Returns:
        str: The movie title, or None if the movie ID is not found.
    """
    return movie_titles.get(movie_id, "Movie ID not found")


# Example usage: Get the title of movie with ID '50'
movie_id = '50'
title = get_movie_title(movie_id, movie_titles)
print(f"Movie ID {movie_id} corresponds to the title: '{title}'")
