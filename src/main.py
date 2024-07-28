import os
import pandas as pd
import numpy as np
import tf_funct
import funct

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

response = int(input("Would you like recommendations based on your ratings of movies or find similar movies to a movie you like? (Enter 1 or 2): "))
if (response == 1):
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    movieList_df = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
    movieList_df.columns = ['Rating Count', 'Mean Rating']
    movieList_df = movieList_df.merge(movies[['movieId', 'title']], on='movieId')
    movie_titles = movieList_df['title'].tolist()
    
    R, Y, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = tf_funct.create_matrices(ratings)

    num_movies, num_users = Y.shape
    movie_idx = dict(zip(movie_titles, range(num_movies)))

    num_rate = int(input("How many movies would you like to rate? "))
    movie_indices = []
    ratings_given = []
    for i in range(num_rate):
        m = input("Please enter movie name: ")
        m = funct.movie_finder(m)
        rating = 0
        while (rating < 0.5 or rating > 5):
            rating = float(input(f"Enter a rating for {m} (1 - 5): "))
        idx = int(movie_idx[m])
        movie_indices.append(idx)
        ratings_given.append(rating)
            
    my_ratings, my_rated = tf_funct.add_ratings(num_movies, movie_indices, ratings_given)
    Y = np.c_[my_ratings, Y]
    R = np.c_[(my_ratings != 0).astype(int), R]
    num_movies, num_users = Y.shape
    Ynorm, Ymean = tf_funct.normalizeRatings(Y, R)

    X, W, b = tf_funct.train(Ynorm, R, num_users, num_movies, 100, iterations = 300)

    my_predictions, ix = tf_funct.predict(X, W, b, Ymean)

    tf_funct.filtered_output(my_ratings, my_rated, my_predictions, ix, movieList_df)
    
else:
    movie = input("Please enter the movie name: ")
    similar_movies, title = funct.get_content_based_recommendations(movie)
    print(f"Since you liked {title}, you may also like:")
    for m in (similar_movies.values):
        print(m)