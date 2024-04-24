import pandas as pd
import numpy as np
import os
from django.conf import settings
import difflib

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# from "./static/api/books.csv" import books


def get_local_nlp_recommendations(name):

    print(os.getcwd())

    # file_path = os.path.join(settings.STATIC_ROOT, 'my_app', 'my_file.txt')

    with open('api/static/api/books.csv', 'rb') as f:
        df = pd.read_csv(f)

    with open('api/static/api/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    with open('api/static/api/cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)

    indices = pd.Series(df.index, index=df['name']).drop_duplicates()


    if name in df['name'].values:
        idx = indices[name]
    elif name.title() in df['name'].values:
        idx = indices[name.title()]
    elif name.lower() in df['name'].values:
        idx = indices[name.lower()]
    elif name.upper() in df['name'].values:
        idx = indices[name.upper()]
    elif df['name'].str.contains(name, case=False).any():
        idx = df['name'].str.contains(name , case=False).idxmax()
    else:
        return "Sorry, we don't have this book in our database. Please check the spelling or try with some other book."    
    


    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:5]

    book_indices = [i[0] for i in sim_scores]

    return df['name'].iloc[book_indices]

# print(get_local_nlp_recommendations('game'))

def get_collaborative_filtering_recommendations(searchTerm):

    with open('api/static/api/model_knn.pkl', 'rb') as f:
        model = pickle.load(f)
    
    rating_popular_books = pd.read_csv('api/static/api/rating_popular_book.csv')
    rating_popular_book_pivot = rating_popular_books.pivot(index='title', columns='user_id', values='rating').fillna(0)

    if searchTerm not in rating_popular_book_pivot.index:
        # Find the closest Match
        matches = difflib.get_close_matches(searchTerm, rating_popular_book_pivot.index, n=1)

        try:
            if len(matches) > 0:
                searchTerm = matches[0]
            else:
                searchTerm = rating_popular_book_pivot.index[rating_popular_book_pivot.index.str.contains(searchTerm, case=False)].tolist()[0]
        except:
            print("Book not found. Please try again.")
            res = []
            res.append("Book not found. Please try again.")
            return res
        
    distances, indices = model.kneighbors(rating_popular_book_pivot.loc[searchTerm].values.reshape(1, -1), n_neighbors = 8)

    res = []

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(searchTerm))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, rating_popular_book_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
            # res.append(rating_popular_book_pivot.index[indices.flatten()[i]])
            res.append([rating_popular_book_pivot.index[indices.flatten()[i]] , distances.flatten()[i]])
    
    return res

    
        









    


