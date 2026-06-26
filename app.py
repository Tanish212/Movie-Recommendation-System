import pandas as pd
import difflib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies_data = pd.read_csv("movies.csv")
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = (
    movies_data['genres'] + ' ' +
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' +
    movies_data['cast'] + ' ' +
    movies_data['director']
)
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)
def recommend_movies(movie_name):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if len(find_close_match) == 0:
        return None
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data['title'] == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score,key=lambda x: x[1],reverse=True)
    recommended_movies = []
    for movie in sorted_similar_movies[1:11]:
        index = movie[0]
        title_from_index = movies_data[movies_data['index'] == index]['title'].values[0]
        recommended_movies.append(title_from_index)
    return close_match, recommended_movies
st.title("Movie Recommendation System")
movie_name = st.text_input("Enter your favourite movie name")
if st.button("Recommend"):
    result = recommend_movies(movie_name)
    if result is None:
        st.error("Movie not found. Please enter a valid movie name.")
    else:
        close_match, recommendations = result
        st.subheader(f"Movies similar to '{close_match}'")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")