import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from surprise import dump
from app.models import get_data_from_mongo



# Collaborative Filtering Recommendations (Cosine Similarity based)
def similar_users(user_index, interactions_matrix, similarity_threshold=0.2):
    similarity_matrix = cosine_similarity(interactions_matrix)
    user_index = interactions_matrix.index.get_loc(user_index)
    similarity_scores = similarity_matrix[user_index]
    sorted_users = np.argsort(similarity_scores)[::-1]
    similar_users = []
    for idx in sorted_users:
        if similarity_scores[idx] > similarity_threshold and idx != user_index:
            similar_users.append((interactions_matrix.index[idx], similarity_scores[idx]))
    return similar_users

def collaborative_filtering_recommendations(user_index, num_of_products):
    from app.models import get_data_from_mongo
    df_final = get_data_from_mongo()
    sparse_matrix = pd.pivot_table(df_final, values='overall', index='reviewerID', columns='asin', fill_value=0)
    interactions_matrix = csr_matrix(sparse_matrix)
    similar_users_list = similar_users(user_index, sparse_matrix, 0.1)
    interacted_products = set(sparse_matrix.columns[np.where(sparse_matrix.loc[user_index] > 0)])
    recommendations = []
    observed_interactions = interacted_products.copy()
    for similar_user, _ in similar_users_list:
        similar_user_prod_ids = set(sparse_matrix.columns[np.where(sparse_matrix.loc[similar_user] > 0)])
        new_recommendations = list(similar_user_prod_ids.difference(observed_interactions))
        recommendations.extend(new_recommendations)
        observed_interactions.update(similar_user_prod_ids)
        if len(recommendations) >= num_of_products:
            break
    return recommendations[:num_of_products]

# KNN Recommendations using exported ratings_matrix.pkl

def knn_recommendations(user_id, n=5):
    ratings_matrix = pd.read_pickle('/app/models/ratings_matrix.pkl')
    if user_id not in ratings_matrix.index:
        return []
    user_ratings = ratings_matrix.loc[user_id]
    # Recommend items the user hasn't rated yet, sorted by predicted rating
    unrated_items = user_ratings[user_ratings.isna()]
    predicted_ratings = user_ratings.dropna()
    top_n = predicted_ratings.sort_values(ascending=False).head(n)
    return top_n.index.tolist()

# Rank-based Recommendations
def rank_based_recommendations(n=5, min_interaction=20):
    df_final = get_data_from_mongo()
    df_final['overall'] = pd.to_numeric(df_final['overall'], errors='coerce')
    df_final["overall"] = df_final['overall'].fillna(df_final['overall'].median())
    average_rating = df_final.groupby('asin')['overall'].mean()
    count_rating = df_final.groupby('asin')['overall'].count()
    final_rating = pd.DataFrame({'avg_rating': average_rating, 'rating_count': count_rating})
    final_rating['weighted_avg'] = final_rating['avg_rating'] * np.log(final_rating['rating_count'] + 1)
    final_rating = final_rating.sort_values(by='weighted_avg', ascending=False)
    recommendations = final_rating[final_rating['rating_count'] > min_interaction]
    max_interaction = final_rating['rating_count'].max()
    threshold = max_interaction * 0.1
    recommendations = recommendations[recommendations['rating_count'] >= threshold]
    return recommendations.index.tolist()[:n]


def item_based_recommendations(user_id, n=5, min_interaction=20):
    # Get the data
    df_final = get_data_from_mongo()
    df_final['overall'] = pd.to_numeric(df_final['overall'], errors='coerce')
    df_final["overall"] = df_final['overall'].fillna(df_final['overall'].median())

    # Create a pivot table: Users as rows, Products (ASIN) as columns, and Ratings as values
    pivot_table = df_final.pivot_table(index='user_id', columns='asin', values='overall', aggfunc='mean')

    # Calculate the cosine similarity matrix between items (products)
    similarity_matrix = cosine_similarity(pivot_table.fillna(0).T)  # Transpose so we compare items
    similarity_df = pd.DataFrame(similarity_matrix, index=pivot_table.columns, columns=pivot_table.columns)

    # Filter items with minimum interaction
    item_counts = df_final.groupby('asin')['overall'].count()
    eligible_items = item_counts[item_counts >= min_interaction].index
    similarity_df = similarity_df[eligible_items][eligible_items]

    # Get the products that the user has rated
    user_rated_items = df_final[df_final['user_id'] == user_id]['asin'].unique()

    # Rank items based on similarity and weighted by interaction count
    recommendations = {}
    for item in user_rated_items:
        similar_items = similarity_df[item]
        similar_items = similar_items.sort_values(ascending=False)
        recommended_items = similar_items.index.tolist()[:n]  # Top N similar items
        recommendations[item] = recommended_items

    return recommendations