import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from surprise import dump

# Load the saved SVD model
model_filename = './svdpp_model_with_item_features'
_, algo_svd = dump.load(model_filename)

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

# SVD Recommendations
def svd_recommendations(user_id, n=5):
    from app.models import get_data_from_mongo
    df_final = get_data_from_mongo()
    product_ids = df_final['asin'].unique()
    recommendations = []
    for product_id in product_ids:
        prediction = algo_svd.predict(user_id, product_id)
        recommendations.append((product_id, prediction.est))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]

# Rank-based Recommendations
def rank_based_recommendations(n=5, min_interaction=20):
    from app.models import get_data_from_mongo
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
