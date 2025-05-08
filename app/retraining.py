import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import MultiLabelBinarizer
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
from sklearn.impute import SimpleImputer
from surprise import dump
import numpy as np

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["pfa"]
collection = db["reviews"]

# Fetch interaction data (ratings)
data = list(collection.find())
df_final = pd.DataFrame(data)

# Handle missing values in ratings using SimpleImputer (mean imputation)
imputer = SimpleImputer(strategy='mean')
df_final['overall'] = imputer.fit_transform(df_final[['overall']])

# Select relevant columns (ratings data)
df_final = df_final[['reviewerID', 'asin', 'overall']]

# Fetch product data (e.g., asin, category)
product_data = list(db["products"].find())  # Assuming you have a 'products' collection with item features
df_products = pd.DataFrame(product_data)

# Rename '_id' column to 'asin' in both dataframes (to merge correctly)
df_products.rename(columns={'_id': 'asin'}, inplace=True)

# Drop irrelevant columns for the model (e.g., brand, description)
df_products = df_products.drop(columns=["brand", "description", "imUrl", "title"])

# Flatten the categories column if it's a list of lists
df_products['categories'] = df_products['categories'].apply(
    lambda x: [item for sublist in x for item in sublist] if isinstance(x, list) else []
)

# Initialize MultiLabelBinarizer to handle multiple categories per product
mlb = MultiLabelBinarizer()

# Apply binarizer to the 'categories' column for one-hot encoding
category_encoded = mlb.fit_transform(df_products['categories'])

# Create DataFrame with one-hot encoded categories
category_columns = mlb.classes_  # Column names (categories)
category_df = pd.DataFrame(category_encoded, columns=category_columns)

# Merge the one-hot encoded categories with the original product data
df_products = pd.concat([df_products, category_df], axis=1)

# Merge product features (e.g., categories) with the rating data
df_final = df_final.merge(df_products[['asin'] + list(category_columns)], on='asin', how='left')

# Prepare the data for Surprise (Ratings data only)
reader = Reader(rating_scale=(df_final['overall'].min(), df_final['overall'].max()))
data = Dataset.load_from_df(df_final[['reviewerID', 'asin', 'overall']], reader)

# Train/test split
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize SVD++ model
algo = SVDpp()

# Train the model
algo.fit(trainset)

# Test the model
predictions = algo.test(testset)

# Compute RMSE for the test set
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Hyperparameter tuning using cross-validation
# You can customize the parameters for tuning (e.g., n_factors, lr_all, reg_all)
cv_results = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)

# Example of making predictions for a specific user
user_id = 'A1TQMJ4QZK55ND'  # Replace with a valid user ID
item_ids = df_final['asin'].unique()

# Predict ratings for all items for the specific user
user_predictions = [algo.predict(user_id, item_id) for item_id in item_ids]

# Sort predictions by the predicted rating (descending order)
user_predictions.sort(key=lambda x: x.est, reverse=True)

# Get the top N recommendations (e.g., top 10)
top_n = user_predictions[:10]

# Print the recommended items (ASINs)
recommended_items = [item.iid for item in top_n]
print("Top Recommendations for User", user_id, ":", recommended_items)

# Optionally, save the trained model
dump.dump('../svdpp_model_with_item_features', algo=algo)
print("Model saved.")
