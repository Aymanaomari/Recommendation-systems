import pandas as pd
from app import db

# Function to fetch data from MongoDB and clean it
def get_data_from_mongo():
    collection = db["reviews"]  # Access the 'reviews' collection
    data = list(collection.find())  # Fetch all data from the collection
    df = pd.DataFrame(data)  # Convert data to a DataFrame
    df = df[['reviewerID', 'asin', 'overall']].dropna()  # Clean the data
    return df

# Function to fetch product data from MongoDB
def get_product_data_from_mongo():
    collection = db["products"]  # Access the 'products' collection
    data = list(collection.find())  # Fetch all data from the collection
    df = pd.DataFrame(data)  # Convert data to a DataFrame
    df.rename(columns={'_id': 'asin'}, inplace=True)  # Rename _id to asin
    return df
