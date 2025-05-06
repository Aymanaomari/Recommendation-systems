import os


class Config:
    # MongoDB URI for connecting to the database
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = "pfa"  # Name of the MongoDB database
