from flask import Flask
from pymongo import MongoClient
from app.config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize PyMongo without Flask-PyMongo, using pymongo directly
client = MongoClient(app.config['MONGO_URI'])
db = client[app.config['DB_NAME']]

# Import routes after initializing the app
from app import routes
