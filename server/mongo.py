from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os

uri = os.getenv('MONGOKEY')
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")

    db = client['Cluster0']
    collection = db['cards']
    # collection.insert_one(dummy_card)

except Exception as e:
    print(e)

def get_cards():
    db = client['Cluster0']
    collection = db['cards']
    return list(collection.find({}, {'_id': 0}))

def add_card(json):
    db = client['Cluster0']
    collection = db['cards']
    collection.insert_one(json)
