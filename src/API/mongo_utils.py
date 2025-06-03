import os
from pymongo import MongoClient

_client = None
_db = None


def get_mongo_client(username='yourUsername', cluster_url='cluster404.letters.mongodb.net', db_name='yourDatabase',
                     password_env_var='MONGODB_ATLAS_PASSWORD'):
    """
    Fetches a MongoDB client. Reuses the existing client if already created.
    """
    global _client, _db

    if _client is not None:
        return _client

    password = os.getenv(password_env_var)
    if not password:
        raise Exception(f"{password_env_var} environment variable not set")

    connection_uri = f"mongodb+srv://{username}:{password}@{cluster_url}/{db_name}?retryWrites=true&w=majority"

    try:
        _client = MongoClient(connection_uri)
        _db = _client[db_name]
        print("Successfully created MongoDB client.")
        return _client
    except Exception as e:
        print(f"Failed to create MongoDB client: {e}")
        return None


def get_collection(collection_name):
    """
    Fetches a MongoDB collection by name. Ensures the client is initialized.
    """
    global _db

    if _client is None:
        raise Exception("MongoDB client is not initialized. Call `get_mongo_client` first.")
    if _db is None:
        raise Exception("Database connection is not initialized.")

    return _db[collection_name]
