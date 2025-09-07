# BD
import os
from pymongo.server_api import ServerApi
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv() 
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "testdb")

client = AsyncIOMotorClient(
    MONGO_URI,
    server_api=ServerApi("1"),
)
db = client[DATABASE_NAME]
users_coll = db["users"]
sessions_coll = db["sessions"]
