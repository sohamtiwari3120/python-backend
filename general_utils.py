import pinecone
from typing import Any
import dotenv
from dotenv import dotenv_values

def get_retrieval_index(index_name: str):
    
    config = dotenv_values(".env")
    pinecone.init(
        api_key=config["PINECONE_API_KEY"],
        environment=config["PINECONE_ENVIRONMENT"]
    )
    return pinecone.Index(index_name)