"""
Configuration file for the document ingestion system.
Copy this file to .env and fill in your actual API keys.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "document-embeddings")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# Document Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

# Supported file types
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
