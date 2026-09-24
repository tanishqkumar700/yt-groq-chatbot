import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
LLM_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.2