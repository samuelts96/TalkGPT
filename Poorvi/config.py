import os
from dotenv import load_dotenv

def load_API_KEY():
    load_dotenv()  # Loads environment variables from the .env file
    return os.getenv("OPENAI_API_KEY")