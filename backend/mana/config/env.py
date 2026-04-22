
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
FRED_API_KEY = os.environ.get("FRED_API_KEY")

DATA_FOLDER = Path(__file__).parent.parent.parent / "data"