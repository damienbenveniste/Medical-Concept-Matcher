import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)