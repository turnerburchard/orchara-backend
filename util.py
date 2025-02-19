import psycopg2
import os
from dotenv import load_dotenv

def get_connection():
    """
    Establishes a PostgreSQL connection using credentials from the .env file.
    """
    load_dotenv()  # Ensure .env variables are loaded
    return psycopg2.connect(
        dbname=os.environ.get('DB_NAME'),
        user=os.environ.get('DB_USER'),
        password=os.environ.get('DB_PASSWORD'),
        host=os.environ.get('DB_HOST'),
        port=os.environ.get('DB_PORT')
    )