import os
import asyncpg

async def get_async_connection():
    return await asyncpg.connect(
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT', '5432')
    )

async def ensure_user_papers_table_exists():
    """Ensure the user_papers table exists in the database."""
    conn = await get_async_connection()
    try:
        # Check if the table exists
        table_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'user_papers'
            )
            """
        )
        
        if not table_exists:
            # Create the user_papers table with all required columns
            await conn.execute(
                """
                CREATE TABLE user_papers (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    title TEXT,
                    abstract TEXT,
                    authors TEXT,
                    full_text TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, paper_id)
                )
                """
            )
            print("Created user_papers table with title, abstract, authors, and full_text columns")
        else:
            print("user_papers table already exists")
    finally:
        await conn.close() 