from typing import Dict, List
from app.utils.db import get_async_connection
from .config import SearchConfig
from app.models import Paper

class DatabaseService:
    def __init__(self, config: SearchConfig):
        self.config = config

    async def get_paper_details(self, paper_ids: List[str]) -> Dict[str, Paper]:
        """Fetches paper details, optionally filtering for valid abstracts."""
        try:
            conn = await get_async_connection()
            if not conn:
                print("Database connection failed!")
                return {}

            async with conn.cursor() as cur:
                if self.config.REQUIRE_ABSTRACT:
                    query = """
                        SELECT id, title, abstract, url
                        FROM public.papers
                        WHERE id = ANY(%s)
                          AND abstract IS NOT NULL
                          AND abstract <> ''
                    """
                else:
                    query = """
                        SELECT id, title, abstract, url
                        FROM public.papers
                        WHERE id = ANY(%s)
                    """
                    
                await cur.execute(query, (paper_ids,))
                rows = await cur.fetchall()
                return {
                    row[0]: Paper(
                        paper_id=row[0],
                        title=row[1], 
                        abstract=row[2] or "",  # Convert None to empty string
                        url=row[3]
                    )
                    for row in rows
                }
        except Exception as e:
            print(f"Database error: {str(e)}")
            return {} 