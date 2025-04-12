from typing import List, Dict, Any
from app.utils.db import get_async_connection
from app.models import Paper

class DatabaseService:
    def __init__(self):
        self.connection = None
        self.require_abstract = False

    async def connect(self):
        if not self.connection:
            self.connection = await get_async_connection()

    async def close(self):
        if self.connection:
            await self.connection.close()
            self.connection = None

    async def get_papers(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve papers by their IDs."""
        await self.connect()
        if not paper_ids:
            return []
        
        try:
            query = """
                SELECT id, title, abstract, url
                FROM papers 
                WHERE id = ANY($1)
            """
            if self.require_abstract:
                query += " AND abstract IS NOT NULL AND abstract <> ''"
                
            rows = await self.connection.fetch(query, paper_ids)
            return [
                {
                    'paper_id': str(row['id']), 
                    'title': row['title'],
                    'abstract': row['abstract'] or "", 
                    'url': row['url']
                }
                for row in rows
            ]
        except Exception as e:
            print(f"Database error: {str(e)}")
            return []

    async def get_paper_by_id(self, paper_id: str) -> Dict[str, Any]:
        """Retrieve a single paper by ID."""
        await self.connect()
        try:
            row = await self.connection.fetchrow(
                """
                SELECT id, title, abstract, url
                FROM papers 
                WHERE id = $1
                """, 
                paper_id
            )
            if not row:
                return None
            return {
                'paper_id': str(row['id']), 
                'title': row['title'],
                'abstract': row['abstract'] or "",
                'url': row['url']
            }
        except Exception as e:
            print(f"Database error: {str(e)}")
            return None

    async def update_paper(self, paper: Dict[str, Any]) -> bool:
        """Update an existing paper."""
        await self.connect()
        try:
            await self.connection.execute(
                """
                UPDATE papers 
                SET title = $1, abstract = $2, url = $3
                WHERE id = $4
                """,
                paper['title'],
                paper['abstract'],
                paper['url'],
                paper['paper_id']
            )
            return True
        except Exception as e:
            print(f"Database error: {str(e)}")
            return False

    async def create_paper(self, paper: Dict[str, Any]) -> bool:
        """Create a new paper."""
        await self.connect()
        try:
            await self.connection.execute(
                """
                INSERT INTO papers (id, title, abstract, url)
                VALUES ($1, $2, $3, $4)
                """,
                paper['paper_id'],
                paper['title'],
                paper['abstract'],
                paper['url']
            )
            return True
        except Exception as e:
            print(f"Database error: {str(e)}")
            return False
            
    async def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper by ID."""
        await self.connect()
        try:
            await self.connection.execute(
                """
                DELETE FROM papers 
                WHERE id = $1
                """,
                paper_id
            )
            return True
        except Exception as e:
            print(f"Database error: {str(e)}")
            return False 