import os
from datetime import datetime
from app.services.pdf.file import PDFFile
from app.services.pdf.text_extraction import TextExtractionService
from app.services.embedding import EmbeddingService
from app.services.database import DatabaseService
from typing import List, Dict, Any
from app.utils.db import get_async_connection
import numpy as np
import json


# TODO allow for persistent cloud storage 
class LocalStorage:
    def __init__(self, base_path: str = "/app/uploads"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        os.chmod(self.base_path, 0o777) 
        self.text_service = TextExtractionService()
        self.embedding_service = EmbeddingService()
        self.database_service = DatabaseService()
        
    def _get_user_path(self, user_id: str) -> str:
        """Get the base path for a user's uploads"""
        user_path = os.path.join(self.base_path, user_id)
        os.makedirs(user_path, exist_ok=True)
        os.chmod(user_path, 0o777)
        return user_path
    
    def _generate_storage_path(self, pdf_file: PDFFile) -> str:
        """Generate a unique storage path for the file"""
        timestamp = pdf_file.upload_time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            pdf_file.user_id,
            f"{timestamp}_{pdf_file.safe_filename}"
        )
    
    async def save_file(self, pdf_file: PDFFile, paper_id: str, metadata: Dict[str, Any] = None, full_text: str = "") -> str:
        """Save file to storage and return the full path"""
        # Use provided metadata or extract it if not provided
        if metadata is None:
            try:
                metadata = await self.text_service.extract_metadata_from_pdf(pdf_file) or {}
                full_text = await self.text_service.extract_full_text_from_pdf(pdf_file) or ""
            except Exception as e:
                print(f"Error extracting metadata: {str(e)}")
                metadata = {}
                full_text = ""
        
        title = metadata.get('title', '') if metadata else ''
        abstract = metadata.get('abstract', '') if metadata else ''
        authors = metadata.get('authors', []) if metadata else []
        doi = metadata.get('doi', '') if metadata else ''
        
        # Ensure authors is a list and convert to JSON string
        if isinstance(authors, str):
            authors = [authors]
        authors_json = json.dumps(authors)
        
        content = await pdf_file.get_content()
        storage_path = self._generate_storage_path(pdf_file)
        full_path = os.path.join(self.base_path, storage_path)
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        os.chmod(os.path.dirname(full_path), 0o777)
        
        with open(full_path, "wb") as f:
            f.write(content)
        
        # Generate embedding for title + abstract
        combined_text = f"{title}\n{abstract}".strip()
        embedding = await self.embedding_service.get_embedding_async(combined_text, normalize=True)
        
        # Format embedding for PostgreSQL vector type
        embedding_str = f"[{','.join(map(str, embedding))}]"
        
        # Start transaction and update both tables
        conn = await get_async_connection()
        try:
            async with conn.transaction():
                # Update user_papers
                await conn.execute(
                    """
                    INSERT INTO user_papers (user_id, paper_id, file_path, title, abstract, authors, full_text, upload_date)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (user_id, paper_id) DO UPDATE
                    SET file_path = $3, title = $4, abstract = $5, authors = $6, full_text = $7, upload_date = $8
                    """,
                    pdf_file.user_id, paper_id, full_path, title, abstract, authors_json, full_text, pdf_file.upload_time
                )

                # Insert into papers table with auto-incrementing ID
                # Use paper_id as DOI if available, otherwise NULL
                await conn.execute(
                    """
                    INSERT INTO papers (doi, title, abstract, url, authors, published_date, embedding)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::vector)
                    ON CONFLICT (doi) DO UPDATE
                    SET title = $2, abstract = $3, url = $4, authors = $5::jsonb, published_date = $6, embedding = $7::vector
                    """,
                    doi if doi else None,  # Only use DOI if available
                    title, 
                    abstract, 
                    f"/uploads/{pdf_file.user_id}/{os.path.basename(full_path)}", 
                    authors_json, 
                    pdf_file.upload_time, 
                    embedding_str
                )
        finally:
            await conn.close()
            
        return full_path
    
    async def delete_file(self, user_id: str, paper_id: str) -> bool:
        """Delete file from storage using user_id and paper_id."""
        try:
            conn = await get_async_connection()
            try:
                async with conn.transaction():
                    result = await conn.fetchrow(
                        "SELECT file_path FROM user_papers WHERE user_id = $1 AND paper_id = $2",
                        user_id, paper_id
                    )
                    
                    if result and result['file_path']:
                        file_path = result['file_path']
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            # Delete from user_papers
                            await conn.execute(
                                "DELETE FROM user_papers WHERE user_id = $1 AND paper_id = $2",
                                user_id, paper_id
                            )
                            # Delete from papers using DOI or URL pattern for user-generated papers
                            await conn.execute(
                                """
                                DELETE FROM papers 
                                WHERE doi = $1 OR (doi IS NULL AND url LIKE $2)
                                """,
                                paper_id,
                                f"/uploads/{user_id}/%"
                            )
                            return True
            finally:
                await conn.close()
            return False
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
            return False

    async def get_user_papers(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all papers stored for a user."""
        papers = []
        conn = await get_async_connection()
        try:
            rows = await conn.fetch(
                """
                SELECT paper_id, file_path, title, abstract, authors, full_text, upload_date 
                FROM user_papers 
                WHERE user_id = $1
                ORDER BY upload_date DESC
                """,
                user_id
            )
            
            for row in rows:
                if os.path.exists(row['file_path']):
                    papers.append({
                        'paper_id': row['paper_id'],
                        'title': row['title'] or os.path.splitext(os.path.basename(row['file_path']))[0],
                        'abstract': row['abstract'] or '',
                        'authors': row['authors'] or '',
                        'full_text': row['full_text'] or '',
                        'url': f"/uploads/{user_id}/{os.path.basename(row['file_path'])}",
                        'upload_date': row['upload_date']
                    })
        finally:
            await conn.close()
        
        return papers 