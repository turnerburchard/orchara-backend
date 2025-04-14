import os
from datetime import datetime
from app.services.pdf.file import PDFFile
from app.services.pdf.text_extraction import TextExtractionService
from typing import List, Dict, Any
from app.utils.db import get_async_connection


# TODO allow for persistent cloud storage
class LocalStorage:
    def __init__(self, base_path: str = "/app/uploads"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        os.chmod(self.base_path, 0o777) 
        self.text_service = TextExtractionService()
        
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
    
    async def save_file(self, pdf_file: PDFFile, paper_id: str) -> str:
        """Save file to storage and return the full path"""
        content = await pdf_file.get_content()
        storage_path = self._generate_storage_path(pdf_file)
        full_path = os.path.join(self.base_path, storage_path)
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        os.chmod(os.path.dirname(full_path), 0o777)
        
        with open(full_path, "wb") as f:
            f.write(content)
        
        # Extract metadata and full text
        metadata = await self.text_service.extract_metadata_from_pdf(pdf_file)
        full_text = await self.text_service.extract_full_text_from_pdf(pdf_file)
        
        title = metadata.get('title', '') if metadata else ''
        abstract = metadata.get('abstract', '') if metadata else ''
        authors = metadata.get('authors', '') if metadata else ''
        
        conn = await get_async_connection()
        try:
            await conn.execute(
                """
                INSERT INTO user_papers (user_id, paper_id, file_path, title, abstract, authors, full_text, upload_date)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (user_id, paper_id) DO UPDATE
                SET file_path = $3, title = $4, abstract = $5, authors = $6, full_text = $7, upload_date = $8
                """,
                pdf_file.user_id, paper_id, full_path, title, abstract, authors, full_text, pdf_file.upload_time
            )
        finally:
            await conn.close()
            
        return full_path
    
    async def delete_file(self, user_id: str, paper_id: str) -> bool:
        """Delete file from storage using user_id and paper_id."""
        try:
            conn = await get_async_connection()
            try:
                result = await conn.fetchrow(
                    "SELECT file_path FROM user_papers WHERE user_id = $1 AND paper_id = $2",
                    user_id, paper_id
                )
                
                if result and result['file_path']:
                    file_path = result['file_path']
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        await conn.execute(
                            "DELETE FROM user_papers WHERE user_id = $1 AND paper_id = $2",
                            user_id, paper_id
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
                SELECT id, user_id, paper_id, file_path, title, abstract, authors, full_text, upload_date 
                FROM user_papers 
                WHERE user_id = $1
                ORDER BY upload_date DESC
                """,
                user_id
            )
            
            for row in rows:
                if os.path.exists(row['file_path']):
                    papers.append({
                        'id': row['id'],
                        'user_id': row['user_id'],
                        'paper_id': row['paper_id'],
                        'title': row['title'] or os.path.splitext(os.path.basename(row['file_path']))[0],
                        'abstract': row['abstract'] or '',
                        'authors': row['authors'] or '',
                        'full_text': row['full_text'] or '',
                        'url': f"/uploads/{user_id}/{os.path.basename(row['file_path'])}",
                        'file_path': row['file_path'],
                        'upload_date': row['upload_date'].isoformat() if row['upload_date'] else None
                    })
        finally:
            await conn.close()
        
        return papers 