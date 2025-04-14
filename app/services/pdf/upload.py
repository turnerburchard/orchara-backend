"""
Service for handling PDF uploads and processing.
"""

from typing import Optional, Dict, Any
import uuid
import os
from app.services.pdf.file import PDFFile
from app.services.pdf.storage import LocalStorage
from app.services.pdf.text_extraction import TextExtractionService
from app.services.pdf.match import MatchService
from app.services.database import DatabaseService
from app.models import PDFUploadResult, Paper
import logging

logger = logging.getLogger(__name__)

class UploadService:
    def __init__(self):
        self.storage = LocalStorage()
        self.text_service = TextExtractionService()
        self.db = DatabaseService()
        self.match_service = MatchService()
    
    async def process_pdf(self, pdf_file: PDFFile) -> PDFUploadResult:
        """Process a PDF file and store it in the user's papers directory."""
        try:
            # Save the file first
            file_path = await self.storage.save_file(pdf_file, str(uuid.uuid4()))
            
            # Try to match with existing papers
            match_result = await self.match_service.match_paper(pdf_file)
            
            if match_result.found:
                # Update existing paper with new file path
                paper_id = match_result.paper_id
                await self.db.update_paper_file_path(paper_id, file_path)
                return PDFUploadResult(
                    success=True,
                    paper=Paper(
                        paper_id=paper_id,
                        title=match_result.title or '',
                        abstract=match_result.abstract or '',
                        url=f"/uploads/{pdf_file.user_id}/{os.path.basename(file_path)}"
                    ),
                    missing_doi=False
                )
            else:
                # Extract metadata for new paper
                try:
                    metadata = await self.extract_metadata(file_path)
                    logger.info(f"Extracted metadata: {metadata}")
                except Exception as e:
                    logger.error(f"Error extracting metadata: {str(e)}")
                    metadata = {}
                
                # Create new paper with extracted metadata
                new_paper = {
                    'paper_id': str(uuid.uuid4()),
                    'title': metadata.get('title', pdf_file.safe_filename),
                    'abstract': metadata.get('abstract', ''),
                    'file_path': file_path,
                    'user_id': pdf_file.user_id
                }
                await self.db.create_paper(new_paper)
                return PDFUploadResult(
                    success=True,
                    paper=Paper(
                        paper_id=new_paper['paper_id'],
                        title=new_paper['title'],
                        abstract=new_paper['abstract'],
                        url=f"/uploads/{pdf_file.user_id}/{os.path.basename(file_path)}"
                    ),
                    missing_doi=True
                )
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file.filename}: {str(e)}")
            return PDFUploadResult(
                success=False,
                error=str(e),
                paper=Paper(
                    paper_id=str(uuid.uuid4()),
                    title=pdf_file.safe_filename,
                    abstract="",
                    url=""
                ),
                missing_doi=True
            )

    async def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from a PDF file."""
        try:
            # Create a PDFFile object
            pdf_file = PDFFile(
                filename=os.path.basename(file_path),
                file_path=file_path
            )
            
            # Extract metadata using the text service
            metadata = await self.text_service.extract_metadata_from_pdf(pdf_file)
            logger.info(f"Extracted metadata from {file_path}: {metadata}")
            
            return metadata or {}
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return {}


