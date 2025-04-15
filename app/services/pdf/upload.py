"""
Service for handling PDF uploads and processing.
"""

from typing import Dict, Any, Tuple
import uuid
import os
from app.services.pdf.file import PDFFile
from app.services.pdf.storage import LocalStorage
from app.services.pdf.text_extraction import TextExtractionService
from app.models import PDFUploadResult, Paper
import logging

logger = logging.getLogger(__name__)

class UploadService:
    def __init__(self):
        self.storage = LocalStorage()
        self.text_service = TextExtractionService()
    
    async def process_pdf(self, pdf_file: PDFFile) -> PDFUploadResult:
        """Process a PDF file and store it in the user's papers directory."""
        paper_id = str(uuid.uuid4())
        
        # Extract metadata and text with error handling
        metadata, full_text = await self._extract_content(pdf_file)
        
        try:
            # Save the file
            file_path = await self.storage.save_file(pdf_file, paper_id, metadata, full_text)
            url = f"/uploads/{pdf_file.user_id}/{os.path.basename(file_path)}"
            
            # Prepare paper data
            title = metadata.get('title') or pdf_file.safe_filename
            abstract = metadata.get('abstract', '')
            doi = metadata.get('doi', '')
            
            return PDFUploadResult(
                success=True,
                paper=Paper(
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    url=url
                ),
                missing_doi=not bool(doi)
            )
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file.filename}: {str(e)}")
            return PDFUploadResult(
                success=False,
                error=str(e),
                paper=Paper(
                    paper_id=paper_id,
                    title=pdf_file.safe_filename,
                    abstract="",
                    url=""
                ),
                missing_doi=True
            )
    
    async def _extract_content(self, pdf_file: PDFFile) -> Tuple[Dict[str, Any], str]:
        """Extract metadata and full text from a PDF file with error handling."""
        metadata = {}
        full_text = ""
        
        try:
            metadata = await self.text_service.extract_metadata_from_pdf(pdf_file)
            logger.info(f"Extracted metadata from {pdf_file.filename}: {metadata}")
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_file.filename}: {str(e)}")
        
        try:
            full_text = await self.text_service.extract_full_text_from_pdf(pdf_file)
        except Exception as e:
            logger.error(f"Error extracting full text from {pdf_file.filename}: {str(e)}")
        
        return metadata or {}, full_text or ""


