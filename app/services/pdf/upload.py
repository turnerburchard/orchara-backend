"""
Service for handling PDF uploads and processing.
"""

from typing import Optional, Dict, Any
import uuid
import os
from datetime import datetime
from app.services.pdf.file import PDFFile
from app.services.pdf.storage import LocalStorage
from app.services.pdf.text_extraction import TextExtractionService
from app.utils.db import get_async_connection
from app.models import PDFUploadResult, Paper
import logging
from fastapi import UploadFile

logger = logging.getLogger(__name__)

class UploadService:
    def __init__(self):
        self.storage = LocalStorage()
        self.text_service = TextExtractionService()
    
    async def process_pdf(self, pdf_file: PDFFile) -> PDFUploadResult:
        """Process a PDF file and store it in the user's papers directory."""
        try:
            paper_id = str(uuid.uuid4())
            
            metadata = {}
            full_text = ""
            try:
                metadata = await self.extract_metadata(pdf_file)
                full_text = await self.extract_full_text(pdf_file)
                logger.info(f"Extracted metadata: {metadata}")
            except Exception as e:
                logger.error(f"Error extracting metadata: {str(e)}")
            
            file_path = await self.storage.save_file(pdf_file, paper_id, metadata, full_text)
            
            url = f"/uploads/{pdf_file.user_id}/{os.path.basename(file_path)}"
            
            title = metadata.get('title') if metadata.get('title') else pdf_file.safe_filename
            abstract = metadata.get('abstract', '')
            doi = metadata.get('doi', '')

            # do we really need to return the paper?
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
                    paper_id=str(uuid.uuid4()),
                    title=pdf_file.safe_filename,
                    abstract="",
                    url=""
                ),
                missing_doi=True
            )

    async def extract_metadata(self, original_pdf: PDFFile) -> Dict[str, Any]:
        """Extract metadata from a PDF file using the original PDFFile instance."""
        try:
            metadata = await self.text_service.extract_metadata_from_pdf(original_pdf)
            logger.info(f"Extracted metadata from {original_pdf.filename}: {metadata}")
            
            return metadata or {}
        except Exception as e:
            logger.error(f"Error extracting metadata from {original_pdf.filename}: {str(e)}")
            return {}

    async def extract_full_text(self, original_pdf: PDFFile) -> str:
        """Extract full text from a PDF file."""
        try:
            full_text = await self.text_service.extract_full_text_from_pdf(original_pdf)
            return full_text or ""
        except Exception as e:
            logger.error(f"Error extracting full text from {original_pdf.filename}: {str(e)}")
            return ""


