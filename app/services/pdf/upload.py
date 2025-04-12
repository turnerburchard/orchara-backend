"""
Service for handling PDF uploads and processing.
"""

from typing import Optional
import uuid
from app.services.pdf.file import PDFFile
from app.services.pdf.storage import LocalStorage
from app.services.pdf.text_extraction import TextExtractionService
from app.services.pdf.match import MatchService
from app.services.database import DatabaseService
from app.models import PDFUploadResult, Paper

class UploadService:
    def __init__(self):
        self.storage = LocalStorage()
        self.text_service = TextExtractionService()
        self.db = DatabaseService()
        self.match_service = MatchService()
    
    async def process_pdf(self, pdf_file: PDFFile) -> PDFUploadResult:
        """
        Process a PDF file: store it, extract text, and update database.
        
        Args:
            pdf_file: PDFFile object containing the uploaded PDF
            
        Returns:
            PDFUploadResult with processing status
        """
        try:
            # Extract full text
            full_text = await self.text_service.extract_full_text_from_pdf(pdf_file)
            if not full_text:
                paper_id = str(uuid.uuid4())
                # Store the file with the paper_id
                file_path = await self.storage.save_file(pdf_file, paper_id)
                return PDFUploadResult(
                    success=False,
                    error="Could not extract text from PDF",
                    paper=Paper(
                        paper_id=paper_id,
                        title=pdf_file.safe_filename,
                        abstract="",
                        url=file_path
                    )
                )
            
            # Extract metadata
            metadata = await self.text_service.extract_metadata_from_pdf(pdf_file)
            
            # Get DOI from metadata
            doi = metadata.get("doi", "") if metadata else ""
            
            # If no DOI is found, generate a UUID
            if not doi:
                paper_id = str(uuid.uuid4())
                title = metadata.get("title", pdf_file.safe_filename) if metadata else pdf_file.safe_filename
                abstract = metadata.get("abstract", "") if metadata else ""
                
                # Store the file with the paper_id
                file_path = await self.storage.save_file(pdf_file, paper_id)
                
                new_paper = {
                    'paper_id': paper_id,
                    'title': title,
                    'abstract': abstract,
                    'url': file_path
                }
                
                # Save to database
                await self.db.create_paper(new_paper)
                
                return PDFUploadResult(
                    success=True,
                    paper=Paper(**new_paper),
                    missing_doi=True
                )
            
            # Try to match with existing paper in database using DOI
            match_result = await self.match_service.match_paper(pdf_file)
            
            if match_result.found:
                # Get the matched paper from database
                paper = await self.db.get_paper_by_id(match_result.paper_id)
                if paper:
                    # Store the file with the paper_id
                    file_path = await self.storage.save_file(pdf_file, paper['paper_id'])
                    # Update the paper with the new file path
                    paper['url'] = file_path
                    await self.db.update_paper(paper)
                    return PDFUploadResult(
                        success=True,
                        paper=Paper(**paper),
                        missing_doi=False
                    )
            
            # Create a new paper in the database using DOI as paper_id
            title = metadata.get("title", pdf_file.safe_filename) if metadata else pdf_file.safe_filename
            abstract = metadata.get("abstract", "") if metadata else ""
            
            # Store the file with the DOI as paper_id
            file_path = await self.storage.save_file(pdf_file, doi)
            
            new_paper = {
                'paper_id': doi,  # Use DOI as paper_id
                'title': title,
                'abstract': abstract,
                'url': file_path
            }
            
            # Save to database
            await self.db.create_paper(new_paper)
            
            return PDFUploadResult(
                success=True,
                paper=Paper(**new_paper),
                missing_doi=False
            )
            
        except Exception as e:
            paper_id = str(uuid.uuid4())
            # Store the file with the paper_id even in error case
            file_path = await self.storage.save_file(pdf_file, paper_id)
            return PDFUploadResult(
                success=False,
                error=str(e),
                paper=Paper(
                    paper_id=paper_id,
                    title=pdf_file.safe_filename,
                    abstract="",
                    url=file_path
                )
            )


