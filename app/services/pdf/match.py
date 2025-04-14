"""
This module contains the logic for matching a PDF to a paper in the database.
"""

from pydantic import BaseModel
from typing import Optional
from app.services.database import DatabaseService
from app.services.pdf.text_extraction import TextExtractionService
from app.services.pdf.file import PDFFile

class MatchResult(BaseModel):
    found: bool
    paper_id: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    confidence: float = 0.0
    error: Optional[str] = None

class MatchService:
    def __init__(self):
        self.db = DatabaseService()
        self.text_service = TextExtractionService()
    
    async def match_paper(self, pdf_file: PDFFile) -> MatchResult:
        """
        Attempts to match a PDF file to an existing paper in the database.
        Returns MatchResult with match status and paper_id if found.
        """
        try:
            metadata = await self.text_service.extract_metadata_from_pdf(pdf_file)
            if not metadata:
                return MatchResult(
                    found=False,
                    error="Could not extract metadata from PDF"
                )
            
            if "doi" in metadata:
                paper_id = await self.match_by_doi(metadata["doi"])
                if paper_id:
                    paper = await self.db.get_paper_by_id(paper_id)
                    if paper:
                        return MatchResult(
                            found=True,
                            paper_id=paper_id,
                            title=paper['title'],
                            abstract=paper['abstract'],
                            confidence=1.0  
                        )
            
            # TODO: Try title match if DOI match fails
            return MatchResult(found=False)
            
        except Exception as e:
            return MatchResult(
                found=False,
                error=f"Error matching paper: {str(e)}"
            )
    
