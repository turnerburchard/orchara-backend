from typing import Optional, Dict
import fitz
import re
from io import BytesIO
from app.services.pdf.file import PDFFile

class TextExtractionError(Exception):
    """Custom exception for text extraction errors."""
    pass

class TextExtractionService:
    def __init__(self):
        """Initialize the text extraction service."""
        pass

    async def extract_metadata_from_pdf(self, pdf_file: PDFFile) -> Optional[Dict[str, str]]:
        """
        Extract metadata fields (DOI, title, authors, abstract) from the first page of a PDF.

        Args:
            pdf_file: PDFFile object containing the PDF content

        Returns:
            Dictionary with metadata fields (if found), or None
        """
        try:
            # Get content from PDFFile
            content = await pdf_file.get_content()
            
            # Create memory stream
            stream = BytesIO(content)
            
            # Open PDF document
            doc = fitz.open(stream=stream, filetype="pdf")
            if len(doc) == 0:
                return None

            # Extract text from first page
            first_page_text = doc[0].get_text()
            clean_text = self._clean_text(first_page_text)

            # Extract metadata
            metadata = self._extract_metadata_fields(clean_text)
            
            # Store extracted text in PDFFile for reuse
            pdf_file.set_extracted_text(clean_text)
            
            return metadata

        except Exception as e:
            raise TextExtractionError(f"Failed to extract metadata: {str(e)}")

    def _extract_metadata_fields(self, text: str) -> Dict[str, str]:
        """
        Extract DOI, title, authors, and abstract heuristically from text.
        """
        result = {}

        # DOI regex
        doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.IGNORECASE)
        if doi_match:
            result["doi"] = doi_match.group(0)

        # Title candidate: first non-empty, non-author-like line
        lines = text.split('\n')
        candidates = [line.strip() for line in lines if len(line.strip()) > 10]
        if candidates:
            result["title"] = candidates[0]

        # Authors: look for line after title with "and" or commas
        for line in candidates[1:4]:
            if re.search(r"\band\b|,|\b[A-Z]\.", line):
                result["authors"] = line
                break

        # Abstract: search for "abstract" followed by block of text
        abstract_match = re.search(r"(?i)\babstract\b[:\-]?\s*(.+?)(?=\bkeywords\b|\bintroduction\b|$)", text, re.DOTALL)
        if abstract_match:
            result["abstract"] = abstract_match.group(1).strip()

        return result

    def _clean_text(self, text: str) -> str:
        """Remove extra whitespace and normalize text."""
        return " ".join(text.split()).strip()

    async def extract_full_text_from_pdf(self, pdf_file: PDFFile) -> Optional[str]:
        """
        Extract full text from all pages of a PDF.

        Args:
            pdf_file: PDFFile object containing the PDF content

        Returns:
            Full extracted text as string, or None if extraction fails
        """
        try:
            # Get content from PDFFile
            content = await pdf_file.get_content()
            
            # Create memory stream
            stream = BytesIO(content)
            
            # Open PDF document
            doc = fitz.open(stream=stream, filetype="pdf")
            if len(doc) == 0:
                return None

            # Extract text from all pages
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n\n"
            
            # Clean and return text
            clean_text = self._clean_text(full_text)
            
            # Store extracted text in PDFFile for reuse
            pdf_file.set_extracted_text(clean_text)
            
            return clean_text

        except Exception as e:
            raise TextExtractionError(f"Failed to extract full text: {str(e)}")
