from typing import Optional, Dict
import fitz
import re
from io import BytesIO
from app.services.pdf.file import PDFFile

class TextExtractionService:
    async def extract_metadata_from_pdf(self, pdf_file: PDFFile) -> Optional[Dict[str, str]]:
        """Extract metadata fields (DOI, title, authors, abstract) from the first page of a PDF."""
        try:
            content = await pdf_file.get_content()
            
            stream = BytesIO(content)
            
            doc = fitz.open(stream=stream, filetype="pdf")
            if len(doc) == 0:
                return None

            first_page_text = doc[0].get_text()
            clean_text = self._clean_text(first_page_text)

            metadata = self._extract_metadata_fields(clean_text)
            
            pdf_file.set_extracted_text(clean_text)
            
            return metadata

        except Exception as e:
            raise Exception(f"Failed to extract metadata: {str(e)}")

    def _extract_metadata_fields(self, text: str) -> Dict[str, str]:
        """Extract DOI, title, authors, and abstract heuristically from text."""
        result = {}

        doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.IGNORECASE)
        if doi_match:
            result["doi"] = doi_match.group(0)

        lines = text.split('\n')
        candidates = [line.strip() for line in lines if len(line.strip()) > 10]
        if candidates:
            result["title"] = candidates[0]

        for line in candidates[1:4]:
            if re.search(r"\band\b|,|\b[A-Z]\.", line):
                result["authors"] = line
                break

        abstract_match = re.search(r"(?i)\babstract\b[:\-]?\s*(.+?)(?=\bkeywords\b|\bintroduction\b|$)", text, re.DOTALL)
        if abstract_match:
            result["abstract"] = abstract_match.group(1).strip()

        return result

    def _clean_text(self, text: str) -> str:
        """Remove extra whitespace and normalize text."""
        return " ".join(text.split()).strip()

    async def extract_full_text_from_pdf(self, pdf_file: PDFFile) -> Optional[str]:
        """Extract full text from all pages of a PDF."""
        try:
            content = await pdf_file.get_content()
            
            stream = BytesIO(content)
            
            doc = fitz.open(stream=stream, filetype="pdf")
            if len(doc) == 0:
                return None

            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n\n"
            
            clean_text = self._clean_text(full_text)
            
            pdf_file.set_extracted_text(clean_text)
            
            return clean_text

        except Exception as e:
            raise Exception(f"Failed to extract full text: {str(e)}")
