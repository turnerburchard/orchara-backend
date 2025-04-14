from typing import Optional, Dict
import fitz
import re
from io import BytesIO
from app.services.pdf.file import PDFFile

# TODO use more robust library
# Run GROBID in docker and call API from here

class TextExtractionService:
    async def extract_metadata_from_pdf(self, pdf_file: PDFFile) -> Optional[Dict[str, str]]:
        """Extract metadata fields (title, authors, abstract) from the PDF."""
        try:
            content = await pdf_file.get_content()
            
            stream = BytesIO(content)
            
            doc = fitz.open(stream=stream, filetype="pdf")
            if len(doc) == 0:
                return None

            first_page_text = doc[0].get_text()
            title = self._extract_title(first_page_text)
            
            authors = self._extract_authors(first_page_text)
            
            abstract = self._extract_abstract(doc)
            
            metadata = {
                "title": title,
                "authors": authors,
                "abstract": abstract
            }
            
            return metadata

        except Exception as e:
            raise Exception(f"Failed to extract metadata: {str(e)}")

    def _extract_title(self, text: str) -> str:
        """Extract title from the first page of text."""
        lines = text.split('\n')
        title_lines = []
        
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            
            if not line:
                continue
                
            if re.search(r'^(abstract|introduction|references|acknowledgments|doi|http|www)', line.lower()):
                break
                
            if re.search(r'\band\b|,|\b[A-Z]\.', line) and len(line) > 5:
                if title_lines:
                    break
                elif len(line) < 100:
                    title_lines.append(line)
                    if re.search(r'[.!?]$', line):
                        break
                continue
            
            if not title_lines or (len(title_lines[-1]) < 100):
                if len(line) > 5 and len(line) < 200:
                    title_lines.append(line)
                    
                    if re.search(r'[.!?]$', line):
                        break
            else:
                if re.search(r'\band\b|,|\b[A-Z]\.', line):
                    break
                elif len(line) > 5 and len(line) < 200:
                    title_lines.append(line)
        
        title = ' '.join(title_lines)
        
        if not title:
            for line in lines:
                if line.strip():
                    return line.strip()
        
        return title

    # TODO bugs in authors
    def _extract_authors(self, text: str) -> str:
        """Extract authors from the first page of text."""
        lines = text.split('\n')
        
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if re.search(r'\band\b|,|\b[A-Z]\.', line) and len(line) > 5 and len(line) < 200:
                if not re.search(r'^(abstract|introduction|references|acknowledgments|doi|http|www)', line.lower()):
                    return line
        
        return ""

    def _extract_abstract(self, doc) -> str:
        """Extract abstract from the first few pages."""
        abstract_text = ""
        
        for i in range(min(3, len(doc))):
            page_text = doc[i].get_text()
            
            abstract_match = re.search(r"(?i)\babstract\b[:\-]?\s*(.+?)(?=\bkeywords\b|\bintroduction\b|\bindex terms\b|\bindex\b|\bterms\b|$)", page_text, re.DOTALL)
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
                
                if re.search(r"(?i)\bindex terms\b|\bkeywords\b", abstract_text):
                    abstract_text = re.sub(r"(?i)(.+?)(?=\bindex terms\b|\bkeywords\b).*$", r"\1", abstract_text, flags=re.DOTALL)
                
                break
        
        return abstract_text

    def _clean_text(self, text: str) -> str:
        """Remove extra whitespace and normalize text."""
        try:
            # Handle potential bytes input
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            
            # Handle potential None or non-string input
            if text is None:
                return ""
                
            # Replace null bytes and other problematic characters
            text = text.replace('\x00', '')
            
            # Normalize whitespace
            return " ".join(text.split()).strip()
        except Exception as e:
            print(f"Error cleaning text: {str(e)}")
            return ""

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
            
            return clean_text

        except Exception as e:
            raise Exception(f"Failed to extract full text: {str(e)}")
