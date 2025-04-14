from typing import Optional, Dict
import fitz
import re
from io import BytesIO
from app.services.pdf.file import PDFFile

# TODO use more robust library
# GROBID docker?

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
        
        # Look for title patterns in the first 10 lines
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip lines that look like headers, footers, or other metadata
            if re.search(r'^(abstract|introduction|references|acknowledgments|doi|http|www)', line.lower()):
                break
                
            # Check if this line looks like authors (contains typical author patterns)
            if re.search(r'\band\b|,|\b[A-Z]\.', line) and len(line) > 5:
                # If we already have title lines, this is likely the authors section
                if title_lines:
                    break
                # If this is the first line, it might be a title with "and" in it
                # Only include it if it's not too long (titles with "and" are usually shorter)
                elif len(line) < 100:
                    title_lines.append(line)
                    # If this line ends with punctuation, it's likely the end of the title
                    if re.search(r'[.!?]$', line):
                        break
                continue
            
            # If this is the first line or previous line was part of the title
            if not title_lines or (len(title_lines[-1]) < 100):
                # Check if this line looks like a title continuation
                if len(line) > 5 and len(line) < 200:
                    title_lines.append(line)
                    
                    # If this line ends with punctuation, it's likely the end of the title
                    if re.search(r'[.!?]$', line):
                        break
            else:
                # If we've found what looks like authors, stop
                if re.search(r'\band\b|,|\b[A-Z]\.', line):
                    break
                # Otherwise, this might be part of the title
                elif len(line) > 5 and len(line) < 200:
                    title_lines.append(line)
        
        # Join the title lines with spaces
        title = ' '.join(title_lines)
        
        # If no title was found, fall back to the first non-empty line
        if not title:
            for line in lines:
                if line.strip():
                    return line.strip()
        
        return title

    def _extract_authors(self, text: str) -> str:
        """Extract authors from the first page of text."""
        lines = text.split('\n')
        
        # Look for author patterns in the first few lines
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            # Author lines typically contain commas, "and", or initials
            if re.search(r'\band\b|,|\b[A-Z]\.', line) and len(line) > 5 and len(line) < 200:
                # Skip lines that look like headers or other metadata
                if not re.search(r'^(abstract|introduction|references|acknowledgments|doi|http|www)', line.lower()):
                    return line
        
        return ""

    def _extract_abstract(self, doc) -> str:
        """Extract abstract from the first few pages."""
        abstract_text = ""
        
        # Check first 3 pages for abstract
        for i in range(min(3, len(doc))):
            page_text = doc[i].get_text()
            
            # Look for abstract section
            abstract_match = re.search(r"(?i)\babstract\b[:\-]?\s*(.+?)(?=\bkeywords\b|\bintroduction\b|\bindex terms\b|\bindex\b|\bterms\b|$)", page_text, re.DOTALL)
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
                
                # Clean up the abstract by removing index terms if they were captured
                if re.search(r"(?i)\bindex terms\b|\bkeywords\b", abstract_text):
                    abstract_text = re.sub(r"(?i)(.+?)(?=\bindex terms\b|\bkeywords\b).*$", r"\1", abstract_text, flags=re.DOTALL)
                
                break
        
        return abstract_text

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
            
            return clean_text

        except Exception as e:
            raise Exception(f"Failed to extract full text: {str(e)}")
