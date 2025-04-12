import pytest
from fastapi import UploadFile
from app.services.pdf.file import PDFFile
from datetime import datetime

@pytest.fixture
def mock_upload_file():
    class MockUploadFile:
        def __init__(self):
            self.filename = "test.pdf"
            self.content = b"test content"
            self._read = False
            
        async def read(self):
            if self._read:
                return b""
            self._read = True
            return self.content
            
    return MockUploadFile()

@pytest.mark.asyncio
async def test_pdf_file_initialization(mock_upload_file):
    pdf_file = PDFFile(mock_upload_file, "user0")
    
    assert pdf_file.filename == "test.pdf"
    assert pdf_file.safe_filename == "test.pdf"
    assert pdf_file.user_id == "user0"
    assert pdf_file.content is None
    assert pdf_file.extracted_text is None
    assert isinstance(pdf_file.upload_time, datetime)

@pytest.mark.asyncio
async def test_pdf_file_content_caching(mock_upload_file):
    pdf_file = PDFFile(mock_upload_file, "user0")
    
    # First read
    content1 = await pdf_file.get_content()
    assert content1 == b"test content"
    
    # Second read should use cached content
    content2 = await pdf_file.get_content()
    assert content2 == b"test content"
    
    # Should only read once
    assert mock_upload_file._read

@pytest.mark.asyncio
async def test_pdf_file_text_caching(mock_upload_file):
    pdf_file = PDFFile(mock_upload_file, "user0")
    
    # Set extracted text
    pdf_file.set_extracted_text("test text")
    assert pdf_file.get_extracted_text() == "test text"
    
    # Change extracted text
    pdf_file.set_extracted_text("new text")
    assert pdf_file.get_extracted_text() == "new text" 