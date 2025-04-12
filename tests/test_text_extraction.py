import pytest
from unittest.mock import MagicMock, patch
from app.services.pdf.text_extraction import TextExtractionService, TextExtractionError
from app.services.pdf.file import PDFFile

@pytest.fixture
def mock_pdf_file():
    class MockUploadFile:
        def __init__(self):
            self.filename = "test.pdf"
            self.content = b"test content"
            
        async def read(self):
            return self.content
            
    return PDFFile(MockUploadFile(), "user0")

@pytest.fixture
def text_service():
    return TextExtractionService()

@pytest.mark.asyncio
async def test_extract_metadata_with_doi(mock_pdf_file, text_service):
    # Mock PDF content with DOI
    mock_text = """
    Title: Test Paper
    Authors: John Doe and Jane Smith
    DOI: 10.1234/test.5678
    Abstract: This is a test abstract.
    """
    
    with patch('fitz.open') as mock_fitz:
        # Mock PyMuPDF document and page
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = mock_text
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.return_value = mock_doc
        
        # Extract metadata
        metadata = await text_service.extract_metadata_from_pdf(mock_pdf_file)
        
        # Check results
        assert metadata is not None
        assert metadata["doi"] == "10.1234/test.5678"
        assert metadata["title"] == "Title: Test Paper"
        assert metadata["authors"] == "Authors: John Doe and Jane Smith"
        assert "Abstract: This is a test abstract" in metadata["abstract"]

@pytest.mark.asyncio
async def test_extract_metadata_no_doi(mock_pdf_file, text_service):
    # Mock PDF content without DOI
    mock_text = """
    Title: Test Paper
    Authors: John Doe and Jane Smith
    Abstract: This is a test abstract.
    """
    
    with patch('fitz.open') as mock_fitz:
        # Mock PyMuPDF document and page
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = mock_text
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.return_value = mock_doc
        
        # Extract metadata
        metadata = await text_service.extract_metadata_from_pdf(mock_pdf_file)
        
        # Check results
        assert metadata is not None
        assert "doi" not in metadata
        assert metadata["title"] == "Title: Test Paper"
        assert metadata["authors"] == "Authors: John Doe and Jane Smith"
        assert "Abstract: This is a test abstract" in metadata["abstract"]

@pytest.mark.asyncio
async def test_extract_metadata_empty_pdf(mock_pdf_file, text_service):
    with patch('fitz.open') as mock_fitz:
        # Mock empty PDF
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 0
        mock_fitz.return_value = mock_doc
        
        # Extract metadata
        metadata = await text_service.extract_metadata_from_pdf(mock_pdf_file)
        
        # Check results
        assert metadata is None

@pytest.mark.asyncio
async def test_extract_full_text(mock_pdf_file, text_service):
    # Mock PDF content with multiple pages
    mock_text_page1 = "Page 1 content"
    mock_text_page2 = "Page 2 content"
    
    with patch('fitz.open') as mock_fitz:
        # Mock PyMuPDF document and pages
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_page1.get_text.return_value = mock_text_page1
        mock_page2.get_text.return_value = mock_text_page2
        mock_doc.__len__.return_value = 2
        mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
        mock_fitz.return_value = mock_doc
        
        # Extract full text
        full_text = await text_service.extract_full_text_from_pdf(mock_pdf_file)
        
        # Check results
        assert full_text is not None
        assert "Page 1 content" in full_text
        assert "Page 2 content" in full_text
        
        # Check text was cached in PDFFile
        assert mock_pdf_file.get_extracted_text() == full_text

@pytest.mark.asyncio
async def test_extract_full_text_empty_pdf(mock_pdf_file, text_service):
    with patch('fitz.open') as mock_fitz:
        # Mock empty PDF
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 0
        mock_fitz.return_value = mock_doc
        
        # Extract full text
        full_text = await text_service.extract_full_text_from_pdf(mock_pdf_file)
        
        # Check results
        assert full_text is None

@pytest.mark.asyncio
async def test_extract_metadata_invalid_pdf(mock_pdf_file, text_service):
    with patch('fitz.open') as mock_fitz:
        # Mock invalid PDF
        mock_fitz.side_effect = Exception("Invalid PDF")
        
        # Extract metadata should raise error
        with pytest.raises(TextExtractionError) as exc_info:
            await text_service.extract_metadata_from_pdf(mock_pdf_file)
        
        assert "Failed to extract metadata" in str(exc_info.value)

@pytest.mark.asyncio
async def test_clean_text(text_service):
    # Test text cleaning
    dirty_text = "  This   is  a  test  \n with  extra  spaces  "
    clean_text = text_service._clean_text(dirty_text)
    
    assert clean_text == "This is a test with extra spaces" 