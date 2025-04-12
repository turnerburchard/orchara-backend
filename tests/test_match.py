import pytest
from app.services.pdf.match import MatchService, MatchResult
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
def match_service():
    return MatchService()

@pytest.mark.asyncio
async def test_match_paper_no_text(match_service, mock_pdf_file):
    # Mock text extraction to return None
    match_service.text_service.extract_text_from_pdf = lambda x: None
    
    result = await match_service.match_paper(mock_pdf_file)
    
    assert isinstance(result, MatchResult)
    assert not result.found
    assert result.error == "Could not extract text from PDF"
    assert result.paper_id is None
    assert result.confidence == 0.0

@pytest.mark.asyncio
async def test_match_paper_with_text(match_service, mock_pdf_file):
    # Mock text extraction to return some text
    match_service.text_service.extract_text_from_pdf = lambda x: "test text"
    
    # Mock DOI matching to return None (no match)
    match_service.match_by_doi = lambda x: None
    
    result = await match_service.match_paper(mock_pdf_file)
    
    assert isinstance(result, MatchResult)
    assert not result.found
    assert result.error is None
    assert result.paper_id is None
    assert result.confidence == 0.0

@pytest.mark.asyncio
async def test_match_paper_with_doi_match(match_service, mock_pdf_file):
    # Mock text extraction to return some text
    match_service.text_service.extract_text_from_pdf = lambda x: "test text"
    
    # Mock DOI matching to return a paper ID
    match_service.match_by_doi = lambda x: "paper123"
    
    result = await match_service.match_paper(mock_pdf_file)
    
    assert isinstance(result, MatchResult)
    assert result.found
    assert result.error is None
    assert result.paper_id == "paper123"
    assert result.confidence == 1.0 