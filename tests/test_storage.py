import pytest
import os
import shutil
from app.services.pdf.storage import LocalStorage
from app.services.pdf.file import PDFFile

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory"""
    storage_dir = tmp_path / "uploads"
    storage_dir.mkdir()
    return str(storage_dir)

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
def storage_service(temp_storage_dir):
    """Create a storage service with temporary directory"""
    return LocalStorage(base_path=temp_storage_dir)

@pytest.mark.asyncio
async def test_save_file(storage_service, mock_pdf_file, temp_storage_dir):
    # Save file
    saved_path = await storage_service.save_file(mock_pdf_file)
    
    # Check file exists
    assert os.path.exists(saved_path)
    
    # Check content
    with open(saved_path, "rb") as f:
        content = f.read()
    assert content == b"test content"
    
    # Check path structure
    assert mock_pdf_file.user_id in saved_path
    assert mock_pdf_file.safe_filename in saved_path

@pytest.mark.asyncio
async def test_delete_file(storage_service, mock_pdf_file, temp_storage_dir):
    # Save file first
    saved_path = await storage_service.save_file(mock_pdf_file)
    assert os.path.exists(saved_path)
    
    # Delete file
    success = await storage_service.delete_file(mock_pdf_file)
    assert success
    assert not os.path.exists(saved_path)

@pytest.mark.asyncio
async def test_user_directory_creation(storage_service, temp_storage_dir):
    # Create path for new user
    user_path = os.path.join(temp_storage_dir, "newuser")
    assert not os.path.exists(user_path)
    
    # Save file for new user
    mock_file = PDFFile(MockUploadFile(), "newuser")
    await storage_service.save_file(mock_file)
    
    # Check directory created
    assert os.path.exists(user_path)
    assert os.path.isdir(user_path) 