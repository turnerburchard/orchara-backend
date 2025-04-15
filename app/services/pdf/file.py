from fastapi import UploadFile
from typing import Optional
from datetime import datetime

class PDFFile:
    def __init__(self, upload_file: UploadFile, user_id: str):
        self.upload_file = upload_file
        self.user_id = user_id
        self.content: Optional[bytes] = None
        self.extracted_text: Optional[str] = None
        self.upload_time = datetime.utcnow()
        
    @property
    def filename(self) -> str:
        return self.upload_file.filename
        
    @property
    def safe_filename(self) -> str:
        return self.filename.replace(" ", "_")
        
    async def get_content(self) -> bytes:
        if self.content is None:
            self.content = await self.upload_file.read()
        return self.content
    
    def set_extracted_text(self, text: str):
        self.extracted_text = text
    
    def get_extracted_text(self) -> Optional[str]:
        return self.extracted_text 