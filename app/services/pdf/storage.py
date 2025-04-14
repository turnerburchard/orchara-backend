import os
import json
from datetime import datetime
from app.services.pdf.file import PDFFile
from typing import List, Dict, Any


# TODO allow for persistent cloud storage
class LocalStorage:
    def __init__(self, base_path: str = "/app/uploads"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        os.chmod(self.base_path, 0o777) 
        
    def _get_user_path(self, user_id: str) -> str:
        """Get the base path for a user's uploads"""
        user_path = os.path.join(self.base_path, user_id)
        os.makedirs(user_path, exist_ok=True)
        os.chmod(user_path, 0o777)
        return user_path
    

    # TODO move to database
    def _get_mapping_path(self, user_id: str) -> str:
        """Get the path to the user's paper ID mapping file"""
        return os.path.join(self._get_user_path(user_id), "paper_mapping.json")
    
    def _load_mapping(self, user_id: str) -> Dict[str, str]:
        """Load the mapping between filenames and paper IDs"""
        mapping_path = self._get_mapping_path(user_id)
        print(f"Loading mapping from {mapping_path}")
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                    print(f"Loaded mapping: {mapping}")
                    return mapping
            except Exception as e:
                print(f"Error loading mapping: {e}")
                return {}
        print("Mapping file does not exist")
        return {}
    
    def _save_mapping(self, user_id: str, mapping: Dict[str, str]):
        """Save the mapping between filenames and paper IDs"""
        mapping_path = self._get_mapping_path(user_id)
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f)
    
    def _generate_storage_path(self, pdf_file: PDFFile) -> str:
        """Generate a unique storage path for the file"""
        timestamp = pdf_file.upload_time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            pdf_file.user_id,
            f"{timestamp}_{pdf_file.safe_filename}"
        )
    
    async def save_file(self, pdf_file: PDFFile, paper_id: str) -> str:
        """Save file to storage and return the full path"""
        content = await pdf_file.get_content()
        storage_path = self._generate_storage_path(pdf_file)
        full_path = os.path.join(self.base_path, storage_path)
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        os.chmod(os.path.dirname(full_path), 0o777)
        
        with open(full_path, "wb") as f:
            f.write(content)
        
        mapping = self._load_mapping(pdf_file.user_id)
        mapping[os.path.basename(full_path)] = paper_id
        self._save_mapping(pdf_file.user_id, mapping)
            
        return full_path
    
    async def delete_file(self, user_id: str, paper_id: str) -> bool:
        """Delete file from storage using user_id and paper_id."""
        try:
            user_path = self._get_user_path(user_id)
            mapping = self._load_mapping(user_id)
            
            filename = None
            for fname, pid in mapping.items():
                if pid == paper_id:
                    filename = fname
                    break
            
            if filename:
                file_path = os.path.join(user_path, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    # Update mapping
                    del mapping[filename]
                    self._save_mapping(user_id, mapping)
                    return True
            return False
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
            return False

    async def get_user_papers(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all papers stored for a user."""
        user_path = self._get_user_path(user_id)
        print(f"Looking for papers in {user_path}")
        mapping = self._load_mapping(user_id)
        papers = []
        
        if os.path.exists(user_path):
            print(f"Found user directory")
            for filename in os.listdir(user_path):
                print(f"Found file: {filename}")
                if filename.endswith('.pdf') and filename in mapping:
                    print(f"Processing PDF file: {filename} with mapping: {mapping[filename]}")
                    file_path = os.path.join(user_path, filename)
                    papers.append({
                        'paper_id': mapping[filename],
                        'title': os.path.splitext(filename)[0],
                        'url': f"/uploads/{user_id}/{filename}",
                        'abstract': ""
                    })
        else:
            print(f"User directory does not exist")
        
        print(f"Returning papers: {papers}")
        return papers 