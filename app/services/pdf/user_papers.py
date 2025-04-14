from typing import List, Dict, Any
from app.services.pdf.storage import LocalStorage

class UserPapersManager:
    def __init__(self):
        self.storage = LocalStorage()

    async def get_user_papers(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all papers associated with a user.
        """
        papers = await self.storage.get_user_papers(user_id)
        return {"papers": papers}

    async def delete_paper(self, user_id: str, paper_id: str) -> bool:
        """
        Delete a paper from local storage.
        """
        return await self.storage.delete_file(user_id, paper_id) 