from fastapi import APIRouter, HTTPException
from app.services.pdf.user_papers import UserPapersManager
from typing import Dict, List, Any

router = APIRouter()
user_papers_manager = UserPapersManager()

@router.get("/user-papers")
async def get_user_papers(user_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all papers associated with a user.
    """
    try:
        return await user_papers_manager.get_user_papers(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/user-papers/{paper_id}")
async def delete_paper(paper_id: str, user_id: str) -> Dict[str, bool]:
    """
    Delete a paper.
    """
    try:
        success = await user_papers_manager.delete_paper(user_id, paper_id)
        if not success:
            raise HTTPException(status_code=404, detail="Paper not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 