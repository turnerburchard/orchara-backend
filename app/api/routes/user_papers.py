from fastapi import APIRouter, HTTPException
from app.services.pdf.storage import LocalStorage
from typing import Dict, List, Any

router = APIRouter()
storage = LocalStorage()

@router.get("/user-papers")
async def get_user_papers(user_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all papers associated with a user.
    """
    try:
        papers = await storage.get_user_papers(user_id)
        return {"papers": papers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/user-papers/{paper_id}")
async def delete_paper(paper_id: str, user_id: str) -> Dict[str, bool]:
    """
    Delete a paper.
    """
    try:
        success = await storage.delete_file(user_id, paper_id)
        if not success:
            raise HTTPException(status_code=404, detail="Paper not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 