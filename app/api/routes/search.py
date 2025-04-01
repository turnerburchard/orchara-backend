from fastapi import APIRouter, HTTPException
from app.api.models import SearchRequest
from app.services.search import search_api

router = APIRouter()

@router.post("/search")
async def api_search(request: SearchRequest):
    try:
        results = search_api(request.query, request.cluster_size)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 