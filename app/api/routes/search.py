from fastapi import APIRouter, HTTPException
from app.models import SearchRequest
from app.services.search import SearchService, default_config

router = APIRouter()
search_service = SearchService(default_config)

@router.post("/search")
async def api_search(request: SearchRequest):
    try:
        results = search_service.search(request.query, request.cluster_size)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 