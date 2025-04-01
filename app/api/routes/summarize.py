from fastapi import APIRouter, HTTPException
from app.api.models import SummarizeRequest, SummarizePapersRequest
from app.services.summarize import Summarizer

router = APIRouter()

@router.post("/summarize")
async def api_summarize(request: SummarizeRequest):
    summarizer = Summarizer()
    try:
        response = summarizer.summarize(request.text)
        return {"summary": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize_papers")
async def api_summarize_papers(request: SummarizePapersRequest):
    if not request.papers:
        raise HTTPException(status_code=400, detail="Papers list cannot be empty")
    
    summarizer = Summarizer()
    try:
        result = summarizer.summarize_with_citations(request.papers)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 