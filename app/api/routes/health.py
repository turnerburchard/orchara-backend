from fastapi import APIRouter, HTTPException
from app.utils.db import get_async_connection

router = APIRouter()

@router.get("/health")
async def health_check():
    try:
        conn = await get_async_connection()
        try:
            row = await conn.fetchrow("SELECT COUNT(*) FROM papers")
            return {"rows_loaded": row[0]}
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 