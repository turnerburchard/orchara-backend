from fastapi import APIRouter, HTTPException
from app.utils.db import get_connection

router = APIRouter()

@router.get("/health")
async def health_check():
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM papers")
            row_count = cur.fetchone()[0]
        return {"rows_loaded": row_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close() 