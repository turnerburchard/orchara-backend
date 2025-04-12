from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from app.services.pdf.upload import UploadService
from app.services.pdf.file import PDFFile
from app.models import PDFUploadResult

router = APIRouter()
upload_service = UploadService()

@router.post("/upload-pdf", response_model=PDFUploadResult)
async def api_upload(
    file: UploadFile = File(...),
    user_id: str = "user0"  # Default user for now, could be from auth later
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=422,
            detail="Only PDF files are supported"
        )
    
    try:
        pdf_file = PDFFile(file, user_id)
        result = await upload_service.process_pdf(pdf_file)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
