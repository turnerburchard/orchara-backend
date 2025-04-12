from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import health, search, summarize, upload, user_papers

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for paper search and summarization",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(search.router, prefix=settings.API_V1_STR)
app.include_router(summarize.router, prefix=settings.API_V1_STR)
app.include_router(upload.router, prefix=settings.API_V1_STR)
app.include_router(user_papers.router, prefix=settings.API_V1_STR)

router = app 