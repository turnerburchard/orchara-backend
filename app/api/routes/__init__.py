from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api.routes import health, search, summarize, upload, user_papers
from app.utils.db import ensure_user_papers_table_exists

@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_user_papers_table_exists()
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for paper search and summarization",
    lifespan=lifespan,
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