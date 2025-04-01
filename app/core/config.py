from typing import List

class Settings:
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Orchara API"
    
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",  # Vite dev server
        "https://orchara.com",
        "https://www.orchara.com",
        "https://api.orchara.com"
    ]

settings = Settings() 