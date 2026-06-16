from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ModelViz", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routers.parse import router as parse_router
from .routers.compare import router as compare_router
app.include_router(parse_router)
app.include_router(compare_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
