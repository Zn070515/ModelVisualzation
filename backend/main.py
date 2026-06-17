from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import ErrorDetail, HealthStatus

_log = logging.getLogger(__name__)

app = FastAPI(title="ModelViz", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    _log.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=ErrorDetail(detail=f"Internal server error: {exc!s}").model_dump(),
    )


from .routers.parse import router as parse_router  # noqa: E402
from .routers.compare import router as compare_router  # noqa: E402

app.include_router(parse_router)
app.include_router(compare_router)


@app.get("/api/health", response_model=HealthStatus)
def health():
    return {"status": "ok"}
