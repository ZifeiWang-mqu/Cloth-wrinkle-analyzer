"""FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /health
    POST /api/inspect-wrinkle
    POST /api/feedback
Uploaded images are served read-only under /uploads for the frontend overlay.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.db.database import init_db
from app.routes import feedback as feedback_routes
from app.routes import inspect as inspect_routes
from app.settings import settings

logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="イラストの衣服の皺の不整合を検出する作画支援API（MVP）",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure storage dirs exist, then serve uploaded images so the frontend
    # can draw overlays on them.
    settings.ensure_dirs()
    app.mount(
        "/uploads",
        StaticFiles(directory=str(settings.upload_dir)),
        name="uploads",
    )

    @app.on_event("startup")
    def _startup() -> None:
        init_db()

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "app": settings.app_name, "version": __version__}

    @app.get("/", tags=["meta"], include_in_schema=False)
    def root() -> HTMLResponse:
        return HTMLResponse(
            content="<h1>Cloth Wrinkle Analysis API is running</h1>",
            status_code=200,
        )

    app.include_router(inspect_routes.router)
    app.include_router(feedback_routes.router)
    return app


app = create_app()
