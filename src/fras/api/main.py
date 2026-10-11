"""Main FastAPI application module assembling routers, middlewares, and startup events."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from fras.api.routers import (
    analytics,
    attendance,
    auth,
    health,
    sessions,
    students,
)
from fras.db.session import init_db

HTML_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle handler ensuring database initialization."""
    init_db()
    yield


app = FastAPI(
    title="Face Recognition Attendance System (FRAS) API",
    description="Production-grade biometric attendance orchestration platform with open-set verification.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local frontend and testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoint routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(students.router)
app.include_router(sessions.router)
app.include_router(attendance.router)
app.include_router(analytics.router)


@app.get("/")
def root(request: Request) -> Response:
    """Root metadata endpoint serving HTML kiosk in browser or JSON metadata for API clients."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept and HTML_TEMPLATE_PATH.exists():
        return HTMLResponse(content=HTML_TEMPLATE_PATH.read_text(encoding="utf-8"))

    return JSONResponse(
        content={
            "title": "Face Recognition Attendance System (FRAS)",
            "version": "1.0.0",
            "docs_url": "/docs",
            "health_check": "/healthz",
            "kiosk_url": "/kiosk",
        }
    )


@app.get("/kiosk", response_class=HTMLResponse)
def kiosk() -> HTMLResponse:
    """Dedicated interactive web biometric kiosk."""
    if HTML_TEMPLATE_PATH.exists():
        return HTMLResponse(content=HTML_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>FRAS Web Kiosk</h1>")
