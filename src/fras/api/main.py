"""Main FastAPI application module assembling routers, middlewares, and startup events."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fras.api.routers import (
    analytics,
    attendance,
    auth,
    health,
    sessions,
    students,
)
from fras.db.session import init_db


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
def root() -> dict[str, str]:
    """Root metadata endpoint."""
    return {
        "title": "Face Recognition Attendance System (FRAS)",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/healthz",
    }
