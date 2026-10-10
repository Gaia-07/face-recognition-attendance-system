"""Attendance tracking session lifecycle router."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from fras.api.dependencies import require_role
from fras.api.schemas import SessionCreate, SessionResponse
from fras.db import repository
from fras.db.models import User
from fras.db.session import get_db

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    body: SessionCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role("faculty"))],
) -> SessionResponse:
    """Opens a new attendance tracking session."""
    session = repository.create_session(
        db=db,
        session_name=body.session_name,
        camera_id=body.camera_id,
        created_by=current_user.username,
    )
    return SessionResponse.model_validate(session)


@router.get("/active", response_model=SessionResponse | None)
def get_active_session(
    db: Annotated[Session, Depends(get_db)],
) -> SessionResponse | None:
    """Returns the currently active open attendance session, or None."""
    session = repository.get_active_session(db)
    if not session:
        return None
    return SessionResponse.model_validate(session)


@router.get("", response_model=list[SessionResponse])
def list_sessions(
    db: Annotated[Session, Depends(get_db)],
    limit: int = 50,
) -> list[SessionResponse]:
    """Lists historical attendance tracking sessions."""
    sessions = repository.list_sessions(db, limit=limit)
    return [SessionResponse.model_validate(s) for s in sessions]


@router.post("/{session_id}/end", response_model=SessionResponse)
def end_session(
    session_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role("faculty"))],
) -> SessionResponse:
    """Closes an active attendance session."""
    session = repository.get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID {session_id} not found.",
        )
    if session.ended_at is not None:
        return SessionResponse.model_validate(session)

    ended = repository.end_session(db, session_id)
    return SessionResponse.model_validate(ended)
