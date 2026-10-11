"""Vercel Serverless Function entrypoint for Face Recognition Attendance System (FRAS)."""

import os
import sys
from pathlib import Path

# Set VERCEL environment flag
os.environ["VERCEL"] = "1"

# Add project root and src/ to sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fras.api.main import app
from fras.db.session import init_db

# Initialize database tables on serverless cold start
try:
    init_db()
except Exception:
    pass

# Export ASGI app instance for Vercel Python runtime
handler = app
