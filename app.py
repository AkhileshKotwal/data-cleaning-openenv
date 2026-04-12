# This file is a placeholder for HF Spaces compatibility.
# The actual FastAPI app is in server/app.py
# The Dockerfile runs: uvicorn server.app:app

from server.app import app

__all__ = ["app"]
