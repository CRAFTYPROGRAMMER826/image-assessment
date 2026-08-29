"""Vercel-compatible root entrypoint for the existing FastAPI application."""

from app.main import app

__all__ = ["app"]
