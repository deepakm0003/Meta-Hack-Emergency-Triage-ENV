"""
server/app.py — OpenEnv multi-mode deployment entry point.
Imports and re-exports the main FastAPI app.
"""
import sys
import os

# Ensure root is on path so main.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: F401


def start():
    """Entry point for [project.scripts] server command."""
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        reload=False,
    )


if __name__ == "__main__":
    start()
