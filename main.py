"""Backend entry script for local development."""
import sys
from pathlib import Path

# Add parent directory to Python path so 'backend' module can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
