from .health import router as health_router
from .transcription import router as transcription_router
from .document import router as document_router

__all__ = ["health_router", "transcription_router", "document_router"]
