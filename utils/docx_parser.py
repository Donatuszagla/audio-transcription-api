"""
utils/docx_parser.py
─────────────────────
Extract plain text from Word documents.

Supported: .docx (OOXML) via mammoth, and .doc (legacy binary) via sharepoint-to-text.

Dependency: mammoth, sharepoint-to-text (add to requirements.txt)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_docx(file_path: str | Path) -> str:
    """
    Extract raw plain text from a .doc, .docx, RTF, or plain text file.

    Parameters
    ----------
    file_path : path to the uploaded Word file (.doc or .docx)

    Returns
    -------
    Plain text string (paragraphs separated by ``\n\n``).

    Raises
    ------
    RuntimeError
        If parsing fails (corrupt / wrong format).
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    # Read the magic bytes to determine the true file type
    try:
        with path.open("rb") as fh:
            header = fh.read(8)
    except Exception as exc:
        raise RuntimeError(f"Failed to read file header: {exc}") from exc

    is_docx = header.startswith(b"PK\x03\x04")
    is_doc = header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    is_rtf = header.startswith(b"{\\rtf")

    if is_docx:
        import mammoth
        try:
            with path.open("rb") as fh:
                result = mammoth.extract_raw_text(fh)

            if result.messages:
                for msg in result.messages:
                    logger.warning("mammoth [%s]: %s", msg.type, msg.message)

            raw = result.value or ""
            if not raw.strip():
                raise RuntimeError(
                    "mammoth extracted an empty document. "
                    "The file may be image-only, password-protected, or corrupted."
                )

            text = _normalise_paragraphs(raw)
            logger.info("Extracted %d chars from %s (OOXML format)", len(text), path.name)
            return text
        except Exception as exc:
            raise RuntimeError(f"Failed to parse OOXML docx file {path.name}: {exc}") from exc

    elif is_doc:
        try:
            import sharepoint2text
            # read_file returns a generator of ExtractionInterface objects
            results = list(sharepoint2text.read_file(path))
            if not results:
                raise RuntimeError("No content extracted from .doc file.")

            raw = results[0].get_full_text()
            if not raw.strip():
                raise RuntimeError(
                    "sharepoint-to-text extracted an empty document. "
                    "The file may be password-protected or corrupted."
                )

            text = "\n\n".join(line.strip() for line in raw.split("\n") if line.strip())
            logger.info("Extracted %d chars from %s (legacy OLE2 format)", len(text), path.name)
            return text
        except Exception as exc:
            raise RuntimeError(f"Failed to parse legacy doc file {path.name}: {exc}") from exc

    elif is_rtf:
        try:
            import io
            import sharepoint2text
            with path.open("rb") as fh:
                file_bytes = fh.read()
            results = list(sharepoint2text.read_rtf(io.BytesIO(file_bytes), path=str(path)))
            if not results:
                raise RuntimeError("No content extracted from RTF file.")

            raw = results[0].get_full_text()
            if not raw.strip():
                raise RuntimeError(
                    "sharepoint-to-text extracted an empty document. "
                    "The file may be corrupted."
                )

            text = "\n\n".join(line.strip() for line in raw.split("\n") if line.strip())
            logger.info("Extracted %d chars from %s (RTF format)", len(text), path.name)
            return text
        except Exception as exc:
            raise RuntimeError(f"Failed to parse RTF file {path.name}: {exc}") from exc

    else:
        # Fallback: try plain text reading
        try:
            with path.open("r", encoding="utf-8-sig") as fh:
                text = fh.read()
            if text.strip():
                # Normalise plain text newlines to double newlines if they are single
                if "\n\n" not in text:
                    text = "\n\n".join(line.strip() for line in text.split("\n") if line.strip())
                logger.info("Extracted %d chars from %s (plain text format)", len(text), path.name)
                return text
        except Exception:
            pass

        # Final generic parser fallbacks
        try:
            import mammoth
            with path.open("rb") as fh:
                result = mammoth.extract_raw_text(fh)
            raw = result.value or ""
            if raw.strip():
                text = _normalise_paragraphs(raw)
                logger.info("Extracted %d chars from %s via mammoth fallback", len(text), path.name)
                return text
        except Exception:
            pass

        try:
            import sharepoint2text
            results = list(sharepoint2text.read_file(path))
            if results:
                raw = results[0].get_full_text()
                if raw.strip():
                    text = "\n\n".join(line.strip() for line in raw.split("\n") if line.strip())
                    logger.info("Extracted %d chars from %s via sharepoint2text fallback", len(text), path.name)
                    return text
        except Exception:
            pass

        raise RuntimeError(
            f"Unsupported or unrecognized file format for {path.name}. "
            "Please upload a valid .docx, .doc, RTF, or plain text file."
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_paragraphs(text: str) -> str:
    """Convert mammoth's single-newline output to double-newline paragraphs."""
    import re
    # Collapse 3+ newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Ensure at least double-newline between apparent paragraphs
    # (mammoth sometimes uses single \n between sentences in the same paragraph)
    # We keep single \n within paragraphs and only double between them.
    # mammoth actually already uses \n between paragraphs, so just ensure \n\n.
    lines = text.split("\n")
    # Re-join: blank lines become paragraph breaks
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.strip())
        else:
            if current:
                paragraphs.append(" ".join(current))
                current = []
    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)
