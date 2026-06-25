import sys
sys.path.insert(0, '.')
from utils.chunker import chunk_text
from utils.doc_pre_cleaner import pre_clean_text

# ── Chunker tests ─────────────────────────────────────────────────────────────
# Test 1: basic split respects paragraph boundaries
text_short = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
chunks = chunk_text(text_short, max_words=5)
assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
print(f"[PASS] Chunker basic split: {len(chunks)} chunks")

# Test 2: single oversized paragraph splits on sentences
long_para = " ".join([f"This is sentence number {i}." for i in range(200)])
chunks2 = chunk_text(long_para, max_words=50)
for c in chunks2:
    wc = len(c.split())
    assert wc <= 60, f"Chunk too long: {wc} words"
print(f"[PASS] Chunker sentence-level split: {len(chunks2)} chunks, all <=~55 words")

# Test 3: empty input returns empty list
assert chunk_text("", 1000) == []
print("[PASS] Chunker empty input -> []")

# ── Pre-cleaner tests ─────────────────────────────────────────────────────────
# Test 4: filler phrase removal
dirty = "Praise the Lord, God wants us to walk in love. You understand? Amen amen amen."
cleaned = pre_clean_text(dirty)
assert "praise the lord" not in cleaned.lower(), "Filler 'praise the lord' not removed"
assert "you understand" not in cleaned.lower(), "Filler 'you understand' not removed"
assert "amen amen" not in cleaned.lower(), "Repeated amen not removed"
assert "walk in love" in cleaned.lower(), "Teaching content wrongly removed"
print(f"[PASS] Pre-cleaner filler removal OK  ->  {cleaned.strip()!r}")

# Test 5: repeated consecutive sentences collapsed
dup = "God is love. God is love. He calls us to serve."
cleaned2 = pre_clean_text(dup)
assert cleaned2.count("God is love") <= 1, "Duplicate sentence not collapsed"
print(f"[PASS] Pre-cleaner duplicate sentence OK  ->  {cleaned2.strip()!r}")

# Test 6: normalise whitespace
ws = "God  loves    you.\n\n\n\nAmen."
cleaned3 = pre_clean_text(ws)
assert "\n\n\n" not in cleaned3, "Triple blank line not normalised"
print("[PASS] Pre-cleaner whitespace normalisation OK")

# ── Document parser tests ──────────────────────────────────────────────────────
from utils.docx_parser import extract_text_from_docx
import os

docx_path = "../OFFICAL CV TEMPLATE - KCSC.docx"
if os.path.exists(docx_path):
    text = extract_text_from_docx(docx_path)
    assert len(text) > 0, "Extracted text is empty"
    assert "\n\n" in text, "Double newlines should be present in paragraphs"
    print(f"[PASS] Document parser .docx parsing OK: {len(text)} chars extracted")
else:
    print("[SKIP] Document parser .docx test (sample file not found at expected path)")

print()
print("All unit tests passed!")
