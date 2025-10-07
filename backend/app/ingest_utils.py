import os, re, math, time
from typing import List, Dict, Optional, Tuple
from pypdf import PdfReader
from docx import Document as DocxDocument

# ---- Loaders ----
def load_txt(path: str) -> str:
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def load_md(path: str) -> str:
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts)

def load_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_any(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt"]: return load_txt(path)
    if ext in [".md"]:  return load_md(path)
    if ext in [".pdf"]: return load_pdf(path)
    if ext in [".docx"]:return load_docx(path)
    return None

# ---- Simple cleaner ----
WS_RE = re.compile(r"[ \t]+")
def clean_text(s: str) -> str:
    s = s.replace("\u00A0", " ")  # nbsp
    s = WS_RE.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ---- Chunking (yaklaşık 600 token hedefi, %25 overlap) ----
# token tahmini: ~1 token ≈ 4 karakter (Türkçe/EN için kaba)
# 600 tok ~ 2400 char; 800 tok ~ 3200 char
def chunk_text(s: str, target_chars: int = 2600, overlap_ratio: float = 0.25) -> List[str]:
    s = clean_text(s)
    if not s: return []
    chunks = []
    n = len(s)
    step = int(target_chars * (1 - overlap_ratio))
    i = 0
    while i < n:
        piece = s[i:i+target_chars]
        # paragrafa yakın yerde kes (son 300 char içinde bir \n\n ara)
        cut = piece.rfind("\n\n", max(0, len(piece)-300))
        if cut != -1 and cut > 0.5*len(piece):
            piece = piece[:cut]
        chunks.append(piece.strip())
        i += step
        if i >= n: break
    return [c for c in chunks if c]

# ---- Dosya tarama ----
def scan_folder(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in [".txt",".md",".pdf",".docx"]:
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)
