import sys, pathlib, re

FORBIDDEN = re.compile(r"\bchroma(tools|db|[-_.\w]*)\b", re.IGNORECASE)

def scan_dir(p):
    p = pathlib.Path(p)
    if not p.exists():
        return True
    for path in p.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".ts", ".tsx", ".md", ".txt", ".json", ".yml", ".yaml"}:
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if FORBIDDEN.search(txt):
                print(f"FORBIDDEN: {path}")
                return False
    return True

ok = scan_dir("backend") and scan_dir("ui")
sys.exit(0 if ok else 1)
