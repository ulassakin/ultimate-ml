#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.content import ContentError, load_library

try:
    library = load_library()
except ContentError as exc:
    print(f"Content validation failed:\n{exc}")
    raise SystemExit(1)
print(f"Content is valid: {len(library.topics)} topics, {len(library.questions)} questions.")

