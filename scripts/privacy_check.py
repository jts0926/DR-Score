from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".ipynb", ".csv"}
FORBIDDEN_FILE_SUFFIXES = {
    ".dcm", ".dicom", ".nii", ".mha", ".mhd", ".sav", ".ckpt", ".pth",
    ".jpg", ".jpeg", ".tif", ".tiff",
}
PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE),
    "Unix home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "private URL credential": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
}


def main() -> None:
    findings = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts) or not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_FILE_SUFFIXES:
            findings.append(f"forbidden file type: {path.relative_to(ROOT)}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {path.relative_to(ROOT)}")
    if findings:
        raise SystemExit("Privacy check failed:\n- " + "\n- ".join(sorted(set(findings))))
    print("Privacy check passed: no private paths, credentials, emails, or raw data files found.")


if __name__ == "__main__":
    main()
