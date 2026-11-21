"""Pack the anonymous supplement and refuse identifying strings."""

import gzip
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent.parent / "output/udlex-emnlp-supplement.zip"
# The repository slug is the only identifying string in the shipped files.
SLUG = re.compile("gull" + "dan/udlex")
IDENTITY = re.compile(
    r"gull" + r"dan|/Users/|/mnt/|/home/|qw" + r"erty|noreply", re.IGNORECASE
)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE)
PROSE = {".py", ".md", ".rs", ".tex"}


def main():
    files = [ROOT / "SUPPLEMENT.md"]
    for folder in ("experiments", "evidence"):
        files += sorted(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            raw = path.read_bytes()
            packed = path.suffix == ".gz"
            text = SLUG.sub(
                "anonymous/repository",
                (gzip.decompress(raw) if packed else raw).decode(),
            )
            if IDENTITY.search(text) or (path.suffix in PROSE and EMAIL.search(text)):
                raise ValueError(f"Identifying string in {path.relative_to(ROOT)}")
            data = gzip.compress(text.encode(), mtime=0) if packed else text.encode()
            info = zipfile.ZipInfo(
                str(path.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    named = sum(
        "udlex" in p.read_bytes().decode(errors="ignore")
        for p in files
        if p.suffix in PROSE
    )
    print(
        len(files),
        "files,",
        OUT.stat().st_size,
        "bytes;",
        named,
        "source files name the package",
    )


if __name__ == "__main__":
    main()
