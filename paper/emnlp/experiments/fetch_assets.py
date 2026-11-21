"""Materialize a flat copy of the pinned GitHub draft, reusing verified local files."""

import argparse
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verify_snapshot import EVIDENCE, read, sha


def materialize(destination, caches):
    destination.mkdir(parents=True, exist_ok=True)
    pins = read(EVIDENCE / "asset-digests.json")

    def materialize_one(pin):
        name = pin["name"]
        if Path(name).name != name:
            raise ValueError(name)
        out = destination / name

        def matches(path):
            return (
                path.is_file()
                and path.stat().st_size == pin["size"]
                and sha(path) == pin["sha256"]
            )

        if matches(out):
            return
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as fh:
            temp = Path(fh.name)
            try:
                cached = next(
                    (base / name for base in caches if matches(base / name)), None
                )
                if cached:
                    with cached.open("rb") as source:
                        shutil.copyfileobj(source, fh)
                else:
                    subprocess.run(
                        [
                            "gh",
                            "api",
                            "-H",
                            "Accept: application/octet-stream",
                            f"repos/gulldan/udlex/releases/assets/{pin['id']}",
                        ],
                        stdout=fh,
                        check=True,
                    )
                fh.flush()
                if not matches(temp):
                    raise ValueError(f"GitHub asset digest mismatch for {name}")
                temp.replace(out)
            finally:
                temp.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(materialize_one, pins))
    print(f"Verified {len(pins)} assets in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--cache", action="append", default=[], type=Path)
    args = parser.parse_args()
    materialize(args.destination, args.cache)
