from __future__ import annotations

import hashlib
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Archive member escapes extraction directory: {member_name}") from exc
    return target


def safe_extract_archive(archive: str | Path, destination: str | Path) -> Path:
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                _safe_target(destination, member.filename)
            handle.extractall(destination)
        return destination

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as handle:
            members = handle.getmembers()
            for member in members:
                _safe_target(destination, member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive links are not allowed: {member.name}")
            handle.extractall(destination, members=members)
        return destination

    raise ValueError(f"Unsupported archive format: {archive}")


def download_resumable(
    url: str,
    destination: str | Path,
    cookie_header: str | None = None,
    retries: int = 3,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "HearMyHands-DatasetBootstrap/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        if cookie_header:
            headers["Cookie"] = cookie_header
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                supports_resume = response.status == 206
                mode = "ab" if offset and supports_resume else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(chunk_size):
                        handle.write(chunk)
            partial.replace(destination)
            return destination
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Download failed")


def archive_candidates(directory: str | Path, name_fragment: str = "mediapi") -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    suffixes = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and name_fragment.lower() in path.name.lower()
            and any(path.name.lower().endswith(suffix) for suffix in suffixes)
            and not path.name.lower().endswith((".part", ".crdownload"))
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
