"""Download + prepare the Kather colorectal histology dataset (Zenodo 53169).

This script downloads the 5,000 150x150 tiles ZIP from Zenodo, verifies its MD5,
extracts it, and (optionally) reorganizes the class folders into an ImageFolder-
compatible layout.

Why this exists
- The repo's training code expects a directory structure like:
    data/colorectal_histology/<class_name>/*.tif
- Zenodo provides the dataset as:
    Kather_texture_2016_image_tiles_5000.zip
      Kather_texture_2016_image_tiles_5000/
        01_TUMOR/ ...
        02_STROMA/ ...
        ...

Default behavior
- Downloads + extracts to: data/raw/
- Prepares ImageFolder structure to: data/colorectal_histology/

Dataset source
- Zenodo record: https://doi.org/10.5281/zenodo.53169
- File: Kather_texture_2016_image_tiles_5000.zip (md5 provided by Zenodo API)

License
- CC-BY-4.0 (see Zenodo record for details)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from tqdm import tqdm


ZENODO_RECORD_ID = 53169
ZENODO_API_RECORD_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
DEFAULT_ZIP_NAME = "Kather_texture_2016_image_tiles_5000.zip"


@dataclass(frozen=True)
class ZenodoFile:
    name: str
    size_bytes: int
    md5: str
    download_url: str


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "pytorch-colorectal-histopathology-classifier/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(req) as response:
        data = response.read().decode("utf-8")
    return json.loads(data)


def get_zenodo_file(record_url: str, filename: str) -> ZenodoFile:
    record = _http_get_json(record_url)

    for file_info in record.get("files", []):
        if file_info.get("key") != filename:
            continue

        checksum = file_info.get("checksum", "")
        if not checksum.startswith("md5:"):
            raise ValueError(f"Unexpected checksum format from Zenodo: {checksum}")

        download_url = file_info.get("links", {}).get("self")
        if not download_url:
            raise ValueError("Zenodo response missing download link")

        return ZenodoFile(
            name=filename,
            size_bytes=int(file_info.get("size", 0)),
            md5=checksum[len("md5:"):],
            download_url=download_url,
        )

    raise FileNotFoundError(f"File not found in Zenodo record: {filename}")


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest_path: Path, expected_size: Optional[int] = None) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "pytorch-colorectal-histopathology-classifier/1.0"},
        method="GET",
    )

    with urllib.request.urlopen(req) as response:
        total = expected_size
        if total is None:
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                total = int(content_length)

        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest_path.name}") as pbar:
            with dest_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
            for m in members:
                zf.extract(m, path=extract_dir)
                pbar.update(1)

    # Best-effort: return the top-level folder if it exists
    top_level = extract_dir / zip_path.stem
    if top_level.exists() and top_level.is_dir():
        return top_level

    # Otherwise, try to infer a single folder
    children = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]

    return extract_dir


def prepare_imagefolder(
    extracted_root: Path,
    output_dir: Path,
    class_map: Dict[str, str],
    overwrite: bool = False,
) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the directory that contains the 01_TUMOR/... folders
    # Usually: extracted_root/Kather_texture_2016_image_tiles_5000/<class_dirs>
    candidates = [extracted_root]
    nested = extracted_root / "Kather_texture_2016_image_tiles_5000"
    if nested.exists():
        candidates.insert(0, nested)

    class_parent = None
    for c in candidates:
        if any((c / k).exists() for k in class_map.keys()):
            class_parent = c
            break

    if class_parent is None:
        raise FileNotFoundError(
            "Could not locate expected class folders (e.g., 01_TUMOR) after extraction. "
            f"Looked under: {extracted_root}"
        )

    for src_name, dst_name in class_map.items():
        src_dir = class_parent / src_name
        if not src_dir.exists():
            print(f"Warning: missing class folder {src_dir}, skipping")
            continue

        dst_dir = output_dir / dst_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        images = [
            p
            for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
        ]

        for p in tqdm(images, desc=f"Copying {src_name} -> {dst_name}", unit="img"):
            shutil.copy2(p, dst_dir / p.name)


def default_class_map() -> Dict[str, str]:
    """Map Zenodo folder names to simplified class folder names."""

    # Zenodo class folders:
    # - 01_TUMOR, 02_STROMA, 03_COMPLEX, 04_LYMPHO,
    # - 05_DEBRIS, 06_MUCOSA, 07_ADIPOSE, 08_EMPTY
    return {
        "01_TUMOR": "tumor",
        "02_STROMA": "stroma",
        "03_COMPLEX": "complex",
        "04_LYMPHO": "lympho",
        "05_DEBRIS": "debris",
        "06_MUCOSA": "mucosa",
        "07_ADIPOSE": "adipose",
        "08_EMPTY": "empty",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download + prepare Kather CRC histology tiles (Zenodo 53169)")
    p.add_argument("--raw-dir", type=str, default="data/raw", help="Where to store downloaded ZIP and extraction")
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/colorectal_histology",
        help="Where to write ImageFolder-style class subfolders",
    )
    p.add_argument("--filename", type=str, default=DEFAULT_ZIP_NAME, help="Zenodo filename to download")
    p.add_argument("--skip-prepare", action="store_true", help="Only download+extract; do not reorganize classes")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing prepared output directory")
    p.add_argument("--keep-zip", action="store_true", help="Keep ZIP after extraction")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    print(f"Fetching Zenodo record metadata: {ZENODO_API_RECORD_URL}")
    zfile = get_zenodo_file(ZENODO_API_RECORD_URL, args.filename)

    zip_path = raw_dir / zfile.name
    extract_dir = raw_dir / zip_path.stem

    if zip_path.exists():
        print(f"ZIP already exists: {zip_path}")
    else:
        print(f"Downloading: {zfile.download_url}")
        download_file(zfile.download_url, zip_path, expected_size=zfile.size_bytes)

    print("Verifying MD5...")
    actual_md5 = md5sum(zip_path)
    if actual_md5.lower() != zfile.md5.lower():
        raise RuntimeError(
            f"MD5 mismatch for {zip_path.name}: expected {zfile.md5}, got {actual_md5}. "
            "Delete the ZIP and re-run."
        )

    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Extraction directory already exists: {extract_dir}")
        extracted_root = extract_dir
    else:
        print(f"Extracting to: {extract_dir}")
        extracted_root = extract_zip(zip_path, extract_dir)

    if not args.skip_prepare:
        print(f"Preparing ImageFolder dataset to: {output_dir}")
        prepare_imagefolder(
            extracted_root=extracted_root,
            output_dir=output_dir,
            class_map=default_class_map(),
            overwrite=args.overwrite,
        )

        # Quick sanity info
        class_counts: Dict[str, int] = {}
        for class_name in sorted(default_class_map().values()):
            class_dir = output_dir / class_name
            if not class_dir.exists():
                class_counts[class_name] = 0
                continue
            class_counts[class_name] = len([p for p in class_dir.iterdir() if p.is_file()])

        print("\nPrepared class counts:")
        for k, v in class_counts.items():
            print(f"- {k}: {v}")

    if not args.keep_zip:
        try:
            zip_path.unlink(missing_ok=True)
        except TypeError:
            # Python < 3.8 fallback
            if zip_path.exists():
                zip_path.unlink()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
