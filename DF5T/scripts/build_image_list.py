"""Build a validation image list file from a folder of sample images."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from natsort import natsorted

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def list_images_in_folder(folder_path: Path, output_file: Path) -> None:
    valid_extensions = {".png", ".tif", ".jpg", ".jpeg", ".tiff"}
    valid_files = [
        filename
        for filename in folder_path.iterdir()
        if filename.is_file() and filename.suffix.lower() in valid_extensions
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for filename in natsorted(valid_files):
            handle.write(f"{filename.stem} 1\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write image-name list files for dataset manifests.")
    parser.add_argument(
        "--folder",
        type=str,
        default=str(ROOT / "demo" / "datasets" / "MitEM" / "MitEM"),
        help="Folder containing sample images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "demo" / "MitEM_val_1k.txt"),
        help="Output list file path.",
    )
    args = parser.parse_args()
    list_images_in_folder(Path(args.folder), Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
