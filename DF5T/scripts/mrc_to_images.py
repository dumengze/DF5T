"""Convert MRC volume files into PNG slice images."""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import mrcfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mrcfile")


def save_mrc_slices_as_images(mrc_path: str | os.PathLike, output_folder: str | os.PathLike) -> list[str]:
    try:
        os.makedirs(output_folder, exist_ok=True)
        image_list: list[str] = []

        print(f"Processing MRC file: {mrc_path}")
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            data = mrc.data
            if data is None or data.size == 0:
                print(f"Warning: The MRC file {mrc_path} contains empty data.")
                return []

            if data.ndim == 3:
                num_slices = data.shape[0]
            else:
                num_slices = 1
                data = data[np.newaxis, ...]

            print(f"Processing {num_slices} slices...")
            for i in range(num_slices):
                try:
                    slice_data = data[i].copy()
                    if np.all(slice_data == 0):
                        print(f"Warning: Slice {i + 1} contains all zero values.")
                        continue

                    slice_data = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    output_path = os.path.join(output_folder, f"slice_{i + 1:04d}.png")
                    cv2.imwrite(output_path, slice_data)
                    if os.path.exists(output_path):
                        image_list.append(output_path)
                        print(f"Slice {i + 1}/{num_slices} saved successfully.")
                    else:
                        print(f"Warning: Failed to save slice {i + 1}.")
                except Exception as exc:
                    print(f"Error processing slice {i + 1}: {exc}")

        print(f"Successfully saved {len(image_list)} slices.")
        return image_list
    except Exception as exc:
        print(f"Error processing MRC file {mrc_path}: {exc}")
        return []


def main() -> None:
    default_input = ROOT / "demo" / "datasets" / "mrc"
    default_output = ROOT / "demo" / "datasets"
    parser = argparse.ArgumentParser(description="Convert MRC files to image slices.")
    parser.add_argument("--input_folder", type=str, default=str(default_input), help="Folder containing MRC files.")
    parser.add_argument("--output_base_folder", type=str, default=str(default_output), help="Base folder for slice output.")
    args = parser.parse_args()

    os.makedirs(args.output_base_folder, exist_ok=True)
    for file_name in os.listdir(args.input_folder):
        if not file_name.endswith(".mrc"):
            continue
        try:
            print(f"Processing file: {file_name}\n")
            mrc_path = os.path.join(args.input_folder, file_name)
            output_folder = os.path.join(args.output_base_folder, os.path.splitext(file_name)[0])
            save_mrc_slices_as_images(mrc_path, output_folder)
            print(f"Processing completed for file: {file_name}\n")
        except Exception as exc:
            print(f"File processing failed: {file_name}: {exc}")


if __name__ == "__main__":
    main()
