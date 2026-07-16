"""Download a batch of Roboflow images for INT8 static-quantization calibration.

Pulls from the same dataset as fetch_test_images.py but explicitly excludes
any filename already present in test_images/, so calibration data never
overlaps with the evaluation set. The full downloaded dataset is deleted
afterwards; only the copied calibration images are kept.
"""
import shutil
from pathlib import Path

from roboflow import Roboflow

API_KEY = "YOUR_API_KEY"  # get your own from https://app.roboflow.com/ — never commit a real key
WORKSPACE = "depo-mtj0m"
PROJECT = "ch2-ohr"
VERSION = 1
FORMAT = "yolov11"
N_CALIB = 20

ROOT = Path(__file__).resolve().parent
TEST_IMAGES_DIR = ROOT / "test_images"
CALIB_IMAGES_DIR = ROOT / "calib_images"


def main():
    CALIB_IMAGES_DIR.mkdir(exist_ok=True)
    excluded_names = {p.name for p in TEST_IMAGES_DIR.iterdir()} if TEST_IMAGES_DIR.is_dir() else set()

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download(FORMAT)
    dataset_dir = Path(dataset.location).resolve()

    pool = []
    for split in ("valid", "train", "test"):
        images_dir = dataset_dir / split / "images"
        if images_dir.is_dir():
            pool.extend(sorted(images_dir.iterdir()))

    candidates = [p for p in pool if p.name not in excluded_names]
    if len(candidates) < N_CALIB:
        raise RuntimeError(
            f"Only found {len(candidates)} candidate images disjoint from test_images/, need {N_CALIB}."
        )
    calib_images = candidates[:N_CALIB]

    for img in calib_images:
        shutil.copy2(img, CALIB_IMAGES_DIR / img.name)

    shutil.rmtree(dataset_dir)

    print(f"Copied {len(calib_images)} calibration images to {CALIB_IMAGES_DIR}")


if __name__ == "__main__":
    main()
