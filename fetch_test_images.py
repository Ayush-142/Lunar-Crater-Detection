"""Download a small Roboflow dataset and pull 5 sample images into test_images/.

The full downloaded dataset is deleted afterwards; only the 5 copied images
are kept, so this can be re-run without accumulating dataset folders.
"""
import shutil
from pathlib import Path

from roboflow import Roboflow

API_KEY = "YOUR_API_KEY"  # get your own from https://app.roboflow.com/ — never commit a real key
WORKSPACE = "depo-mtj0m"
PROJECT = "ch2-ohr"
VERSION = 1
FORMAT = "yolov11"
N_IMAGES = 5

ROOT = Path(__file__).resolve().parent
TEST_IMAGES_DIR = ROOT / "test_images"


def main():
    TEST_IMAGES_DIR.mkdir(exist_ok=True)

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download(FORMAT)
    dataset_dir = Path(dataset.location).resolve()

    valid_images_dir = dataset_dir / "valid" / "images"
    if not valid_images_dir.is_dir():
        raise FileNotFoundError(f"Expected {valid_images_dir} to exist in downloaded dataset")

    images = sorted(valid_images_dir.iterdir())[:N_IMAGES]
    if not images:
        raise FileNotFoundError(f"No images found in {valid_images_dir}")

    for img in images:
        shutil.copy2(img, TEST_IMAGES_DIR / img.name)

    shutil.rmtree(dataset_dir)

    print(f"Copied {len(images)} images to {TEST_IMAGES_DIR}:")
    for img in images:
        print(f"  - {img.name}")


if __name__ == "__main__":
    main()
