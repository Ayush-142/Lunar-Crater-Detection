"""Benchmark PyTorch-CPU vs ONNX-CPU (FP32 + dynamic INT8 + static INT8/QDQ) for a YOLO checkpoint.

Usage:
    python benchmark.py weights/ohrc_best.pt test_images/some_image.jpg

Requires the patched Ultralytics build from PR #19609 to load checkpoints
with the custom ResNet101/timm backbone (see README.md). Static quantization
needs calibration images in calib_images/ (run fetch_calib_images.py first).
"""
import argparse
import contextlib
import os
import statistics
import sys
import time
from pathlib import Path

# Avoid ultralytics silently trying to pip-install an older pinned onnx
# version on export (no Python 3.14 wheel exists for it, so it wastes
# minutes building from source before falling back to the onnx we have).
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import psutil
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO

MIN_PARAMS = 40_000_000
IMGSZ = 640
CONF = 0.25
N_WARMUP = 3
N_TIMED = 20
INTRA_OP_THREADS = 6  # physical cores on the target Ryzen 5 7530U
INTER_OP_THREADS = 1
MAX_DETECTION_DEVIATION = 0.10

ROOT = Path(__file__).resolve().parent
TEST_IMAGES_DIR = ROOT / "test_images"
CALIB_IMAGES_DIR = ROOT / "calib_images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_params(yolo_model: YOLO) -> int:
    return sum(p.numel() for p in yolo_model.model.parameters())


def cpu_name() -> str:
    import platform

    name = platform.processor() or platform.uname().processor
    return name or "unknown CPU"


def cpu_info_line() -> str:
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    freq = psutil.cpu_freq()
    freq_str = f", {freq.max:.0f} MHz max" if freq and freq.max else ""
    return f"{cpu_name()} ({physical} physical / {logical} logical cores{freq_str})"


@contextlib.contextmanager
def tuned_ort_session(intra_threads: int = INTRA_OP_THREADS, inter_threads: int = INTER_OP_THREADS):
    """Force any onnxruntime.InferenceSession created in this block to use tuned
    SessionOptions, since ultralytics' AutoBackend hardcodes its own (untuned)
    session construction and exposes no hook to pass session_options through.

    Yields a list that the patched constructor appends a verification record to
    every time a session is actually built, so callers can confirm the tuned
    settings were really used rather than assuming the patch took effect.
    """
    original_init = ort.InferenceSession.__init__
    applied = []

    def patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
        if sess_options is None:
            sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_threads
        sess_options.inter_op_num_threads = inter_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        original_init(self, path_or_bytes, sess_options, providers, provider_options, **kwargs)
        applied.append(
            {
                "intra_op_num_threads": sess_options.intra_op_num_threads,
                "inter_op_num_threads": sess_options.inter_op_num_threads,
                "graph_optimization_level": sess_options.graph_optimization_level,
                "providers": self.get_providers(),
            }
        )

    ort.InferenceSession.__init__ = patched_init
    try:
        yield applied
    finally:
        ort.InferenceSession.__init__ = original_init


def time_inferences(yolo_model: YOLO, image_path: str, n: int, imgsz: int = IMGSZ, conf: float = CONF) -> list[float]:
    times_ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        yolo_model.predict(source=image_path, imgsz=imgsz, conf=conf, device="cpu", verbose=False)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    return times_ms


def benchmark_backend(model: YOLO, image_path: str, label: str) -> tuple[float, float]:
    print(f"Warming up {label} ({N_WARMUP} runs)...")
    time_inferences(model, image_path, N_WARMUP)
    print(f"Timing {label} ({N_TIMED} runs)...")
    times = time_inferences(model, image_path, N_TIMED)
    return statistics.mean(times), statistics.median(times)


def collect_test_images() -> list[Path]:
    images = sorted(p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No test images found in {TEST_IMAGES_DIR}. Run fetch_test_images.py first.")
    return images


def collect_calib_images() -> list[Path]:
    if not CALIB_IMAGES_DIR.is_dir():
        sys.exit(f"{CALIB_IMAGES_DIR} not found. Run fetch_calib_images.py first.")
    images = sorted(p for p in CALIB_IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No calibration images found in {CALIB_IMAGES_DIR}. Run fetch_calib_images.py first.")
    return images


def letterbox(im: np.ndarray, new_shape: int = IMGSZ, color: tuple = (114, 114, 114)) -> np.ndarray:
    """Resize+pad to a square, aspect-ratio preserved — matches ultralytics' own
    preprocessing closely enough for calibration purposes (exact pixel parity
    isn't required, just a representative activation distribution)."""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def preprocess_image(path: Path, imgsz: int = IMGSZ) -> np.ndarray:
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(f"cv2 could not read {path}")
    im = letterbox(im, imgsz)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    im = im.transpose(2, 0, 1)[None]  # HWC -> BCHW
    return np.ascontiguousarray(im)


class YoloCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str, imgsz: int = IMGSZ):
        self.input_name = input_name
        self.imgsz = imgsz
        self._iter = iter(image_paths)

    def get_next(self):
        path = next(self._iter, None)
        if path is None:
            return None
        return {self.input_name: preprocess_image(path, self.imgsz)}


def build_static_qdq_int8(onnx_path: Path, calib_images: list[Path]) -> Path:
    preproc_path = onnx_path.parent / f"{onnx_path.stem}_preproc.onnx"
    try:
        quant_pre_process(str(onnx_path), str(preproc_path))
        quant_input = preproc_path
        print(f"quant_pre_process OK -> {preproc_path}")
    except Exception as e:
        print(f"quant_pre_process failed ({e}); quantizing the un-preprocessed export instead")
        quant_input = onnx_path

    model = onnx.load(str(quant_input))
    input_name = model.graph.input[0].name
    tail_node_names = [n.name for n in model.graph.node[-2:] if n.name]

    int8_path = onnx_path.parent / f"{onnx_path.stem}_int8_qdq.onnx"

    def _quantize(nodes_to_exclude):
        quantize_static(
            model_input=str(quant_input),
            model_output=str(int8_path),
            calibration_data_reader=YoloCalibrationDataReader(calib_images, input_name),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=nodes_to_exclude,
        )

    try:
        _quantize(nodes_to_exclude=None)
    except Exception as e:
        print(
            f"quantize_static failed with no exclusions ({e}); retrying once, excluding the "
            f"final output nodes from quantization: {tail_node_names}"
        )
        _quantize(nodes_to_exclude=tail_node_names or None)

    return int8_path


def run_accuracy_check(model: YOLO, image_paths: list[Path], label: str) -> tuple[int, float]:
    print(f"Accuracy check for {label} on {len(image_paths)} images (conf={CONF})...")
    total_detections = 0
    confidences: list[float] = []
    for img in image_paths:
        results = model.predict(source=str(img), imgsz=IMGSZ, conf=CONF, device="cpu", verbose=False)
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes):
                total_detections += len(boxes)
                confidences.extend(float(c) for c in boxes.conf)
    mean_conf = statistics.mean(confidences) if confidences else 0.0
    return total_detections, mean_conf


def verify_tuned_session(applied: list[dict], label: str) -> None:
    if not applied:
        raise RuntimeError(
            f"Tuned ORT session patch never triggered while benchmarking {label} — "
            "no onnxruntime.InferenceSession was constructed inside the patched context, "
            "so the timed runs may not have used the tuned settings at all."
        )
    info = applied[0]
    if info["intra_op_num_threads"] != INTRA_OP_THREADS or info["inter_op_num_threads"] != INTER_OP_THREADS:
        raise RuntimeError(f"Tuned ORT session settings did not stick for {label}: {info}")
    print(
        f"Verified tuned ORT session for {label}: intra_op_num_threads={info['intra_op_num_threads']}, "
        f"inter_op_num_threads={info['inter_op_num_threads']}, "
        f"graph_optimization_level={info['graph_optimization_level']}, "
        f"providers={info['providers']} "
        "(session_options as constructed; onnxruntime exposes no public read-back of applied "
        "thread counts from a live session, so this confirms the options actually passed in, "
        "not just that we intended to pass them)"
    )


def flag_if_deviates(name: str, det: int, pt_det: int) -> None:
    if pt_det > 0 and abs(det - pt_det) / pt_det > MAX_DETECTION_DEVIATION:
        pct = (det - pt_det) / pt_det * 100
        print(
            f"\n⚠ FLAG: {name} detection count ({det}) deviates {pct:+.1f}% from PyTorch ({pt_det}), "
            f"exceeding the {MAX_DETECTION_DEVIATION * 100:.0f}% threshold — quantization may be degrading accuracy."
        )


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def print_final_table(rows: list[dict], file_sizes: dict[str, float]) -> None:
    print(f"\nCPU: {cpu_info_line()}")
    print("\nModel file sizes:")
    for name, size in file_sizes.items():
        print(f"  {name}: {size:.1f} MB")

    pt_mean = rows[0]["mean"]
    header = f"{'Variant':<20}{'mean ms':>12}{'median ms':>12}{'speedup':>10}{'detections':>12}{'mean conf':>11}"
    print(f"\n{header}")
    for row in rows:
        speedup = pt_mean / row["mean"]
        print(
            f"{row['name']:<20}{row['mean']:>12.2f}{row['median']:>12.2f}{speedup:>9.2f}x"
            f"{row['detections']:>12}{row['mean_conf']:>11.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Path to the .pt checkpoint")
    parser.add_argument("image", type=Path, help="Path to a test image used for the timed runs")
    args = parser.parse_args()

    if not args.weights.is_file():
        sys.exit(f"Checkpoint not found: {args.weights}")
    if not args.image.is_file():
        sys.exit(f"Image not found: {args.image}")

    test_images = collect_test_images()

    print(f"Loading {args.weights} ...")
    pt_model = YOLO(str(args.weights))

    n_params = count_params(pt_model)
    if n_params <= MIN_PARAMS:
        sys.exit(
            f"Aborting: loaded model has only {n_params:,} parameters "
            f"(expected > {MIN_PARAMS:,}).\n"
            "This looks like a stock-Ultralytics load that silently substituted "
            "the wrong architecture instead of the patched ResNet101/timm backbone "
            "from PR #19609 (https://github.com/ultralytics/ultralytics/pull/19609).\n"
            "Confirm `pip show ultralytics` points at the PR branch checkout, not PyPI."
        )
    print(f"Parameter check OK: {n_params:,} parameters\n")

    rows = []

    # --- PyTorch CPU ---
    pt_mean, pt_median = benchmark_backend(pt_model, str(args.image), "PyTorch CPU")
    pt_det, pt_conf = run_accuracy_check(pt_model, test_images, "PyTorch CPU")
    rows.append({"name": "PyTorch CPU", "mean": pt_mean, "median": pt_median, "detections": pt_det, "mean_conf": pt_conf})

    # --- Export FP32 ONNX ---
    print("\nExporting to ONNX (FP32)...")
    onnx_path = Path(pt_model.export(format="onnx", imgsz=IMGSZ))
    print(f"Saved {onnx_path}")

    with tuned_ort_session() as applied:
        print(f"\nLoading {onnx_path} ...")
        fp32_model = YOLO(str(onnx_path))
        fp32_mean, fp32_median = benchmark_backend(fp32_model, str(args.image), "ONNX FP32 (tuned)")
        verify_tuned_session(applied, "ONNX FP32")
        fp32_det, fp32_conf = run_accuracy_check(fp32_model, test_images, "ONNX FP32 (tuned)")
    rows.append(
        {"name": "ONNX FP32 (tuned)", "mean": fp32_mean, "median": fp32_median, "detections": fp32_det, "mean_conf": fp32_conf}
    )

    # --- INT8 dynamic quantization ---
    int8_path = onnx_path.parent / f"{onnx_path.stem}_int8.onnx"
    print(f"\nQuantizing to INT8 (dynamic, weight-only) -> {int8_path} ...")
    quantize_dynamic(model_input=str(onnx_path), model_output=str(int8_path), weight_type=QuantType.QInt8)

    with tuned_ort_session() as applied:
        print(f"\nLoading {int8_path} ...")
        int8_model = YOLO(str(int8_path))
        int8_mean, int8_median = benchmark_backend(int8_model, str(args.image), "ONNX INT8 dynamic (tuned)")
        verify_tuned_session(applied, "ONNX INT8 dynamic")
        int8_det, int8_conf = run_accuracy_check(int8_model, test_images, "ONNX INT8 dynamic (tuned)")
    rows.append(
        {"name": "ONNX INT8 dynamic (tuned)", "mean": int8_mean, "median": int8_median, "detections": int8_det, "mean_conf": int8_conf}
    )
    flag_if_deviates("ONNX INT8 dynamic", int8_det, pt_det)

    # --- INT8 static quantization (QDQ, calibrated) ---
    calib_images = collect_calib_images()
    print(f"\nBuilding static QDQ INT8 model from {len(calib_images)} calibration images...")
    qdq_path = build_static_qdq_int8(onnx_path, calib_images)
    print(f"Saved {qdq_path}")

    with tuned_ort_session() as applied:
        print(f"\nLoading {qdq_path} ...")
        qdq_model = YOLO(str(qdq_path))
        qdq_mean, qdq_median = benchmark_backend(qdq_model, str(args.image), "ONNX INT8 QDQ (tuned)")
        verify_tuned_session(applied, "ONNX INT8 QDQ")
        qdq_det, qdq_conf = run_accuracy_check(qdq_model, test_images, "ONNX INT8 QDQ (tuned)")
    rows.append(
        {"name": "ONNX INT8 QDQ (tuned)", "mean": qdq_mean, "median": qdq_median, "detections": qdq_det, "mean_conf": qdq_conf}
    )
    flag_if_deviates("ONNX INT8 QDQ", qdq_det, pt_det)

    if qdq_mean > fp32_mean:
        print(
            f"\nNote: static QDQ INT8 ({qdq_mean:.2f} ms) came out slower than FP32 ONNX ({fp32_mean:.2f} ms) "
            "on this CPU/model — reporting as-is per a single clean attempt, not iterating further."
        )

    file_sizes = {
        args.weights.name: file_size_mb(args.weights),
        onnx_path.name: file_size_mb(onnx_path),
        int8_path.name: file_size_mb(int8_path),
        qdq_path.name: file_size_mb(qdq_path),
    }
    print_final_table(rows, file_sizes)


if __name__ == "__main__":
    main()
