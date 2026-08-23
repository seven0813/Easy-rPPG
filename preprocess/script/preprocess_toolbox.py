"""Toolbox-compatible face cropping with Easy-rPPG H5 output.

This module ports the face-detection and crop scheduling behavior from
My-rPPG-Toolbox while retaining Easy-rPPG's dataset discovery, ground-truth
alignment, H5 schema, previews, and H5-list workflow.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import platform
import shlex
import sys

import cv2
import h5py
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.script.preprocess_mediapipe_masks import (  # noqa: E402
    iter_dataset_items,
    iter_rgb_frames,
    read_ground_truth,
    validate_ground_truth_lengths,
)


HAAR_CASCADE_PATH = (
    REPO_ROOT / "preprocess" / "face_detector" / "haarcascade_frontalface_default.xml"
)


def add_boolean_argument(parser, name, default, help_text):
    """Add explicit positive and negative forms for a Toolbox boolean option."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}",
        f"--no_{name}",
        dest=name,
        action="store_false",
        help=f"Disable {name}.",
    )
    parser.set_defaults(**{name: default})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Apply rPPG-Toolbox HC/Y5F face cropping and save Easy-rPPG H5 files."
        )
    )
    parser.add_argument(
        "--dataset_name", required=True, choices=["UBFC-rPPG", "PURE"]
    )
    parser.add_argument("--video_dir", required=True, type=str)
    parser.add_argument("--json_dir", required=True, type=str)
    parser.add_argument("--h5_dir", required=True, type=str)
    add_boolean_argument(
        parser,
        "do_crop_face",
        True,
        "Whether to detect and crop the face before resizing.",
    )
    parser.add_argument(
        "--backend",
        default="HC",
        choices=["HC", "Y5F"],
        help="Toolbox face detector backend: Haar Cascade or YOLO5Face.",
    )
    add_boolean_argument(
        parser,
        "use_large_face_box",
        True,
        "Whether to enlarge each detected face box around its center.",
    )
    parser.add_argument("--large_box_coef", default=1.5, type=float)
    add_boolean_argument(
        parser,
        "do_dynamic_detection",
        False,
        "Whether to detect a new face box at a fixed frame frequency.",
    )
    parser.add_argument("--dynamic_detection_frequency", default=32, type=int)
    add_boolean_argument(
        parser,
        "use_median_face_box",
        False,
        "Use one median box computed from all dynamically detected boxes.",
    )
    parser.add_argument("--store_size", default=128, type=int)
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Y5F device, for example cpu or cuda:0. Default: automatic.",
    )
    parser.add_argument("--output_tag", default="toolbox", type=str)
    parser.add_argument("--preview_frames", default=2, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args(argv)


def validate_args(args):
    if args.store_size < 1:
        raise ValueError("--store_size must be positive")
    if args.large_box_coef <= 0:
        raise ValueError("--large_box_coef must be positive")
    if args.dynamic_detection_frequency < 1:
        raise ValueError("--dynamic_detection_frequency must be at least 1")
    if args.preview_frames < 0:
        raise ValueError("--preview_frames must be non-negative")
    if args.num_workers < 1:
        raise ValueError("--num_workers must be at least 1")
    if args.use_median_face_box and not args.do_dynamic_detection:
        print(
            "[WARNING] --use_median_face_box has no effect unless "
            "--do_dynamic_detection is enabled.",
            file=sys.stderr,
            flush=True,
        )


class ToolboxFaceCropper:
    """Port of rPPG-Toolbox ``face_detection`` and ``crop_face_resize``."""

    def __init__(self, backend="HC", device=None):
        self.backend = backend
        self.device = device
        self.detector = None
        if backend == "HC":
            self.detector = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
            if self.detector.empty():
                raise RuntimeError(f"Unable to load Haar Cascade: {HAAR_CASCADE_PATH}")
        elif backend == "Y5F":
            from preprocess.face_detector.YOLO5Face import YOLO5Face

            self.detector = YOLO5Face(backend, device)
        else:
            raise ValueError(f"Unsupported face detection backend: {backend}")

    def face_detection(
        self, frame, backend=None, use_larger_box=False, larger_box_coef=1.0
    ):
        """Detect one face and return Toolbox-format ``[x, y, width, height]``."""
        backend = self.backend if backend is None else backend
        if backend == "HC":
            face_zone = self.detector.detectMultiScale(
                frame[:, :, :3].astype(np.uint8)
            )
            if len(face_zone) < 1:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[1], frame.shape[0]]
            elif len(face_zone) >= 2:
                max_width_index = np.argmax(face_zone[:, 2])
                face_box_coor = face_zone[max_width_index]
                print(
                    "Warning: More than one faces are detected. "
                    "Only cropping the biggest one."
                )
            else:
                face_box_coor = face_zone[0]
        elif "Y5F" in backend:
            result = self.detector.detect_face(frame[:, :, :3].astype(np.uint8))
            if result is not None:
                x_min, y_min, x_max, y_max = result
                width = x_max - x_min
                height = y_max - y_min
                center_x = x_min + width // 2
                center_y = y_min + height // 2
                square_size = max(width, height)
                new_x = center_x - square_size // 2
                new_y = center_y - square_size // 2
                face_box_coor = [new_x, new_y, square_size, square_size]
            else:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[1], frame.shape[0]]
        else:
            raise ValueError(f"Unsupported face detection backend: {backend}")

        if use_larger_box:
            face_box_coor[0] = max(
                0,
                face_box_coor[0]
                - (larger_box_coef - 1.0) / 2 * face_box_coor[2],
            )
            face_box_coor[1] = max(
                0,
                face_box_coor[1]
                - (larger_box_coef - 1.0) / 2 * face_box_coor[3],
            )
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        return face_box_coor

    def crop_face_resize(
        self,
        frames,
        use_face_detection,
        backend,
        use_larger_box,
        larger_box_coef,
        use_dynamic_detection,
        detection_freq,
        use_median_box,
        width,
        height,
        return_regions=False,
    ):
        """Apply Toolbox detection scheduling, cropping, and resizing."""
        if frames.ndim != 4 or frames.shape[-1] < 3 or frames.shape[0] == 0:
            raise ValueError("frames must be a non-empty [T,H,W,C] array")
        if detection_freq < 1:
            raise ValueError("detection_freq must be at least 1")

        if use_dynamic_detection:
            num_dynamic_det = int(np.ceil(frames.shape[0] / detection_freq))
        else:
            num_dynamic_det = 1

        face_region_all = []
        for index in range(num_dynamic_det):
            if use_face_detection:
                face_region_all.append(
                    self.face_detection(
                        frames[detection_freq * index],
                        backend,
                        use_larger_box,
                        larger_box_coef,
                    )
                )
            else:
                face_region_all.append([0, 0, frames.shape[2], frames.shape[1]])
        face_region_all = np.asarray(face_region_all, dtype=int)

        face_region_median = None
        if use_median_box:
            face_region_median = np.median(face_region_all, axis=0).astype(int)

        total_frames, _, _, channels = frames.shape
        resized_frames = np.zeros((total_frames, height, width, channels), dtype=np.uint8)
        assigned_regions = np.zeros((total_frames, 4), dtype=np.int32)
        for index in range(total_frames):
            frame = frames[index]
            reference_index = index // detection_freq if use_dynamic_detection else 0
            if use_median_box:
                face_region = face_region_median
            else:
                face_region = face_region_all[reference_index]
            assigned_regions[index] = face_region

            if use_face_detection:
                frame = frame[
                    max(face_region[1], 0) : min(
                        face_region[1] + face_region[3], frame.shape[0]
                    ),
                    max(face_region[0], 0) : min(
                        face_region[0] + face_region[2], frame.shape[1]
                    ),
                ]
                if frame.size == 0:
                    raise RuntimeError(
                        f"Detected face box produced an empty crop: {face_region.tolist()}"
                    )
            resized_frames[index] = cv2.resize(
                frame, (width, height), interpolation=cv2.INTER_AREA
            )

        if return_regions:
            return resized_frames, assigned_regions
        return resized_frames


def load_frames(dataset_name, source_path):
    frames = list(iter_rgb_frames(dataset_name, source_path))
    if not frames:
        raise RuntimeError(f"No frames found in source: {source_path}")
    return np.stack(frames, axis=0)


def save_previews(
    frames, stored_frames, regions, preview_dir, preview_frames, do_crop_face
):
    preview_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(preview_frames, len(frames))):
        source = frames[index]
        x, y, width, height = regions[index].tolist()
        bbox = source.copy()
        if do_crop_face:
            cv2.rectangle(
                bbox,
                (max(0, x), max(0, y)),
                (
                    min(source.shape[1] - 1, x + width - 1),
                    min(source.shape[0] - 1, y + height - 1),
                ),
                (0, 255, 0),
                2,
            )
            crop = source[
                max(y, 0) : min(y + height, source.shape[0]),
                max(x, 0) : min(x + width, source.shape[1]),
            ]
        else:
            crop = source
        for name, image in {
            "bbox": bbox,
            "crop": crop,
            "stored": stored_frames[index],
        }.items():
            cv2.imwrite(
                str(preview_dir / f"{index:04d}_{name}.jpg"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )


def toolbox_h5(
    dataset_name,
    source_path,
    json_path,
    h5_path,
    do_crop_face=True,
    backend="HC",
    use_large_face_box=True,
    large_box_coef=1.5,
    do_dynamic_detection=False,
    dynamic_detection_frequency=32,
    use_median_face_box=False,
    store_size=128,
    device=None,
    preview_frames=0,
):
    """Crop one video using Toolbox behavior and save an Easy-rPPG H5 file."""
    frames = load_frames(dataset_name, source_path)
    cropper = ToolboxFaceCropper(backend, device)
    stored_frames, regions = cropper.crop_face_resize(
        frames,
        do_crop_face,
        backend,
        use_large_face_box,
        large_box_coef,
        do_dynamic_detection,
        dynamic_detection_frequency,
        use_median_face_box,
        store_size,
        store_size,
        return_regions=True,
    )

    total_frames = len(frames)
    wave_t, bvp, hr = read_ground_truth(dataset_name, total_frames, json_path)
    validate_ground_truth_lengths(
        total_frames, wave_t, bvp, hr, source_path, json_path
    )

    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as file:
        file.create_dataset(
            "imgs",
            data=stored_frames,
            dtype="uint8",
            chunks=(1, store_size, store_size, 3),
            compression="gzip",
            compression_opts=4,
        )
        file.create_dataset("wave_t", data=wave_t)
        file.create_dataset("bvp", data=bvp)
        file.create_dataset("hr", data=hr)
        file.create_dataset(
            "crop_boxes_xywh",
            data=regions,
            dtype="int32",
            chunks=(1, 4),
            compression="gzip",
            compression_opts=4,
        )
        file.attrs["preprocessing_method"] = f"toolbox_{backend.lower()}_face_crop"
        file.attrs["do_crop_face"] = bool(do_crop_face)
        file.attrs["backend"] = backend
        file.attrs["use_large_face_box"] = bool(use_large_face_box)
        file.attrs["large_box_coef"] = large_box_coef
        file.attrs["do_dynamic_detection"] = bool(do_dynamic_detection)
        file.attrs["dynamic_detection_frequency"] = dynamic_detection_frequency
        file.attrs["use_median_face_box"] = bool(use_median_face_box)
        file.attrs["store_size"] = store_size
        file.attrs["total_frames"] = total_frames

    if preview_frames:
        preview_dir = h5_path.with_suffix("").with_name(h5_path.stem + "_preview")
        save_previews(
            frames,
            stored_frames,
            regions,
            preview_dir,
            preview_frames,
            do_crop_face,
        )
    return h5_path


def write_preprocessing_info(args):
    h5_dir = Path(args.h5_dir)
    h5_dir.mkdir(parents=True, exist_ok=True)
    info_path = h5_dir / f"{args.output_tag}_preprocessing_info.txt"
    values = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "dataset_name": args.dataset_name,
        "video_dir": str(Path(args.video_dir).resolve()),
        "json_dir": str(Path(args.json_dir).resolve()),
        "h5_dir": str(h5_dir.resolve()),
        "do_crop_face": args.do_crop_face,
        "backend": args.backend,
        "use_large_face_box": args.use_large_face_box,
        "large_box_coef": args.large_box_coef,
        "do_dynamic_detection": args.do_dynamic_detection,
        "dynamic_detection_frequency": args.dynamic_detection_frequency,
        "use_median_face_box": args.use_median_face_box,
        "store_size": args.store_size,
        "device": args.device or "automatic",
        "preview_frames": args.preview_frames,
        "num_workers": args.num_workers,
        "skip_existing": args.skip_existing,
        "output_tag": args.output_tag,
    }
    lines = ["Easy-rPPG Toolbox preprocessing information", "=" * 43]
    lines.extend(f"{key}: {value}" for key, value in values.items())
    lines.extend(
        [
            "",
            "H5 datasets: imgs, wave_t, bvp, hr, crop_boxes_xywh",
            f"output_pattern: <video_name>_{args.output_tag}_s{args.store_size}.h5",
            "",
        ]
    )
    info_path.write_text("\n".join(lines), encoding="utf-8")
    return info_path


def build_preprocessing_tasks(items, args):
    tasks = []
    for name, source_path, json_path, relative_dir in items:
        h5_path = (
            Path(args.h5_dir)
            / relative_dir
            / f"{name}_{args.output_tag}_s{args.store_size}.h5"
        )
        if args.skip_existing and h5_path.exists():
            continue
        tasks.append(
            (
                args.dataset_name,
                str(source_path),
                str(json_path),
                str(h5_path),
                args.do_crop_face,
                args.backend,
                args.use_large_face_box,
                args.large_box_coef,
                args.do_dynamic_detection,
                args.dynamic_detection_frequency,
                args.use_median_face_box,
                args.store_size,
                args.device,
                args.preview_frames,
            )
        )
    return tasks


def preprocess_toolbox_task(task):
    try:
        return toolbox_h5(*task)
    except Exception as error:
        raise RuntimeError(f"Failed to preprocess {task[1]}: {error}") from error


def run_preprocessing_tasks(tasks, num_workers):
    if num_workers == 1:
        for task in tqdm(tasks, desc="video"):
            preprocess_toolbox_task(task)
        return

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as executor:
        futures = {
            executor.submit(preprocess_toolbox_task, task): task for task in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="video"):
            try:
                future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    write_preprocessing_info(args)
    items = list(iter_dataset_items(args.dataset_name, args.video_dir, args.json_dir))
    tasks = build_preprocessing_tasks(items, args)
    run_preprocessing_tasks(tasks, args.num_workers)


if __name__ == "__main__":
    main()
