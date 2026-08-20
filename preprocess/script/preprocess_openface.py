"""一体化 OpenFace rPPG 数据预处理。

对每段视频依次执行 OpenFace 68 点提取和人脸裁剪，直接写入 H5。OpenFace CSV
默认作为临时中间文件；传入 ``--landmark_dir`` 时会额外保留 CSV。该脚本是
可单独分享和运行的自包含实现，不依赖同目录下的其他预处理脚本。
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy import interpolate
import scipy.io as scio
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
BUAA_SESSIONS = {
    "lux_10.0",
    "lux_15.8",
    "lux_25.1",
    "lux_39.8",
    "lux_63.1",
    "lux_100.0",
}


def resolve_openface_binary(openface_path):
    """接受 FeatureExtraction 文件或 OpenFace 根目录。"""
    openface_path = Path(openface_path)
    if openface_path.is_file():
        if not openface_path.stat().st_mode & 0o111:
            raise PermissionError(f"OpenFace binary is not executable: {openface_path}")
        return openface_path

    if openface_path.is_dir():
        candidates = [
            openface_path / "build" / "bin" / "FeatureExtraction",
            openface_path / "FeatureExtraction",
        ]
        for candidate in candidates:
            if candidate.is_file() and (candidate.stat().st_mode & 0o111):
                return candidate
        raise FileNotFoundError(
            "Could not find executable FeatureExtraction under "
            f"{openface_path}. Tried: {', '.join(str(path) for path in candidates)}"
        )

    raise FileNotFoundError(f"OpenFace path not found: {openface_path}")


def build_video_command(openface_bin, video_path, output_dir, two_d_only=True):
    command = [str(openface_bin), "-f", str(video_path), "-out_dir", str(output_dir)]
    if two_d_only:
        command.append("-2Dfp")
    return command


def build_pure_command(openface_bin, image_dir, output_dir, two_d_only=True):
    command = [str(openface_bin), "-fdir", str(image_dir), "-out_dir", str(output_dir)]
    if two_d_only:
        command.append("-2Dfp")
    return command


def run_openface_with_error_capture(command, quiet_openface):
    run_kwargs = {"check": True, "text": True, "capture_output": True}
    if not quiet_openface:
        run_kwargs.pop("capture_output")
    return subprocess.run(command, **run_kwargs)


def read_ground_truth(dataset_name, total_num_frame, json_path):
    """读取并对齐到视频帧数，保持项目既有 H5 字段兼容。"""
    if dataset_name == "UBFC-rPPG":
        bvp, hr, video_t = np.loadtxt(json_path)
        return video_t, bvp, hr

    if dataset_name == "PURE":
        with open(json_path, "r") as infile:
            gt_data = json.load(infile)
        video_t = np.array(
            [sample["Timestamp"] for sample in gt_data["/Image"]]
        ) * 1e-9
        wave_t = np.array(
            [sample["Timestamp"] for sample in gt_data["/FullPackage"]]
        ) * 1e-9
        bvp = np.array(
            [sample["Value"]["waveform"] for sample in gt_data["/FullPackage"]]
        )
        hr = np.array(
            [sample["Value"]["pulseRate"] for sample in gt_data["/FullPackage"]]
        )
        return video_t, np.interp(video_t, wave_t, bvp), np.interp(video_t, wave_t, hr)

    if dataset_name == "BUAA":
        data_path = os.path.join(json_path, "PPGData.mat")
        pulse = scio.loadmat(data_path)["PPG"]["data"][0][0]
        pulse = np.asarray(pulse, dtype=np.float32).reshape(-1)
        pulse_time = np.linspace(0, total_num_frame, len(pulse))
        video_t = np.linspace(0, total_num_frame, total_num_frame)
        spline = interpolate.splrep(pulse_time, pulse)
        bvp = interpolate.splev(video_t, spline)
        hr = np.full_like(bvp, 60)
        return video_t, bvp, hr

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def image_sequence_paths(directory):
    paths = [
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    return sorted(paths)


def iter_rgb_frames(dataset_name, source_path):
    if dataset_name == "PURE":
        for frame_path in image_sequence_paths(source_path):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise OSError(f"Unable to read image: {frame_path}")
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise OSError(f"Unable to open video: {source_path}")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def iter_dataset_items(dataset_name, video_dir, json_dir):
    """生成 (名称, 帧源路径, GT 路径, 相对输出目录)。"""
    video_dir = Path(video_dir)
    json_dir = Path(json_dir)
    if dataset_name == "UBFC-rPPG":
        for subject_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
            name = subject_dir.name
            yield (
                "vid",
                subject_dir / "vid.avi",
                json_dir / name / "ground_truth.txt",
                Path(name),
            )
        return

    if dataset_name == "PURE":
        for session_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
            name = session_dir.name
            yield name, session_dir / name, json_dir / name / f"{name}.json", Path(name)
        return

    if dataset_name == "BUAA":
        for subject_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
            for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
                if session_dir.name not in BUAA_SESSIONS:
                    continue
                videos = sorted(session_dir.glob("*.avi"))
                if not videos:
                    continue
                video = videos[0]
                relative = Path(subject_dir.name) / session_dir.name
                gt_path = json_dir / relative
                if not (gt_path / "PPGData.mat").exists():
                    continue
                yield video.stem, video, gt_path, relative
        return

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def landmark_xy(landmarks, frame_index):
    x = landmarks.loc[frame_index, [f"x_{index}" for index in range(68)]].to_numpy(
        dtype=np.float64
    )
    y = landmarks.loc[frame_index, [f"y_{index}" for index in range(68)]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite OpenFace landmarks at frame {frame_index}")
    return x, y


def landmark_center(x, y, bbox_size=None, top_priority=0.5):
    min_y = y.min() - 0.2 * (y.max() - y.min())
    center_x = int(np.round((x.min() + x.max()) / 2.0))
    if bbox_size is None:
        center_y = int(np.round((min_y + y.max()) / 2.0))
    else:
        if not 0.0 <= top_priority <= 1.0:
            raise ValueError("top_priority must be in [0, 1]")
        vertical_extra = bbox_size - (y.max() - min_y)
        square_y = min_y - vertical_extra * top_priority
        center_y = int(np.round(square_y + bbox_size / 2.0))
    return center_x, center_y, min_y


def crop_square_edge_padded(frame, center_x, center_y, side):
    """复现旧 OpenFace 流程的边缘重复填充裁剪。"""
    half = side // 2
    y_indices = range(center_y - half, center_y - half + side)
    x_indices = range(center_x - half, center_x - half + side)
    crop = np.take(frame, y_indices, axis=0, mode="clip")
    return np.take(crop, x_indices, axis=1, mode="clip")


def save_crop_preview(
    frame, face, stored_face, center_x, center_y, bbox_size, preview_dir, frame_index
):
    preview_dir.mkdir(parents=True, exist_ok=True)
    half = bbox_size // 2
    bbox_frame = frame.copy()
    cv2.rectangle(
        bbox_frame,
        (max(0, center_x - half), max(0, center_y - half)),
        (
            min(frame.shape[1] - 1, center_x - half + bbox_size - 1),
            min(frame.shape[0] - 1, center_y - half + bbox_size - 1),
        ),
        (0, 255, 0),
        2,
    )
    for name, image in {
        "bbox": bbox_frame,
        "crop": face,
        "stored": stored_face,
    }.items():
        output_path = preview_dir / f"{frame_index:04d}_{name}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def openface_h5(
    dataset_name,
    video_path,
    landmark_path,
    json_path,
    h5_path,
    store_size=128,
    bbox_scale=1.5,
    top_priority=0.5,
    preview_frames=0,
):
    """按 OpenFace 68 点动态平滑方框裁剪，并写入兼容 H5。"""
    if bbox_scale <= 0:
        raise ValueError("bbox_scale must be positive")
    if not 0.0 <= top_priority <= 1.0:
        raise ValueError("top_priority must be in [0, 1]")
    if preview_frames < 0:
        raise ValueError("preview_frames must be non-negative")

    landmarks = pd.read_csv(landmark_path)
    total_frames = len(landmarks)
    if total_frames == 0:
        raise RuntimeError(f"OpenFace CSV is empty: {landmark_path}")
    successful = [
        index for index in range(total_frames) if bool(landmarks.loc[index, "success"])
    ]
    if not successful:
        raise RuntimeError(f"OpenFace did not detect a face in any frame: {landmark_path}")

    smooth_x, smooth_y = landmark_xy(landmarks, successful[0])
    center_x, center_y, extended_min_y = landmark_center(smooth_x, smooth_y)
    bbox_size = max(1, int(np.round(bbox_scale * (smooth_y.max() - extended_min_y))))
    center_x, center_y, _ = landmark_center(
        smooth_x,
        smooth_y,
        bbox_size=bbox_size,
        top_priority=top_priority,
    )
    store_size = bbox_size if store_size is None else store_size
    if store_size < 1:
        raise ValueError("store_size must be positive")

    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = h5_path.with_suffix("").with_name(h5_path.stem + "_preview")

    with h5py.File(h5_path, "w") as file:
        images = file.create_dataset(
            "imgs",
            shape=(total_frames, store_size, store_size, 3),
            dtype="uint8",
            chunks=(1, store_size, store_size, 3),
            compression="gzip",
            compression_opts=4,
        )

        written_frames = 0
        for frame_index, frame in enumerate(iter_rgb_frames(dataset_name, video_path)):
            if frame_index >= total_frames:
                raise RuntimeError(
                    f"Source has more frames than OpenFace CSV: {video_path}"
                )
            if bool(landmarks.loc[frame_index, "success"]):
                current_x, current_y = landmark_xy(landmarks, frame_index)
                smooth_x = 0.9 * smooth_x + 0.1 * current_x
                smooth_y = 0.9 * smooth_y + 0.1 * current_y
                center_x, center_y, _ = landmark_center(
                    smooth_x,
                    smooth_y,
                    bbox_size=bbox_size,
                    top_priority=top_priority,
                )

            face = crop_square_edge_padded(
                frame, center_x, center_y, bbox_size
            )
            stored_face = (
                face
                if store_size == bbox_size
                else cv2.resize(face, (store_size, store_size))
            )
            images[frame_index] = stored_face
            written_frames += 1
            if frame_index < preview_frames:
                save_crop_preview(
                    frame,
                    face,
                    stored_face,
                    center_x,
                    center_y,
                    bbox_size,
                    preview_dir,
                    frame_index,
                )

        if written_frames != total_frames:
            raise RuntimeError(
                f"Frame count differs from OpenFace CSV: {written_frames} != {total_frames}"
            )

        wave_t, bvp, hr = read_ground_truth(dataset_name, total_frames, json_path)
        file.create_dataset("wave_t", data=wave_t)
        file.create_dataset("bvp", data=bvp)
        file.create_dataset("hr", data=hr)
        file.attrs["preprocessing_method"] = "openface_68_landmark_dynamic_crop"
        file.attrs["initial_crop_side"] = bbox_size
        file.attrs["bbox_scale"] = bbox_scale
        file.attrs["top_priority"] = top_priority
        file.attrs["total_frames"] = total_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OpenFace landmark extraction, crop faces, and save H5."
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=["UBFC-rPPG", "PURE", "BUAA"],
    )
    parser.add_argument("--openface_bin", required=True, type=str)
    parser.add_argument("--video_dir", required=True, type=str)
    parser.add_argument("--json_dir", required=True, type=str)
    parser.add_argument("--h5_dir", required=True, type=str)
    parser.add_argument(
        "--landmark_dir",
        default=None,
        type=str,
        help="Optional directory in which OpenFace CSV files are retained.",
    )
    parser.add_argument("--store_size", default=128, type=int)
    parser.add_argument(
        "--bbox_scale",
        default=1.5,
        type=float,
        help=(
            "Scale applied to the first successful landmark height after "
            "forehead extension when fixing the crop side. Lower values crop "
            "tighter. Default: 1.5, matching the original behavior."
        ),
    )
    parser.add_argument(
        "--bbox_margin",
        default=None,
        type=float,
        help=(
            "Optional per-side margin shorthand. When set, overrides "
            "--bbox_scale with bbox_scale = 1 + 2 * bbox_margin; for example "
            "--bbox_margin 0.1 is equivalent to --bbox_scale 1.2."
        ),
    )
    parser.add_argument(
        "--top_priority",
        default=0.5,
        type=float,
        help=(
            "Fraction of vertical extra crop space assigned above the face. "
            "0.5 preserves centered crops; larger values move the crop upward."
        ),
    )
    parser.add_argument("--output_tag", default="openface", type=str)
    parser.add_argument("--preview_frames", default=2, type=int)
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help=(
            "Number of videos processed concurrently. Each worker launches its "
            "own OpenFace subprocess and writes one H5 file. Default: 1."
        ),
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--quiet_openface",
        dest="quiet_openface",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--show_openface_output",
        dest="quiet_openface",
        action="store_false",
    )
    return parser.parse_args()


def write_openface_preprocessing_info(args, openface_bin):
    h5_dir = Path(args.h5_dir)
    h5_dir.mkdir(parents=True, exist_ok=True)
    info_path = h5_dir / f"{args.output_tag}_preprocessing_info.txt"
    lines = [
        "Easy-rPPG OpenFace preprocessing information",
        "=" * 43,
        f"generated_at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"command: {shlex.join([sys.executable, *sys.argv])}",
        f"host: {platform.node()}",
        f"python: {sys.version.split()[0]}",
        "",
        "method: openface_68_landmark_dynamic_crop",
        "landmarks: OpenFace 68-point 2D (-2Dfp)",
        (
            "description: Run OpenFace FeatureExtraction and immediately crop "
            "each video into an H5 file."
        ),
        f"dataset_name: {args.dataset_name}",
        f"openface_bin: {openface_bin}",
        f"video_dir: {Path(args.video_dir).resolve()}",
        f"json_dir: {Path(args.json_dir).resolve()}",
        f"h5_dir: {h5_dir.resolve()}",
        f"landmark_dir: {Path(args.landmark_dir).resolve() if args.landmark_dir else 'temporary only'}",
        f"keep_landmarks: {bool(args.landmark_dir)}",
        f"store_size: {args.store_size}x{args.store_size}",
        f"bbox_margin: {args.bbox_margin if args.bbox_margin is not None else 'not set'}",
        f"bbox_scale: {args.bbox_scale}",
        f"top_priority: {args.top_priority}",
        f"preview_frames: {args.preview_frames}",
        f"num_workers: {args.num_workers}",
        f"skip_existing: {args.skip_existing}",
        f"output_tag: {args.output_tag}",
        f"output_pattern: <video_name>_{args.output_tag}_s{args.store_size}.h5",
        "",
        "H5 datasets: imgs, wave_t, bvp, hr",
        "",
    ]
    info_path.write_text("\n".join(lines), encoding="utf-8")
    return info_path


def build_openface_command(dataset_name, openface_bin, source_path, output_dir):
    if dataset_name == "PURE":
        return build_pure_command(openface_bin, source_path, output_dir, True)
    return build_video_command(openface_bin, source_path, output_dir, True)


def preprocess_openface_item(
    dataset_name,
    openface_bin,
    name,
    source_path,
    json_path,
    relative_dir,
    h5_dir,
    landmark_dir=None,
    store_size=128,
    bbox_scale=1.5,
    top_priority=0.5,
    output_tag="openface",
    preview_frames=2,
    quiet_openface=True,
):
    """提取单段视频的 OpenFace CSV，并立即裁剪写入 H5。"""
    source_path = Path(source_path)
    relative_dir = Path(relative_dir)
    h5_path = Path(h5_dir) / relative_dir / f"{name}_{output_tag}_s{store_size}.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"openface_{name}_") as temp:
        temp_dir = Path(temp)
        command = build_openface_command(
            dataset_name, Path(openface_bin), source_path, temp_dir
        )
        try:
            run_openface_with_error_capture(command, quiet_openface)
        except subprocess.CalledProcessError as error:
            detail = error.stderr or error.stdout or str(error)
            raise RuntimeError(f"OpenFace failed for {source_path}: {detail}") from error

        landmark_path = temp_dir / f"{name}.csv"
        if not landmark_path.exists():
            raise FileNotFoundError(f"OpenFace CSV not found: {landmark_path}")

        if landmark_dir is not None:
            retained_path = Path(landmark_dir) / relative_dir / landmark_path.name
            retained_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(landmark_path, retained_path)

        openface_h5(
            dataset_name,
            str(source_path),
            str(landmark_path),
            str(json_path),
            str(h5_path),
            store_size=store_size,
            bbox_scale=bbox_scale,
            top_priority=top_priority,
            preview_frames=preview_frames,
        )
    return h5_path


def build_preprocessing_tasks(items, args, openface_bin, landmark_dir):
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
                str(openface_bin),
                name,
                str(source_path),
                str(json_path),
                str(relative_dir),
                args.h5_dir,
                landmark_dir,
                args.store_size,
                args.bbox_scale,
                args.top_priority,
                args.output_tag,
                args.preview_frames,
                args.quiet_openface,
            )
        )
    return tasks


def preprocess_openface_task(task):
    (
        dataset_name,
        openface_bin,
        name,
        source_path,
        json_path,
        relative_dir,
        h5_dir,
        landmark_dir,
        store_size,
        bbox_scale,
        top_priority,
        output_tag,
        preview_frames,
        quiet_openface,
    ) = task
    try:
        return preprocess_openface_item(
            dataset_name,
            openface_bin,
            name,
            source_path,
            json_path,
            relative_dir,
            h5_dir,
            landmark_dir=landmark_dir,
            store_size=store_size,
            bbox_scale=bbox_scale,
            top_priority=top_priority,
            output_tag=output_tag,
            preview_frames=preview_frames,
            quiet_openface=quiet_openface,
        )
    except Exception as error:
        raise RuntimeError(f"Failed to preprocess {source_path}: {error}") from error


def run_preprocessing_tasks(tasks, num_workers):
    if num_workers < 1:
        raise ValueError("--num_workers must be at least 1")
    if num_workers == 1:
        for task in tqdm(tasks, desc="video"):
            preprocess_openface_task(task)
        return

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(preprocess_openface_task, task): task for task in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="video"):
            try:
                future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise


def main():
    args = parse_args()
    if args.store_size < 1:
        raise ValueError("--store_size must be positive")
    if args.preview_frames < 0:
        raise ValueError("--preview_frames must be non-negative")
    if args.num_workers < 1:
        raise ValueError("--num_workers must be at least 1")
    if args.bbox_margin is not None:
        if args.bbox_margin < 0:
            raise ValueError("--bbox_margin must be non-negative")
        args.bbox_scale = 1.0 + 2.0 * args.bbox_margin
    if args.bbox_scale <= 0:
        raise ValueError("--bbox_scale must be positive")
    if not 0.0 <= args.top_priority <= 1.0:
        raise ValueError("--top_priority must be in [0, 1]")
    openface_bin = resolve_openface_binary(
        Path(args.openface_bin).expanduser().resolve()
    )
    write_openface_preprocessing_info(args, openface_bin)
    landmark_dir = args.landmark_dir

    items = list(iter_dataset_items(args.dataset_name, args.video_dir, args.json_dir))
    tasks = build_preprocessing_tasks(items, args, openface_bin, landmark_dir)
    run_preprocessing_tasks(tasks, args.num_workers)


if __name__ == "__main__":
    main()
