"""使用 MediaPipe 468 点生成裁剪视频、面部掩码和可靠皮肤先验。

该脚本独立于其他预处理脚本。每段视频执行两遍读取：
1. 检测所有帧的 468 个面部关键点，插值补齐检测丢帧，并计算全局共享方框；
2. 使用固定方框裁剪所有帧，同时生成与裁剪视频严格对齐的三类空间信息：
   face_masks 表示完整面部，skin_masks 表示额头和双颊可靠区域，
   skin_priors 表示额头和双颊的连续软权重。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shlex
import sys

import cv2
import h5py
import numpy as np
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

# MediaPipe Face Mesh 的完整面部轮廓。按轮廓顺序连接后可直接填充为 face mask。
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
OUTER_LIPS = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
)
LEFT_CHEEK = (117, 205)
RIGHT_CHEEK = (346, 425)
FOREHEAD_TOP = 10
CHIN = 152
FACE_LEFT = 234
FACE_RIGHT = 454


def write_preprocessing_info(
    h5_dir,
    output_tag,
    dataset_name,
    video_dir,
    json_dir,
    store_size,
    bbox_margin,
    method,
    method_description,
    extra_parameters=None,
):
    """在 H5 主目录写入本次预处理配置，便于复现实验。"""
    h5_dir = Path(h5_dir)
    h5_dir.mkdir(parents=True, exist_ok=True)
    info_path = h5_dir / f"{output_tag}_preprocessing_info.txt"

    try:
        mediapipe_version = version("mediapipe")
    except PackageNotFoundError:
        mediapipe_version = "not installed"

    lines = [
        "Easy-rPPG preprocessing information",
        "=" * 35,
        f"generated_at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"command: {shlex.join([sys.executable, *sys.argv])}",
        f"host: {platform.node()}",
        f"python: {sys.version.split()[0]}",
        f"mediapipe: {mediapipe_version}",
        "",
        f"method: {method}",
        f"description: {method_description}",
        f"dataset_name: {dataset_name}",
        f"video_dir: {Path(video_dir).resolve()}",
        f"json_dir: {Path(json_dir).resolve()}",
        f"h5_dir: {h5_dir.resolve()}",
        f"store_size: {store_size}x{store_size}",
        f"bbox_margin: {bbox_margin}",
        f"output_tag: {output_tag}",
        f"output_pattern: <video_name>_{output_tag}_s{store_size}.h5",
    ]
    for name, value in (extra_parameters or {}).items():
        lines.append(f"{name}: {value}")
    lines.extend(
        [
            "",
            "H5 datasets: imgs, face_masks, skin_masks, skin_priors, wave_t, bvp, hr",
            "Note: crop mode is recorded in each H5 file.",
            "Note: mask coordinates are strictly aligned with imgs.",
            "",
        ]
    )
    info_path.write_text("\n".join(lines), encoding="utf-8")
    return info_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Crop rPPG videos and save MediaPipe-derived masks in H5 files."
        )
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=["UBFC-rPPG", "PURE", "BUAA"],
    )
    parser.add_argument("--video_dir", required=True, type=str)
    parser.add_argument("--json_dir", required=True, type=str)
    parser.add_argument("--h5_dir", required=True, type=str)
    parser.add_argument(
        "--store_size",
        default=192,
        type=int,
        help="Output face crop size. The paper uses 192.",
    )
    parser.add_argument(
        "--bbox_margin",
        default=0,
        type=float,
        help=(
            "Margin on each side of the landmark rectangle. Applies to both "
            "global and frame crop modes. Default: 0."
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
    parser.add_argument(
        "--crop_mode",
        default="frame",
        choices=["global", "frame"],
        help=(
            "global: use one crop square for the whole video; frame: compute a "
            "tight crop square for every frame from interpolated landmarks."
        ),
    )
    parser.add_argument(
        "--output_tag",
        default="masks",
        type=str,
        help="Tag included in H5 names to avoid overwriting other outputs.",
    )
    parser.add_argument(
        "--preview_frames",
        default=2,
        type=int,
        help="Save source/crop/stored previews for the first N frames.",
    )
    parser.add_argument(
        "--min_detection_confidence",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help=(
            "Number of videos processed in parallel. Each worker owns an "
            "independent MediaPipe FaceMesh instance. Default: 1."
        ),
    )
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def read_ground_truth(dataset_name, total_num_frame, json_path):
    """读取并对齐到视频帧数，保持与现有 OpenFace H5 字段兼容。"""
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


def validate_ground_truth_lengths(
    total_frames,
    wave_t,
    bvp,
    hr,
    source_path,
    json_path,
):
    """Report video/ground-truth mismatches without stopping preprocessing."""
    lengths = {
        "imgs": int(total_frames),
        "wave_t": len(np.asarray(wave_t).reshape(-1)),
        "bvp": len(np.asarray(bvp).reshape(-1)),
        "hr": len(np.asarray(hr).reshape(-1)),
    }
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        print(
            "[WARNING] Video and ground-truth lengths do not match; "
            "continuing preprocessing and preserving each original length: "
            f"{detail}. Video: {source_path}. Ground truth: {json_path}.",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


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
    """每次调用均重新读取一遍 RGB 帧，供检测和 H5 写入两遍使用。"""
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


def landmarks_to_pixel_points(landmarks, frame_height, frame_width):
    """将 468 个归一化 landmarks 转为原始视频中的浮点像素坐标。"""
    points = np.array([(point.x, point.y) for point in landmarks[:468]])
    if points.shape != (468, 2) or not np.isfinite(points).all():
        raise ValueError("MediaPipe must return 468 finite xy landmarks")

    points[:, 0] = np.clip(points[:, 0] * frame_width, 0, frame_width - 1)
    points[:, 1] = np.clip(points[:, 1] * frame_height, 0, frame_height - 1)
    return points.astype(np.float32)


def pixel_points_to_rectangle(points, frame_height, frame_width):
    """将像素 landmarks 转为裁剪到图像范围的 [x1,y1,x2,y2) 矩形。"""
    x = points[:, 0]
    y = points[:, 1]
    x1 = int(np.floor(x.min()))
    y1 = int(np.floor(y.min()))
    x2 = int(np.ceil(x.max())) + 1
    y2 = int(np.ceil(y.max())) + 1
    return x1, y1, min(x2, frame_width), min(y2, frame_height)


def landmarks_to_rectangle(landmarks, frame_height, frame_width):
    """兼容入口：直接由 MediaPipe landmarks 计算矩形。"""
    points = landmarks_to_pixel_points(landmarks, frame_height, frame_width)
    return pixel_points_to_rectangle(points, frame_height, frame_width)


def interpolate_landmark_sequence(landmark_sequence):
    """沿时间轴线性插值检测丢帧，并用最近有效帧填充序列两端。"""
    valid_indices = [
        index for index, landmarks in enumerate(landmark_sequence)
        if landmarks is not None
    ]
    if not valid_indices:
        raise RuntimeError("MediaPipe did not detect a face in any frame")

    total_frames = len(landmark_sequence)
    flattened = np.stack(
        [landmark_sequence[index].reshape(-1) for index in valid_indices],
        axis=0,
    )
    frame_indices = np.arange(total_frames)
    interpolated = np.empty((total_frames, flattened.shape[1]), dtype=np.float32)
    for coordinate in range(flattened.shape[1]):
        interpolated[:, coordinate] = np.interp(
            frame_indices,
            valid_indices,
            flattened[:, coordinate],
        )
    return interpolated.reshape(total_frames, 468, 2)


def transform_landmarks_to_stored(points, square, store_size):
    """将原视频像素坐标映射到固定裁剪并 resize 后的 H5 图像坐标。"""
    x, y, side = square
    transformed = points.copy().astype(np.float32)
    transformed[:, 0] = (transformed[:, 0] - x) * store_size / side
    transformed[:, 1] = (transformed[:, 1] - y) * store_size / side
    return transformed


def _polygon_mask(points, indices, size):
    mask = np.zeros((size, size), dtype=np.uint8)
    polygon = np.rint(points[list(indices)]).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def _local_polygon_mask(center, basis_x, basis_y, offsets, size):
    """将脸部局部坐标中的多边形转换到图像坐标并填充。"""
    polygon = np.stack(
        [
            center + basis_x * offset_x + basis_y * offset_y
            for offset_x, offset_y in offsets
        ]
    )
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return mask


def _oriented_gaussian(grid_x, grid_y, center, basis_x, basis_y, sigma_x, sigma_y):
    offset_x = grid_x - center[0]
    offset_y = grid_y - center[1]
    local_x = offset_x * basis_x[0] + offset_y * basis_x[1]
    local_y = offset_x * basis_y[0] + offset_y * basis_y[1]
    distance = (local_x / max(sigma_x, 1e-6)) ** 2
    distance += (local_y / max(sigma_y, 1e-6)) ** 2
    return np.exp(-0.5 * distance)


def render_aligned_masks(points, square, store_size):
    """渲染与 H5 裁剪帧对齐的 face mask、skin mask 和软 skin prior。"""
    points = transform_landmarks_to_stored(points, square, store_size)
    face_mask = _polygon_mask(points, FACE_OVAL, store_size)

    eye_left = points[list(LEFT_EYE)].mean(axis=0)
    eye_right = points[list(RIGHT_EYE)].mean(axis=0)
    eye_mid = (eye_left + eye_right) * 0.5
    mouth = points[list(OUTER_LIPS)].mean(axis=0)
    forehead = points[FOREHEAD_TOP] * 0.55 + eye_mid * 0.45
    cheek_left = points[list(LEFT_CHEEK)].mean(axis=0)
    cheek_right = points[list(RIGHT_CHEEK)].mean(axis=0)

    basis_x = eye_right - eye_left
    basis_x /= max(float(np.linalg.norm(basis_x)), 1e-6)
    basis_y = np.array([-basis_x[1], basis_x[0]], dtype=np.float32)
    if np.dot(basis_y, mouth - eye_mid) < 0:
        basis_y = -basis_y

    face_width = max(
        float(np.linalg.norm(points[FACE_RIGHT] - points[FACE_LEFT])),
        1.0,
    )
    face_height = max(
        float(np.linalg.norm(points[CHIN] - points[FOREHEAD_TOP])),
        1.0,
    )

    # 论文图中的 mask 是三块分离的硬 ROI：宽额头区域和左右脸颊区域。
    # 多边形由随脸旋转的局部坐标构造，因此能适应平面内头部旋转。
    forehead_mask = _local_polygon_mask(
        forehead - basis_y * (0.05 * face_height),
        basis_x,
        basis_y,
        (
            (-0.39 * face_width, -0.16 * face_height),
            (0.39 * face_width, -0.16 * face_height),
            (0.34 * face_width, 0.13 * face_height),
            (-0.34 * face_width, 0.13 * face_height),
        ),
        store_size,
    )
    cheek_offsets = (
        (-0.16 * face_width, -0.18 * face_height),
        (0.16 * face_width, -0.16 * face_height),
        (0.18 * face_width, 0.12 * face_height),
        (0.00 * face_width, 0.24 * face_height),
        (-0.18 * face_width, 0.12 * face_height),
    )
    cheek_left_mask = _local_polygon_mask(
        cheek_left, basis_x, basis_y, cheek_offsets, store_size
    )
    cheek_right_mask = _local_polygon_mask(
        cheek_right, basis_x, basis_y, cheek_offsets, store_size
    )
    skin_mask = np.maximum.reduce(
        [forehead_mask, cheek_left_mask, cheek_right_mask]
    )
    skin_mask = (skin_mask * face_mask).astype(np.uint8)

    grid_y, grid_x = np.mgrid[0:store_size, 0:store_size].astype(np.float32)

    def gaussian(center, width_scale, height_scale):
        return _oriented_gaussian(
            grid_x,
            grid_y,
            center,
            basis_x,
            basis_y,
            face_width * width_scale,
            face_height * height_scale,
        )

    # 三个高斯核心对应额头和双颊，与编辑器动态解剖先验的语义保持一致。
    reliable = np.maximum.reduce(
        [
            gaussian(forehead, 0.23, 0.16),
            gaussian(cheek_left, 0.18, 0.20),
            gaussian(cheek_right, 0.18, 0.20),
        ]
    )
    unreliable = np.maximum.reduce(
        [
            gaussian(eye_left, 0.15, 0.08),
            gaussian(eye_right, 0.15, 0.08),
            gaussian(mouth, 0.18, 0.10),
        ]
    )
    skin_prior = reliable * (1.0 - unreliable) * skin_mask.astype(np.float32)

    # 额外显式排除眼睛和嘴，避免姿态变化时硬 ROI 与非皮肤区域轻微重叠。
    excluded = np.maximum.reduce(
        [
            _polygon_mask(points, LEFT_EYE, store_size),
            _polygon_mask(points, RIGHT_EYE, store_size),
            _polygon_mask(points, OUTER_LIPS, store_size),
        ]
    )
    skin_prior *= 1.0 - excluded.astype(np.float32)
    skin_prior = np.clip(skin_prior, 0.0, 1.0).astype(np.float32)
    return face_mask, skin_mask, skin_prior


def compute_global_bounding_square(rectangles, bbox_margin=0.0, top_priority=0.5):
    """合并全部帧矩形，并以并集中心扩展为固定正方形 [x,y,side]。"""
    if not rectangles:
        raise RuntimeError("MediaPipe did not detect a face in any frame")
    if bbox_margin < 0:
        raise ValueError("bbox_margin must be non-negative")
    if not 0.0 <= top_priority <= 1.0:
        raise ValueError("top_priority must be in [0, 1]")

    rectangles = np.asarray(rectangles, dtype=np.float64)
    x1 = rectangles[:, 0].min()
    y1 = rectangles[:, 1].min()
    x2 = rectangles[:, 2].max()
    y2 = rectangles[:, 3].max()
    width = x2 - x1
    height = y2 - y1
    side = max(width, height) * (1.0 + 2.0 * bbox_margin)
    side = max(1, int(np.ceil(side)))

    horizontal_extra = side - width
    vertical_extra = side - height
    square_x = int(np.floor(x1 - horizontal_extra * 0.5))
    square_y = int(np.floor(y1 - vertical_extra * top_priority))
    return square_x, square_y, side


def landmark_points_to_square(
    points,
    frame_height,
    frame_width,
    bbox_margin=0.0,
    top_priority=0.5,
):
    """由单帧 landmarks 计算紧贴人脸的正方形裁剪框。"""
    rectangle = pixel_points_to_rectangle(points, frame_height, frame_width)
    square = compute_global_bounding_square(
        [rectangle],
        bbox_margin=bbox_margin,
        top_priority=top_priority,
    )
    return fit_square_to_frame(square, frame_height, frame_width)


def fit_square_to_frame(square, frame_height, frame_width):
    """在不改变边长的前提下将方框移入画面；过大时保留居中越界。"""
    x, y, side = square
    if side <= frame_width:
        x = min(max(x, 0), frame_width - side)
    else:
        x = int(np.floor((frame_width - side) / 2.0))
    if side <= frame_height:
        y = min(max(y, 0), frame_height - side)
    else:
        y = int(np.floor((frame_height - side) / 2.0))
    return x, y, side


def detect_global_bounding_square(
    dataset_name,
    source_path,
    bbox_margin=0.10,
    top_priority=0.5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    """检测整段视频，返回共享正方形和时间插值后的 landmark 序列。"""
    try:
        import mediapipe as mp
    except ImportError as error:
        raise ImportError(
            "MediaPipe preprocessing requires `mediapipe` and its native "
            f"dependencies. Original error: {error}"
        ) from error

    rectangles = []
    landmark_sequence = []
    total_frames = 0
    frame_shape = None
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh:
        for frame in iter_rgb_frames(dataset_name, source_path):
            total_frames += 1
            if frame_shape is None:
                frame_shape = frame.shape
            elif frame.shape != frame_shape:
                raise ValueError("All frames in one video must have the same shape")

            result = face_mesh.process(np.ascontiguousarray(frame))
            if not result.multi_face_landmarks:
                landmark_sequence.append(None)
                continue
            points = landmarks_to_pixel_points(
                result.multi_face_landmarks[0].landmark,
                frame.shape[0],
                frame.shape[1],
            )
            landmark_sequence.append(points)
            rectangles.append(
                pixel_points_to_rectangle(points, frame.shape[0], frame.shape[1])
            )

    if total_frames == 0:
        raise RuntimeError(f"No frames found in {source_path}")
    square = compute_global_bounding_square(
        rectangles,
        bbox_margin=bbox_margin,
        top_priority=top_priority,
    )
    square = fit_square_to_frame(square, frame_shape[0], frame_shape[1])
    landmarks = interpolate_landmark_sequence(landmark_sequence)
    return square, total_frames, len(rectangles), landmarks


def crop_square_with_padding(frame, square):
    """按固定方框裁剪；越界区域用黑色填充，保持整段视频几何坐标固定。"""
    x, y, side = square
    output = np.zeros((side, side, 3), dtype=frame.dtype)
    source_x1 = max(0, x)
    source_y1 = max(0, y)
    source_x2 = min(frame.shape[1], x + side)
    source_y2 = min(frame.shape[0], y + side)
    if source_x1 >= source_x2 or source_y1 >= source_y2:
        return output

    target_x1 = source_x1 - x
    target_y1 = source_y1 - y
    target_x2 = target_x1 + source_x2 - source_x1
    target_y2 = target_y1 + source_y2 - source_y1
    output[target_y1:target_y2, target_x1:target_x2] = frame[
        source_y1:source_y2, source_x1:source_x2
    ]
    return output


def save_preview(
    frame,
    crop,
    stored,
    face_mask,
    skin_mask,
    skin_prior,
    square,
    preview_dir,
    frame_index,
):
    preview_dir.mkdir(parents=True, exist_ok=True)
    x, y, side = square
    bbox_frame = frame.copy()
    cv2.rectangle(
        bbox_frame,
        (max(0, x), max(0, y)),
        (min(frame.shape[1] - 1, x + side - 1), min(frame.shape[0] - 1, y + side - 1)),
        (0, 255, 0),
        2,
    )
    for name, image in {"bbox": bbox_frame, "crop": crop, "stored": stored}.items():
        path = preview_dir / f"{frame_index:04d}_{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    mask_images = {
        "face_mask": face_mask * 255,
        "skin_mask": skin_mask * 255,
        "skin_prior": np.rint(skin_prior * 255).astype(np.uint8),
    }
    for name, image in mask_images.items():
        cv2.imwrite(str(preview_dir / f"{frame_index:04d}_{name}.png"), image)

    overlay = stored.astype(np.float32)
    color = np.zeros_like(overlay)
    color[..., 1] = 255
    weight = skin_prior[..., None] * 0.55
    overlay = np.rint(overlay * (1.0 - weight) + color * weight).astype(np.uint8)
    cv2.imwrite(
        str(preview_dir / f"{frame_index:04d}_skin_overlay.jpg"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )


def mediapipe_masks_h5(
    dataset_name,
    source_path,
    json_path,
    h5_path,
    store_size=192,
    bbox_margin=0.10,
    top_priority=0.5,
    crop_mode="frame",
    preview_frames=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    """裁剪视频并将对齐的面部/可靠皮肤掩码写入 H5。"""
    if store_size < 1:
        raise ValueError("store_size must be positive")
    if preview_frames < 0:
        raise ValueError("preview_frames must be non-negative")
    if crop_mode not in {"global", "frame"}:
        raise ValueError("crop_mode must be either 'global' or 'frame'")
    if not 0.0 <= top_priority <= 1.0:
        raise ValueError("top_priority must be in [0, 1]")

    square, total_frames, detected_frames, landmark_sequence = (
        detect_global_bounding_square(
            dataset_name,
            source_path,
            bbox_margin=bbox_margin,
            top_priority=top_priority,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    )
    wave_t, bvp, hr = read_ground_truth(
        dataset_name,
        total_frames,
        json_path,
    )
    validate_ground_truth_lengths(
        total_frames,
        wave_t,
        bvp,
        hr,
        source_path,
        json_path,
    )

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
        face_masks = file.create_dataset(
            "face_masks",
            shape=(total_frames, store_size, store_size),
            dtype="uint8",
            chunks=(1, store_size, store_size),
            compression="gzip",
            compression_opts=4,
        )
        skin_masks = file.create_dataset(
            "skin_masks",
            shape=(total_frames, store_size, store_size),
            dtype="uint8",
            chunks=(1, store_size, store_size),
            compression="gzip",
            compression_opts=4,
        )
        skin_priors = file.create_dataset(
            "skin_priors",
            shape=(total_frames, store_size, store_size),
            dtype="float16",
            chunks=(1, store_size, store_size),
            compression="gzip",
            compression_opts=4,
        )
        crop_squares = file.create_dataset(
            "crop_squares_xy_side",
            shape=(total_frames, 3),
            dtype="int32",
            chunks=(1, 3),
            compression="gzip",
            compression_opts=4,
        )
        written_frames = 0
        for frame_index, frame in enumerate(iter_rgb_frames(dataset_name, source_path)):
            frame_square = square
            if crop_mode == "frame":
                frame_square = landmark_points_to_square(
                    landmark_sequence[frame_index],
                    frame.shape[0],
                    frame.shape[1],
                    bbox_margin=bbox_margin,
                    top_priority=top_priority,
                )

            crop = crop_square_with_padding(frame, frame_square)
            stored = cv2.resize(crop, (store_size, store_size))
            face_mask, skin_mask, skin_prior = render_aligned_masks(
                landmark_sequence[frame_index],
                frame_square,
                store_size,
            )
            images[frame_index] = stored
            face_masks[frame_index] = face_mask
            skin_masks[frame_index] = skin_mask
            skin_priors[frame_index] = skin_prior.astype(np.float16)
            crop_squares[frame_index] = np.asarray(frame_square, dtype=np.int32)
            written_frames += 1
            if frame_index < preview_frames:
                save_preview(
                    frame,
                    crop,
                    stored,
                    face_mask,
                    skin_mask,
                    skin_prior,
                    frame_square,
                    preview_dir,
                    frame_index,
                )

        if written_frames != total_frames:
            raise RuntimeError(
                f"Frame count changed between passes: {total_frames} -> {written_frames}"
            )

        file.create_dataset("wave_t", data=wave_t)
        file.create_dataset("bvp", data=bvp)
        file.create_dataset("hr", data=hr)
        file.attrs["preprocessing_method"] = f"mediapipe_{crop_mode}_square_with_masks"
        file.attrs["crop_mode"] = crop_mode
        file.attrs["bbox_margin"] = bbox_margin
        file.attrs["top_priority"] = top_priority
        if crop_mode == "global":
            file.attrs["crop_square_xy_side"] = np.asarray(square, dtype=np.int32)
        file.attrs["skin_mask_definition"] = "forehead_and_bilateral_cheek_polygons"
        file.attrs["mask_landmark_missing_policy"] = "linear_interpolation"
        file.attrs["mediapipe_detected_frames"] = detected_frames
        file.attrs["total_frames"] = total_frames


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


def preprocess_dataset_item(task):
    (
        dataset_name,
        source_path,
        json_path,
        h5_path,
        store_size,
        bbox_margin,
        top_priority,
        crop_mode,
        preview_frames,
        min_detection_confidence,
        min_tracking_confidence,
    ) = task
    mediapipe_masks_h5(
        dataset_name,
        source_path,
        json_path,
        h5_path,
        store_size=store_size,
        bbox_margin=bbox_margin,
        top_priority=top_priority,
        crop_mode=crop_mode,
        preview_frames=preview_frames,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return str(source_path)


def build_preprocessing_tasks(items, args):
    tasks = []
    for name, source_path, json_path, relative_dir in items:
        h5_name = f"{name}_{args.output_tag}_s{args.store_size}.h5"
        h5_path = Path(args.h5_dir) / relative_dir / h5_name
        if args.skip_existing and h5_path.exists():
            continue
        tasks.append(
            (
                args.dataset_name,
                str(source_path),
                str(json_path),
                str(h5_path),
                args.store_size,
                args.bbox_margin,
                args.top_priority,
                args.crop_mode,
                args.preview_frames,
                args.min_detection_confidence,
                args.min_tracking_confidence,
            )
        )
    return tasks


def run_preprocessing_tasks(tasks, num_workers):
    if num_workers < 1:
        raise ValueError("--num_workers must be at least 1")
    if num_workers == 1:
        for task in tqdm(tasks, desc="video"):
            try:
                preprocess_dataset_item(task)
            except Exception as error:
                raise RuntimeError(f"Failed to preprocess {task[1]}: {error}") from error
        return

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as executor:
        futures = {
            executor.submit(preprocess_dataset_item, task): task for task in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="video"):
            task = futures[future]
            try:
                future.result()
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"Failed to preprocess {task[1]}: {error}") from error


def main():
    args = parse_args()
    if args.store_size < 1:
        raise ValueError("--store_size must be positive")
    if args.bbox_margin < 0:
        raise ValueError("--bbox_margin must be non-negative")
    if not 0.0 <= args.top_priority <= 1.0:
        raise ValueError("--top_priority must be in [0, 1]")
    if args.num_workers < 1:
        raise ValueError("--num_workers must be at least 1")

    if args.crop_mode == "global":
        method_description = (
            "Merge all framewise MediaPipe landmark rectangles and expand the "
            "union into one symmetric square; save aligned face masks, reliable "
            "forehead/cheek masks, and soft skin priors."
        )
    else:
        method_description = (
            "Compute one symmetric MediaPipe landmark square for each frame from "
            "interpolated landmarks; save aligned face masks, reliable "
            "forehead/cheek masks, and soft skin priors."
        )

    write_preprocessing_info(
        args.h5_dir,
        args.output_tag,
        args.dataset_name,
        args.video_dir,
        args.json_dir,
        args.store_size,
        args.bbox_margin,
        method=f"mediapipe_{args.crop_mode}_square_with_masks",
        method_description=method_description,
        extra_parameters={
            "crop_mode": args.crop_mode,
            "top_priority": args.top_priority,
            "bottom_priority": round(1.0 - args.top_priority, 6),
            "min_detection_confidence": args.min_detection_confidence,
            "min_tracking_confidence": args.min_tracking_confidence,
            "preview_frames": args.preview_frames,
            "skip_existing": args.skip_existing,
            "num_workers": args.num_workers,
            "skin_mask_definition": "forehead and bilateral cheek polygons",
            "missing_landmark_policy": "linear interpolation",
        },
    )

    items = list(iter_dataset_items(args.dataset_name, args.video_dir, args.json_dir))
    tasks = build_preprocessing_tasks(items, args)
    run_preprocessing_tasks(tasks, args.num_workers)


if __name__ == "__main__":
    main()
