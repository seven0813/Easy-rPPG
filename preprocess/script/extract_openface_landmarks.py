import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from tqdm import tqdm


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract OpenFace landmarks for video datasets or PURE image sequences."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["video", "pure"],
        help="Choose 'video' for video files or 'pure' for PURE image sequences.",
    )
    parser.add_argument(
        "--openface_bin",
        required=True,
        type=str,
        help="Path to OpenFace FeatureExtraction binary.",
    )
    parser.add_argument(
        "--input_root",
        required=True,
        type=str,
        help="Root directory containing input data.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        type=str,
        help="Root directory for OpenFace outputs.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        type=str,
        help="Glob pattern for video file names in video mode.",
    )
    parser.add_argument(
        "--session_pattern",
        default="*-*",
        type=str,
        help="Glob pattern for PURE session directories in pure mode.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under input_root.",
    )
    parser.add_argument(
        "--two_d_only",
        action="store_true",
        help="Pass -2Dfp to OpenFace and only export 2D landmarks.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip an item if the expected OpenFace CSV already exists.",
    )
    parser.add_argument(
        "--quiet_openface",
        dest="quiet_openface",
        action="store_true",
        default=True,
        help="Silence OpenFace stdout/stderr. Enabled by default.",
    )
    parser.add_argument(
        "--show_openface_output",
        dest="quiet_openface",
        action="store_false",
        help="Show OpenFace stdout/stderr for debugging.",
    )
    return parser.parse_args()


def resolve_openface_binary(openface_path: Path):
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


def find_videos(input_root: Path, pattern: str, recursive: bool):
    iterator = input_root.rglob(pattern) if recursive else input_root.glob(pattern)
    videos = []
    for path in iterator:
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    return sorted(videos)


def is_pure_sequence_dir(sequence_dir: Path):
    if not sequence_dir.is_dir():
        return False

    image_dir = sequence_dir / sequence_dir.name
    if not image_dir.is_dir():
        return False

    return any(
        image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        for image_path in image_dir.iterdir()
    )


def find_sequences(input_root: Path, pattern: str, recursive: bool):
    iterator = input_root.rglob(pattern) if recursive else input_root.glob(pattern)
    sequences = [path for path in iterator if is_pure_sequence_dir(path)]
    return sorted(sequences)


def build_video_command(openface_bin: Path, video_path: Path, output_dir: Path, two_d_only: bool):
    command = [str(openface_bin), "-f", str(video_path), "-out_dir", str(output_dir)]
    if two_d_only:
        command.append("-2Dfp")
    return command


def build_pure_command(openface_bin: Path, image_dir: Path, output_dir: Path, two_d_only: bool):
    command = [str(openface_bin), "-fdir", str(image_dir), "-out_dir", str(output_dir)]
    if two_d_only:
        command.append("-2Dfp")
    return command


def move_csv_from_temp(temp_dir: Path, final_output_dir: Path, csv_name: str):
    temp_csv = temp_dir / csv_name
    final_csv = final_output_dir / csv_name
    if not temp_csv.exists():
        raise FileNotFoundError(f"Expected CSV not found in temp dir: {temp_csv}")
    shutil.move(str(temp_csv), str(final_csv))


def remove_temp_dir(temp_dir: Path):
    if temp_dir.exists():
        shutil.rmtree(temp_dir)



def run_openface_with_error_capture(command, quiet_openface: bool):
    run_kwargs = {"check": True, "text": True, "capture_output": True}
    if not quiet_openface:
        run_kwargs.pop("capture_output")
    return subprocess.run(command, **run_kwargs)


def get_video_fps(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return fps


def check_video_fps(video_path: Path, expected_fps: float = 30.0, tolerance: float = 0.5):
    fps = get_video_fps(video_path)
    if fps <= 0:
        print(f"[WARN] 无法读取视频帧率: {video_path}")
        return

    if abs(fps - expected_fps) > tolerance:
        print(f"[WARN] 视频帧率不是 {expected_fps:g} fps: {video_path} (检测到 {fps:.2f} fps)")


def process_video_mode(args, openface_bin: Path, input_root: Path, output_root: Path):
    videos = find_videos(input_root, args.pattern, args.recursive)
    if not videos:
        print("No videos found.")
        return

    print(f"Found {len(videos)} videos.")

    for index, video_path in enumerate(tqdm(videos, desc="videos", ncols=80), start=1):
        rel_parent = video_path.parent.relative_to(input_root)
        output_dir = output_root / rel_parent
        output_dir.mkdir(parents=True, exist_ok=True)

        expected_csv = output_dir / f"{video_path.stem}.csv"
        if args.skip_existing and expected_csv.exists():
            tqdm.write(f"Skip existing [{index}/{len(videos)}]: {video_path}")
            continue

        temp_output_dir = Path(
            tempfile.mkdtemp(prefix=f".openface_{video_path.stem}_", dir=str(output_dir))
        )

        check_video_fps(video_path)

        command = build_video_command(openface_bin, video_path, temp_output_dir, args.two_d_only)
        tqdm.write(f"Running [{index}/{len(videos)}]: {video_path}")
        try:
            run_openface_with_error_capture(command, args.quiet_openface)
            move_csv_from_temp(temp_output_dir, output_dir, f"{video_path.stem}.csv")
        except subprocess.CalledProcessError as exc:
            tqdm.write(f"[ERROR] Failed [{index}/{len(videos)}]: {video_path}")
            if exc.stdout:
                tqdm.write(exc.stdout.rstrip())
            if exc.stderr:
                tqdm.write(exc.stderr.rstrip())
            raise
        finally:
            remove_temp_dir(temp_output_dir)

    print("Video OpenFace extraction finished.")


def process_pure_mode(args, openface_bin: Path, input_root: Path, output_root: Path):
    sequences = find_sequences(input_root, args.session_pattern, args.recursive)
    if not sequences:
        print("No PURE sequences found.")
        return

    print(f"Found {len(sequences)} PURE sequences.")

    for index, sequence_dir in enumerate(tqdm(sequences, desc="pure", ncols=80), start=1):
        rel_path = sequence_dir.relative_to(input_root)
        image_dir = sequence_dir / sequence_dir.name
        output_dir = output_root / rel_path
        output_dir.mkdir(parents=True, exist_ok=True)

        expected_csv = output_dir / f"{sequence_dir.name}.csv"
        if args.skip_existing and expected_csv.exists():
            tqdm.write(f"Skip existing [{index}/{len(sequences)}]: {sequence_dir}")
            continue

        temp_output_dir = Path(
            tempfile.mkdtemp(prefix=f".openface_{sequence_dir.name}_", dir=str(output_dir))
        )

        command = build_pure_command(openface_bin, image_dir, temp_output_dir, args.two_d_only)
        tqdm.write(f"Running [{index}/{len(sequences)}]: {sequence_dir}")
        try:
            run_openface_with_error_capture(command, args.quiet_openface)
            move_csv_from_temp(temp_output_dir, output_dir, f"{sequence_dir.name}.csv")
        except subprocess.CalledProcessError as exc:
            tqdm.write(f"[ERROR] Failed [{index}/{len(sequences)}]: {sequence_dir}")
            if exc.stdout:
                tqdm.write(exc.stdout.rstrip())
            if exc.stderr:
                tqdm.write(exc.stderr.rstrip())
            raise
        finally:
            remove_temp_dir(temp_output_dir)

    print("PURE OpenFace extraction finished.")


def main():
    args = parse_args()

    openface_bin = resolve_openface_binary(Path(args.openface_bin).expanduser().resolve())
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"input_root is not a directory: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "video":
        process_video_mode(args, openface_bin, input_root, output_root)
    else:
        process_pure_mode(args, openface_bin, input_root, output_root)


if __name__ == "__main__":
    main()
