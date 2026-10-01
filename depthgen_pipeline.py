"""Generate Wan RGB video, estimate VDA depth, and create a side-by-side preview."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
WAN_WRAPPER = WORKSPACE / "novaflow_video.py"
VDA_DIR = WORKSPACE / "Video-Depth-Anything"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional initial RGB image; omit for text-to-video")
    parser.add_argument("--prompt", required=True, help="Wan video prompt")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run output directory")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--size", choices=("832*480", "480*832"), default="832*480")
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl"), default="vitl")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-res", type=int, default=1280)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run(command, *, cwd=None, log_path=None):
    if log_path:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
    else:
        result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode:
        location = f"; see {log_path}" if log_path else ""
        raise RuntimeError(f"Command failed with exit code {result.returncode}{location}")


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    return streams[0]


def build_commands(args, output_dir):
    rgb_path = output_dir / "rgb.mp4"
    if args.image:
        wan_command = [
            sys.executable, str(WAN_WRAPPER),
            "--first-frame", str(args.image.resolve()),
            "--prompt", args.prompt,
            "--output", str(rgb_path),
            "--seed", str(args.seed),
            "--frames", str(args.frames),
            "--steps", str(args.steps),
            "--shift", str(args.shift),
            "--guide-scale", str(args.guide_scale),
            "--size", args.size,
            "--overwrite",
        ]
    else:
        wan_command = [
            sys.executable, str(WORKSPACE / "Wan2.1" / "generate.py"),
            "--task", "t2v-1.3B",
            "--size", args.size,
            "--ckpt_dir", str(WORKSPACE / "Wan2.1" / "Wan2.1-T2V-1.3B"),
            "--prompt", args.prompt,
            "--base_seed", str(args.seed),
            "--frame_num", str(args.frames),
            "--sample_solver", "unipc",
            "--sample_steps", str(args.steps),
            "--sample_shift", str(args.shift),
            "--sample_guide_scale", str(args.guide_scale),
            "--t5_cpu",
            "--offload_model", "True",
            "--save_file", str(rgb_path),
        ]
    depth_command = [
        sys.executable, "run.py",
        "--input_video", str(rgb_path),
        "--output_dir", str(output_dir / "vda"),
        "--encoder", args.encoder,
        "--input_size", str(args.input_size),
        "--max_res", str(args.max_res),
        "--save_npz",
        "--grayscale",
    ]
    if args.fp32:
        depth_command.append("--fp32")
    return rgb_path, wan_command, depth_command


def main(argv=None):
    args = parse_args(argv)
    image = args.image.resolve() if args.image else None
    output_dir = args.output_dir.resolve()
    if image and not image.is_file():
        print(f"Error: image not found: {image}", file=sys.stderr)
        return 2
    if args.frames < 5 or args.frames % 4 != 1:
        print("Error: --frames must be 4n+1, for example 41 or 81", file=sys.stderr)
        return 2
    if args.seed < 0:
        print("Error: --seed must be nonnegative", file=sys.stderr)
        return 2
    vda_checkpoint = VDA_DIR / "checkpoints" / f"video_depth_anything_{args.encoder}.pth"
    if not vda_checkpoint.is_file():
        print(f"Error: VDA checkpoint not found: {vda_checkpoint}", file=sys.stderr)
        return 2
    if not image:
        t2v_checkpoint = WORKSPACE / "Wan2.1" / "Wan2.1-T2V-1.3B"
        if not (t2v_checkpoint / "diffusion_pytorch_model.safetensors").is_file():
            print(f"Error: T2V checkpoint not found: {t2v_checkpoint}", file=sys.stderr)
            return 2
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        print(f"Error: output directory is not empty; pass --overwrite: {output_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path, wan_command, depth_command = build_commands(args, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "planned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "image-to-video" if image else "text-to-video",
        "input": str(image) if image else None,
        "prompt": args.prompt,
        "seed": args.seed,
        "wan": {"frames": args.frames, "size": args.size, "steps": args.steps, "shift": args.shift, "guide_scale": args.guide_scale},
        "vda": {"encoder": args.encoder, "input_size": args.input_size, "max_res": args.max_res, "depth_mode": "relative", "raw_depth_units": None},
        "artifacts": {
            "rgb": str(rgb_path),
            "depth_preview": str(output_dir / "depth.mp4"),
            "rgb_depth_side_by_side": str(output_dir / "rgb_depth_side_by_side.mp4"),
            "depth_npz": str(output_dir / "depth.npz"),
        },
        "commands": {"wan": wan_command, "vda": depth_command},
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    start = time.monotonic()
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        run(wan_command, cwd=WORKSPACE, log_path=output_dir / "wan.log")
        rgb_meta = probe(rgb_path)
        if int(rgb_meta.get("nb_read_frames", 0)) != args.frames:
            raise RuntimeError(f"Wan produced {rgb_meta.get('nb_read_frames')} frames; expected {args.frames}")

        run(depth_command, cwd=VDA_DIR, log_path=output_dir / "vda.log")
        stem = rgb_path.stem
        vda_dir = output_dir / "vda"
        vda_preview = vda_dir / f"{stem}_vis.mp4"
        vda_npz = vda_dir / f"{stem}_depths.npz"
        if not vda_preview.is_file() or not vda_npz.is_file():
            raise RuntimeError("VDA completed without both visualization MP4 and raw NPZ")

        depth_preview = output_dir / "depth.mp4"
        depth_npz = output_dir / "depth.npz"
        shutil.copy2(vda_preview, depth_preview)
        shutil.copy2(vda_npz, depth_npz)
        depth_meta = probe(depth_preview)
        if int(depth_meta.get("nb_read_frames", 0)) != args.frames:
            raise RuntimeError("RGB and depth preview frame counts do not match")

        side_by_side = output_dir / "rgb_depth_side_by_side.mp4"
        width = int(rgb_meta["width"])
        height = int(rgb_meta["height"])
        ffmpeg = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(rgb_path), "-i", str(depth_preview),
            "-filter_complex", f"[0:v]setpts=PTS-STARTPTS[rgb];[1:v]scale={width}:{height},setpts=PTS-STARTPTS[depth];[rgb][depth]hstack=inputs=2:shortest=1[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(side_by_side),
        ]
        run(ffmpeg, cwd=WORKSPACE, log_path=output_dir / "side_by_side.log")
        manifest["status"] = "complete"
        manifest["metadata"] = {"rgb": probe(rgb_path), "depth_preview": probe(depth_preview), "side_by_side": probe(side_by_side)}
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        print(f"Pipeline failed: {error}", file=sys.stderr)
    finally:
        manifest["duration_seconds"] = round(time.monotonic() - start, 2)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if manifest["status"] == "complete":
        print(f"RGB: {rgb_path}")
        print(f"Depth: {output_dir / 'depth.mp4'}")
        print(f"Side-by-side: {output_dir / 'rgb_depth_side_by_side.mp4'}")
        print(f"Raw depth: {output_dir / 'depth.npz'}")
        print(f"Manifest: {manifest_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
