# Depth-Gen

**Depth-Gen** is an early-stage research and engineering pipeline for generating training data for robotic manipulation. The current prototype uses a video-generation model to synthesize RGB interaction videos and **Video-Depth-Anything (VDA)** to estimate temporally consistent depth from the generated video.

The long-term goal is to generate scalable, controllable, spatially grounded video data that can support robot learning, manipulation policy training, and scene understanding without requiring every training example to be collected on physical hardware.

> **Status:** Prototype / engineering validation
>
> The current implementation validates the RGB-video-to-depth-data workflow. The next phase will investigate grounding Wan2.1 with multiple policies so that spatial understanding is introduced during video generation rather than estimated only after generation.

## Motivation

Collecting robot manipulation data is expensive, slow, and difficult to scale. Physical data collection also limits the diversity of scenes, object configurations, camera viewpoints, and interaction trajectories that can be explored.

This project investigates a synthetic-data alternative:

1. Generate a manipulation-oriented RGB video from text or an initial image.
2. Estimate depth for every generated frame.
3. Package the RGB video, depth video, raw depth arrays, and run metadata as a reproducible training-data sample.
4. Extend the video-generation process with spatial and policy-level conditioning.

The prototype deliberately separates **generation** from **depth estimation**. This makes it possible to evaluate the quality of generated motion and the quality of recovered spatial structure independently before building a larger end-to-end research system.

## Current pipeline

```text
Text prompt or initial RGB image
              |
              v
        Wan2.1 video generation
              |
              v
          Generated RGB video
              |
              v
    Video-Depth-Anything inference
              |
              v
 Relative depth visualization + raw depth NPZ
              |
              v
 RGB/depth preview and manifest for analysis
```

The pipeline is implemented in [`depthgen_pipeline.py`](depthgen_pipeline.py). It orchestrates the following stages:

### 1. RGB video generation

The pipeline supports two modes:

- **Text-to-video:** uses the local Wan2.1 T2V generator and the `Wan2.1-T2V-1.3B` checkpoint.
- **Image-to-video:** uses `novaflow_video.py` with an initial image as the first frame.

The generated RGB video is written as `rgb.mp4` in the run output directory.

### 2. Depth estimation

The generated RGB video is passed to Video-Depth-Anything. VDA produces:

- A visualized depth video.
- Raw depth predictions saved as an `.npz` file.

The current depth output is **relative depth**, not calibrated metric depth. It should therefore be interpreted as a spatial ordering or scene-structure signal unless an additional calibration step is introduced.

### 3. Validation and packaging

The pipeline validates that:

- Required model checkpoints exist.
- The requested frame count is valid for Wan2.1.
- Wan2.1 generated the expected number of frames.
- VDA produced both a preview video and raw depth data.
- RGB and depth previews have matching frame counts.

It then creates a side-by-side RGB/depth preview and writes a `manifest.json` containing the configuration, commands, artifacts, metadata, status, and runtime information.

## Repository layout

```text
Depth-Gen/
├── depthgen_pipeline.py       # Main orchestration script
├── Wan2.1/                    # Wan2.1 source tree and model checkpoints
├── Video-Depth-Anything/      # VDA source tree and checkpoints
└── README.md
```

The model directories are expected to be available locally. Large model weights are not intended to be committed to the repository.

## Requirements

The prototype assumes a Linux-like environment with:

- Python 3.10 or newer recommended.
- A CUDA-capable GPU with sufficient memory for the selected generation and depth models.
- PyTorch and the dependencies required by Wan2.1.
- The dependencies required by Video-Depth-Anything.
- `ffmpeg` and `ffprobe` available on `PATH`.
- Wan2.1 model weights.
- VDA model weights.

Verify the video tools before running:

```bash
ffmpeg -version
ffprobe -version
```

Install the Python dependencies according to the setup instructions for the local Wan2.1 and Video-Depth-Anything directories. Since those components may evolve independently, their environment requirements should be treated as authoritative for their respective inference code.

## Model checkpoints

The current pipeline checks for the following files/directories:

```text
Wan2.1/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth
Video-Depth-Anything/checkpoints/video_depth_anything_vitb.pth
Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth
```

Only the VDA checkpoint selected with `--encoder` is required for a given run. The Wan2.1 T2V checkpoint is required for text-to-video runs; image-to-video runs use the image-generation wrapper path instead.

## Usage

### Text-to-video

Generate an RGB video from a prompt and estimate its depth:

```bash
python depthgen_pipeline.py \\
  --prompt "A robotic gripper reaches toward a red cube on a tabletop" \\
  --output-dir runs/red-cube-t2v
```

### Image-to-video

Use an initial RGB image as the first frame:

```bash
python depthgen_pipeline.py \\
  --image path/to/initial_frame.png \\
  --prompt "A robotic gripper moves toward the red cube and lifts it" \\
  --output-dir runs/red-cube-i2v
```

### Reproducible run with explicit parameters

```bash
python depthgen_pipeline.py \\
  --prompt "A robot arm pushes a blue block across a table" \\
  --output-dir runs/blue-block \\
  --seed 1234 \\
  --frames 81 \\
  --steps 40 \\
  --shift 3.0 \\
  --guide-scale 5.0 \\
  --size 832*480 \\
  --encoder vitl \\
  --input-size 518 \\
  --max-res 1280
```

Wan2.1 requires the frame count to satisfy `4n + 1`; examples include `41`, `81`, and `121`.

### Dry run

Use `--dry-run` to validate the configuration and print the commands without running model inference:

```bash
python depthgen_pipeline.py \\
  --prompt "A robot places a cube into a bin" \\
  --output-dir runs/dry-run \\
  --dry-run
```

### Overwriting an existing run

The pipeline refuses to write into a non-empty output directory unless explicitly requested:

```bash
python depthgen_pipeline.py \\
  --prompt "A robot grasps a cylinder" \\
  --output-dir runs/grasp-cylinder \\
  --overwrite
```

Use `--overwrite` carefully because it allows artifacts from a previous experiment to be replaced.

## Command-line options

| Option | Default | Description |
| --- | ---: | --- |
| `--image` | None | Optional initial image for image-to-video generation. |
| `--prompt` | Required | Text prompt passed to the video generator. |
| `--output-dir` | Required | Directory for all generated artifacts and logs. |
| `--seed` | `1234` | Non-negative generation seed. |
| `--frames` | `81` | Number of generated frames; must be `4n+1`. |
| `--steps` | `40` | Number of video-generation sampling steps. |
| `--shift` | `3.0` | Wan2.1 sampling shift. |
| `--guide-scale` | `5.0` | Wan2.1 guidance scale. |
| `--size` | `832*480` | Supported output resolution: `832*480` or `480*832`. |
| `--encoder` | `vitl` | VDA encoder: `vits`, `vitb`, or `vitl`. |
| `--input-size` | `518` | VDA inference input size. |
| `--max-res` | `1280` | Maximum resolution used by VDA preprocessing. |
| `--fp32` | Off | Run VDA inference in FP32 instead of the default precision. |
| `--overwrite` | Off | Allow writing into a non-empty output directory. |
| `--dry-run` | Off | Print the planned manifest and commands without inference. |

## Output artifacts

A successful run produces a directory similar to:

```text
runs/example/
├── rgb.mp4                    # Generated RGB video
├── depth.mp4                  # VDA depth visualization
├── depth.npz                  # Raw relative depth predictions
├── rgb_depth_side_by_side.mp4 # RGB and depth visualization side by side
├── manifest.json              # Configuration, commands, metadata, and status
├── wan.log                    # Wan2.1 generation log
├── vda.log                    # VDA inference log
├── side_by_side.log           # FFmpeg packaging log
└── vda/                       # Native VDA output directory
```


https://github.com/user-attachments/assets/3a21e395-1036-428a-af5c-21a6caf27783



https://github.com/user-attachments/assets/ad6e6265-5743-4780-b28e-967dc0c6945e



The `manifest.json` is intended to make each generated sample traceable. It records the prompt, seed, generation parameters, VDA parameters, artifact paths, executed commands, timestamps, duration, and failure information when applicable.

## Interpreting the depth output

The depth output should currently be treated as a **relative spatial representation**:

- Lower or higher values indicate relative ordering according to the VDA output convention.
- Values are not guaranteed to be metric distances.
- The output is not automatically aligned to a robot coordinate frame.
- Camera intrinsics, extrinsics, scale, and temporal calibration are not currently recovered by this pipeline.
- Depth quality depends on both the generated RGB content and the generalization behavior of VDA.

For robotic manipulation, future dataset versions will likely need camera calibration, metric-scale recovery, coordinate-frame conventions, confidence estimates, and consistency checks against known object geometry.

## Early validation results

The first engineering experiment validated the complete prototype path:

```text
Wan2.1 RGB generation
        -> VDA depth estimation
        -> raw depth export
        -> side-by-side RGB/depth preview
        -> manifest-backed experiment record
```

This result is an important feasibility check, but it is not yet a benchmark of robotic manipulation performance. The current pipeline demonstrates that generated videos can be converted into synchronized RGB/depth artifacts; it does not yet establish that the resulting data is physically accurate, policy-useful, or sufficient for training a manipulation controller.

When quantitative results are available, this section should be expanded with:

- Prompt and scene categories.
- Video-generation settings.
- Depth-estimation settings.
- Qualitative visualizations.
- Temporal consistency measurements.
- Depth error against simulated or sensor ground truth.
- Object-boundary and contact-region quality.
- Downstream policy-training results.

## Research direction: spatially grounded Wan2.1

The next phase is to move from post-hoc depth estimation toward **spatially grounded video generation**. The planned direction is to ground Wan2.1 using multiple policies or spatial control signals so that generated motion is constrained by the structure required for robotic manipulation.

Potential conditioning signals include:

- Depth or disparity maps.
- Robot pose and end-effector trajectories.
- Object masks, bounding boxes, and keypoints.
- Contact points and grasp affordances.
- Scene geometry and camera parameters.
- Low-level or high-level manipulation policies.
- Temporal action sequences.
- Collision, reachability, and physical-consistency constraints.

A possible future architecture is:

```text
Scene/task specification
        |
        +--> manipulation policy / action trajectory
        +--> object and affordance representation
        +--> depth / geometry representation
        |
        v
   Spatial conditioning module
        |
        v
      Wan2.1 generation
        |
        v
 RGB + depth + actions + masks + metadata
```

The key research question is whether policy-conditioned generation can produce videos that are not only visually plausible, but also spatially useful for learning manipulation behaviors.

## Planned roadmap

### Phase 1: Prototype validation

- Stabilize the RGB-to-depth pipeline.
- Record reproducible manifests for every run.
- Compare prompts, seeds, resolutions, and VDA encoders.
- Inspect temporal depth consistency and object boundaries.

### Phase 2: Dataset representation

- Define a standard sample schema.
- Store synchronized RGB, depth, actions, masks, camera data, and task labels.
- Add validation for frame alignment and missing artifacts.
- Add dataset indexing and train/validation/test splits.

### Phase 3: Spatial conditioning

- Introduce depth, segmentation, keypoint, and trajectory controls.
- Investigate multi-policy conditioning strategies.
- Measure whether conditioning improves spatial consistency and action relevance.

### Phase 4: Robotic manipulation evaluation

- Train policies on generated data.
- Evaluate transfer to simulation and real hardware.
- Measure grasp success, placement success, collision rate, and robustness to scene variation.
- Compare generated-only, real-only, and mixed-data training regimes.

## Limitations

This repository is an engineering prototype and currently has several limitations:

- Generated videos are not guaranteed to obey real-world physics.
- VDA produces relative depth rather than calibrated metric depth.
- No robot actions or policy labels are currently generated by `depthgen_pipeline.py`.
- No camera calibration or robot/world coordinate transform is produced.
- The pipeline does not yet verify contact, grasp validity, collision avoidance, or object permanence.
- Visual plausibility is not equivalent to manipulation usefulness.
- Model checkpoints and environment setup are external dependencies.
- The current side-by-side preview is for inspection and is not a training representation by itself.

## Reproducibility and experiment hygiene

For each experiment:

1. Use a unique output directory.
2. Keep the prompt, seed, model versions, and command-line arguments in the manifest.
3. Preserve `wan.log`, `vda.log`, and `side_by_side.log` when comparing failures.
4. Record whether the run was text-to-video or image-to-video.
5. Do not interpret relative depth as metric depth without calibration.
6. Keep generated artifacts separate from source code and model checkpoints.

## Contributing

Contributions are welcome, especially in the following areas:

- Dataset schema design.
- Temporal depth consistency evaluation.
- Spatial and policy conditioning for Wan2.1.
- Robot-camera calibration and coordinate transforms.
- Synthetic-to-real validation.
- Automated quality checks for generated manipulation videos.

Before opening a pull request, include the command used to reproduce the change and describe any required checkpoints or external model repositories.

## License and model terms

This repository contains orchestration code and expects external model implementations and checkpoints. Review the licenses and usage terms of Wan2.1, Video-Depth-Anything, PyTorch, FFmpeg, and any other dependencies before redistributing model weights or generated datasets.
