# cpr-video-poc

A production-oriented, config-driven Python project for a CPR-focused text-to-video diffusion POC.

The repo now supports two execution modes:

- direct local generation for debugging
- controller/worker orchestration where you submit jobs locally and execute them on Google Colab through a shared storage root

That shared-root design keeps the core pipeline infra-independent, so later you can replace Colab with a rented GPU box without rewriting the generation code.

## Current backend

Primary backend: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`

The implementation uses `diffusers.WanPipeline` with memory-aware loading patterns suitable for Colab-style environments, including optional CPU offload. The official Diffusers Wan docs list the supported Wan pipelines, and the model card recommends `832x480` as the most stable target for the 1.3B checkpoint.

## Architecture

```text
local machine
-> submit job into shared root
-> Colab worker polls shared root
-> worker runs Wan generation on T4
-> artifacts/status/metadata are written back into shared root
-> local machine sees updates through the same synced folder
```

For Google Colab, the shared root is typically a Google Drive folder mounted in Colab and synced locally via Google Drive Desktop.

## Repository layout

```text
cpr-video-poc/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── project.yaml
│   └── backends/
│       └── wan_t2v_1_3b.yaml
├── src/cpr_video_poc/
│   ├── settings.py
│   ├── paths.py
│   ├── logging_utils.py
│   ├── backends/
│   ├── orchestration/
│   │   ├── job_store.py
│   │   └── worker.py
│   ├── pipeline/
│   ├── storage/
│   ├── utils/
│   └── cli/
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   └── 01_generate.ipynb
├── runs/
├── data/
└── tests/
```

## Quickstart

### 1. Install

```bash
pip install -U pip
pip install -e .[dev]
```

### 2. Fast local checks

```bash
pytest -q
python -m cpr_video_poc.cli.main --help
```

### 3. Submit a job locally

Pick a shared folder that both your local machine and Colab can see as the same underlying Drive content.

Example local submit:

```bash
python -m cpr_video_poc.cli.main submit \
  --shared-root "/path/to/google-drive/cpr-video-poc-shared" \
  --prompt "Generate a video with Asian people giving CPR in an office" \
  --seed 42
```

This writes a queued job into:

```text
{shared_root}/jobs/{job_id}/
```

### 4. Process jobs on Colab

On Colab, mount Drive and run:

```bash
python -m cpr_video_poc.cli.main worker \
  --shared-root "/content/drive/MyDrive/cpr-video-poc-shared" \
  --once \
  --json
```

Or keep polling continuously:

```bash
python -m cpr_video_poc.cli.main worker \
  --shared-root "/content/drive/MyDrive/cpr-video-poc-shared"
```

### 5. Check job status locally

```bash
python -m cpr_video_poc.cli.main status \
  --shared-root "/path/to/google-drive/cpr-video-poc-shared"
```

Detailed view for a single job:

```bash
python -m cpr_video_poc.cli.main status \
  --shared-root "/path/to/google-drive/cpr-video-poc-shared" \
  --job-id "<job_id>" \
  --json
```

## Shared-root contents

```text
{shared_root}/
├── jobs/
│   └── {job_id}/
│       ├── job.json
│       ├── status.json
│       ├── project_config.yaml
│       ├── backend_config.yaml
│       ├── result.json          # completed jobs
│       └── error.txt            # failed jobs
└── runs/
    └── {timestamp_run_id}/
        ├── request.json
        ├── parsed_prompt.json
        ├── final_prompt.txt
        ├── negative_prompt.txt
        ├── config.yaml
        ├── metadata.json
        ├── output.mp4
        └── preview.gif
```

## Default generation config

The shipped defaults target a short Colab-friendly clip:

- backend: `wan_t2v_1_3b`
- resolution: `832x480`
- frames: `21`
- steps: `30`
- guidance scale: `5.0`
- fps: `8`

## Resume behavior

Current resume support is job-level, not mid-diffusion-step resume.

That means:

- if a worker dies before completion, the job remains visible in shared storage
- a new worker can reclaim stale jobs and rerun them with the same prompt, seed, and config
- reproducibility is preserved, but partial inference progress is not checkpointed yet

True mid-run resume would require saving diffusion latents and scheduler state during inference, which is intentionally deferred for now.

## Colab notes

This repo avoids hardcoded Colab paths in the core library. Only the notebook and worker invocation need a Colab-specific shared-root path.

The same controller/worker flow can later be moved to another GPU host by changing:

- where the worker runs
- which shared storage path it reads from

The backend and pipeline code can stay the same.

## Evaluation prompts

Configured examples:

- CPR with Asian responders in office
- CPR with African responders in office
- CPR in classroom
- CPR with one responder
- CPR workplace safety scenario

## What is intentionally deferred

Not implemented yet:

- model training
- LoRA training or adapters
- control networks
- retrieval system
- SharePoint backend
- true mid-step diffusion checkpoint resume
- frontend UI
