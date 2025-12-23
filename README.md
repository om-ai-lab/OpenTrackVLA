# OpenTrackVLA 🤖 👀

**Visual Navigation & Following for Everyone.**

[](https://opensource.org/licenses/Apache-2.0) [](https://www.google.com/search?q=) [](https://www.google.com/search?q=) [](https://arxiv.org/abs/2509.12129)

**OpenTrackVLA** is a fully open-source Vision-Language-Action (VLA) stack that turns **monocular video** and **natural-language instructions** into actionable, short-horizon waypoints.

While we explore massive backbones (8B/30B) internally, this repository is dedicated to democratizing embodied AI. We have intentionally released our highly efficient **0.6B checkpoint** along with the **full training pipeline**.

### 🚀 Why OpenTrackVLA?

  * **Fully Open Source:** We release the model weights, inference code, *and* the training stack—not just the inference wrapper.
  * **Accessible:** Designed to reproduce, fine-tune, and deploy with affordable compute .
  * **Multimodal Control:** Combines learned priors with visual input to guide real or simulated robots via simple text prompts.

> **Acknowledgment:** OpenTrackVLA builds on the ideas introduced by the original [TrackVLA project](https://github.com/wsakobe/TrackVLA). Their partially-open release inspired this community-driven effort to keep the ecosystem open so researchers and developers can continue improving the stack together.


## Demo In Action

The system processes video history and text instructions to predict future waypoints. Below are examples of the tracker in action:
<div align="center">
<img src="examples/ex1.gif" width="45%" alt="Tracked clip 1" />
<img src="examples/ex2.gif" width="45%" alt="Tracked clip 2" />
</div>


## 1. Requirements & Environment Setup

1. **Create the Conda env**
   ```bash
   conda create -n omtrack python=3.9 cmake=3.14.0
   conda activate omtrack
   ```
2. **Install Habitat-Sim 0.3.1 (with Bullet)**
   ```bash
   conda install habitat-sim==0.3.1 withbullet -c conda-forge -c aihabitat
   ```
3. **Clone this repository**
   ```bash
   git clone https://github.com/om-ai-lab/OpenTrackVLA.git
   cd OpenTrackVLA
   ```
4. **Install Habitat-Lab**
   ```bash
   pip install -e habitat-lab
   ```
---

## 2. Dataset Preparation (TrackVLA parity)

The simulator assets and humanoid avatars follow the exact instructions from the original TrackVLA github release. Please accept the respective licenses before downloading.

### 2.1 HM3D + MP3D Scenes
   - Request access through the Habitat links used by TrackVLA: [HM3D](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) and [MP3D](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset).
   - After the providers approve your request, download the archives locally and extract them under `data/scene_datasets` so the final layout matches:
     ```
     data/
       scene_datasets/
         hm3d/
           train/...
           val/...
           minival/...
         mp3d/
           1LXtFkjw3qL/...
           ...
     ```
   - If you stage the downloads elsewhere first, move them with a command such as:
     ```bash
     mv /path/to/hm3d data/scene_datasets/
     mv /path/to/mp3d data/scene_datasets/
     ```
     Ensure the directory names stay lowercase (`hm3d`, `mp3d`) so Habitat configs resolve correctly.

### 2.2 Humanoid Avatars
   - From the repository root run:
     ```bash
     python download_humanoid_data.py
     ```
     This script mirrors TrackVLA’s original helper and should populate `data/humanoids`.
   - If the script fails (e.g., quota errors), manually download `humanoids.zip` from [Google Drive](https://drive.google.com/file/d/1aE_wyvPqvOuVmF8px2vTO3trr70DKf1l/view), then extract it with:
     ```bash
     unzip humanoids.zip -d data/
     ```
     Double-check that `data/humanoids/**` now contains the FBX meshes required by Habitat.

### 2.3 Verification
   - Run `ls data/scene_datasets/{hm3d,mp3d}` to confirm both corpora are present.
   - Re-run any Habitat episodes (e.g., `run_eval.py`) to ensure the simulator can resolve the scene assets and humanoid avatars without missing-file errors.

Keeping this layout in sync with TrackVLA’s published instructions avoids surprises when sharing configs or checkpoints across both projects.

---

## 3. Data Processing Pipeline

### 3.1 Build Dataset (`make_tracking_data.py`)
This script integrates egocentric base velocities to emit future indicator curves, waypoints, and sliding-window JSONL shards. Each episode requires the `<stem>.mp4` RGB capture alongside `<stem>_info.json` pose metadata.

**Quick Start (Sample Data)**

The repo ships with a tiny rollout snapshot under `sim_data/sample`. Generate a ready-to-train dataset with:

```bash
python make_tracking_data.py \
    --input_root sim_data/sample \
    --output_root data/sample
```

This mirrors the `sim_data` tree under `data/sample/frames`, writes JSONL shards to `data/sample/jsonl`, and primes `data/sample/dataset.json` if `--out_file` is set.

**Full Usage**

```bash
python make_tracking_data.py \
    --input_root sim_data/sample \
    --output_root data/sample \
    --history 31 --horizon 8 --dt 0.1 \
    --only_success \
    --instruction "Follow the target person without collision."
```

Episodes are replicated under `frames/seed_xxx/...`, with corresponding `jsonl/seed_xxx/scene/episode.jsonl` files storing history frames (`images`), the current frame (`current`), the instruction, future waypoints (`trajectory`), and velocity commands (`actions`). Use `--out_file` to emit a monolithic dataset JSON in addition to per-episode shards.

### 3.2 Precompute Vision Tokens (`precache_frames.py`)
`train.py` stays I/O bound when every frame already has DINO/SiGLIP embeddings cached. `precache_frames.py` precomputes both fine and coarse tokens so the trainer can mmap `.pt` tensors instead of re-encoding on the fly.

```bash
python precache_frames.py \
    --data_root data/sample \
    --cache_root data/sample/vision_cache \
    --batch_size 8 \
    --image_size 384
```

Tokens are stored as `.half()` tensors to save disk, and any missing coarse tokens are backfilled via pooling when only the fine representation exists. Layout auto-detection handles nested `frames/seed_xxx/...` structures.

---

## 4. Training

`train.py` hosts the Qwen-based planner with masked waypoint losses. Point it at the dataset JSONL directory and the cached vision tokens to kick off optimization.

```bash
python train.py \
    --train_json data/sample/jsonl \
    --cache_root data/sample/vision_cache \
    --out_dir ckpt_sample \
    --epochs 2 \
    --batch_size 8 \
    --n_waypoints 8 \
    --history 31 \
    --lr 2e-5 \
    --mixed_precision \
    --save_trajectories
```

- **Checkpointing:** Every 100 steps a `.pt` file lands in `out_dir`; set `--max_ckpts` to cap retention.
- **Visualization:** Enable `--save_trajectories` to collect `.npz` bundles plus overlay images under `vis/`.
- **Resume:** Pass `--resume` or `--resume_ckpt /path/to/model_epochXX_stepXXXXXX.pt` to pick up training midstream.
- **Scaling knobs:** Adjust `--train_json` to target any directory of JSONL shards, tweak `--history` / `--n_waypoints` for different temporal windows, and flip `--distributed` when launching via `torchrun`.

---

## 5. Evaluation

`eval.sh` fans Habitat-based rollouts across configurable chunks and expects `CKPT`, `HF_MODEL_ID`, or `HF_MODEL_DIR` to define which weights to load. Outputs land under `sim_data/eval/<task>` alongside per-episode videos and metrics. Use `bash kill_eval.sh` to cleanly terminate all spawned jobs.

### 5.1 Using the Pre-trained 0.6B Model
- **Option A — Auto-download**
  ```bash
  HF_MODEL_ID=omlab/opentrackvla-qwen06b bash eval.sh
  ```
- **Option B — Manual download**
  ```bash
  huggingface-cli download omlab/opentrackvla-qwen06b --local-dir open_trackvla_hf
  HF_MODEL_DIR=$(pwd)/open_trackvla_hf bash eval.sh
  ```

### 5.2 Evaluating Custom Checkpoints

Point `CKPT` at any artifact produced by `train.py`:

```bash
CKPT=/path/to/model_epoch05_step009000.pt bash eval.sh
```

Tune `CHUNKS`, `NUM_PARALLEL`, or the Habitat config inside `eval.sh` to rebalance throughput vs. coverage.

## 📊 Performance (EVT-Bench)

| Methods      | STT (SR↑ / TR↑ / CR↓) | DT (SR↑ / TR↑ / CR↓) | AT (SR↑ / TR↑ / CR↓) |
|--------------|------------------------|----------------------|----------------------|
| IBVS†        | 42.9 / 56.2 / 3.75     | 10.6 / 28.4 / 6.14   | 15.2 / 39.5 / 4.90   |
| PoliFormer†  | 4.67 / 15.5 / 40.1     | 2.62 / 13.2 / 44.5   | 3.04 / 15.4 / 41.5   |
| EVT          | 24.4 / 39.1 / 42.5     | 3.23 / 11.2 / 47.9   | 17.4 / 21.1 / 45.6   |
| EVT‡         | 32.5 / 49.9 / 40.5     | 15.7 / 35.7 / 53.3   | 18.3 / 21.0 / 44.9   |
| Uni-NaVid (Vicuna-7B)    | 25.7 / 39.5 / 41.9     | 11.3 / 27.4 / 43.5   | 8.26 / 28.6 / 43.7   |
| TrackVLA (Vicuna-7B)     | 85.1 / 78.6 / 1.65     | 57.6 / 63.2 / 5.80   | 50.2 / 63.7 / 17.1   |
| Ours (Qwen-0.6B)  | 64.8 / 84.4 / 5.00     | 33.6 / 66.3 / 8.84   | 39.6 / 76.7 / 6.38   |

† Uses GroundingDINO as the open-vocabulary detector. ‡ Uses SoM + GPT-4o as the vision stack (see the TrackVLA paper Table 2).

**Transparency note:** Despite relying on a compact 0.6B backbone, OpenTrackVLA surpasses the 7B TrackVLA baseline on tracking-rate (TR↑) across all EVT-Bench settings. Success rate (SR↑) still trails the larger Vicuna-7B planner, and narrowing that gap without inflating compute remains an active focus. Larger 8B/30B research checkpoints exist internally, but we prioritize this lightweight release to keep reproduction and fine-tuning accessible.

## 📚 Resources & References
- Baseline checkpoint: [omlab/opentrackvla-qwen06b](https://huggingface.co/omlab/opentrackvla-qwen06b)
- LLM backbone: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- Vision towers: [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)
- TrackVLA: Embodied Visual Tracking in the Wild [arXiv:2505.23189](https://arxiv.org/abs/2505.23189)
- Embodied Navigation Foundation Model [arXiv:2509.12129](https://arxiv.org/abs/2509.12129)

## Citation

If you find OpenTrackVLA useful in your research or applications, please cite it using the following BibTeX:

```bibtex
@misc{opentrackvla2025,
  author       = {Kyusong Lee and Tiancheng Zhao},
  title        = {OpenTrackVLA: Open-Source Visual Language Action Model for Visual Navigation and Following},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {https://github.com/om-ai-lab/OpenTrackVLA}
}
```

**Happy tracking!** Contributions and issue reports are welcome via pull requests.
