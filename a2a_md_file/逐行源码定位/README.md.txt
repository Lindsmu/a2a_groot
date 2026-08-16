00001 | <div align="center">
00002 | 
00003 |   <img src="media/header_compress.png" width="800" alt="NVIDIA Isaac GR00T N1.7 Header">
00004 | 
00005 |   <!-- --- -->
00006 | 
00007 |   <p style="font-size: 1.2em;">
00008 |     <a href="https://developer.nvidia.com/isaac/gr00t"><strong>Website</strong></a> |
00009 |     <a href="https://huggingface.co/collections/nvidia/gr00t-n17"><strong>Model</strong></a> |
00010 |     <a href="https://huggingface.co/collections/nvidia/physical-ai"><strong>Datasets (Physical AI)</strong></a> |
00011 |     <a href="https://arxiv.org/abs/2503.14734"><strong>Paper</strong></a> |
00012 |     <a href="https://developer.nvidia.com/isaac"><strong>NVIDIA Isaac</strong></a> |
00013 |     <a href="FAQ.md"><strong>FAQ</strong></a>
00014 |   </p>
00015 | </div>
00016 | 
00017 | ## Table of Contents
00018 | 
00019 | - [NVIDIA Isaac GR00T](#nvidia-isaac-gr00t)
00020 | - [What's New in GR00T N1.7](#whats-new-in-gr00t-n17)
00021 | - [Installation](#installation)
00022 | - [LeRobot Integration](#lerobot-integration)
00023 | - [Model Checkpoints & Embodiment Tags](#model-checkpoints--embodiment-tags)
00024 | - [Data Format](#data-format)
00025 | - [Inference](#inference)
00026 | - [Fine-tuning](#fine-tuning)
00027 | - [Evaluation](#evaluation)
00028 | - [Contributions](#contributions)
00029 | - [License](#license)
00030 | - [Citation](#citation)
00031 | 
00032 | ---
00033 | 
00034 | ## NVIDIA Isaac GR00T
00035 | 
00036 | <table style="width:100%; table-layout:fixed;">
00037 |   <tr>
00038 |     <td style="width:33.33%; text-align:center;">
00039 |       <img src="media/unitree_g1.gif" style="max-width:100%; height:auto;">
00040 |     </td>
00041 |     <td style="width:33.33%; text-align:center;">
00042 |       <img src="media/agibot_g1.gif" style="max-width:100%; height:auto;">
00043 |     </td>
00044 |     <td style="width:33.33%; text-align:center;">
00045 |       <img src="media/yam.gif" style="max-width:100%; height:auto;">
00046 |     </td>
00047 |   </tr>
00048 | </table>
00049 | 
00050 | > We just released GR00T N1.7 General Availability, the latest version of GR00T N1 with a new VLM backbone (Cosmos-Reason2-2B / Qwen3-VL) and improved performance.
00051 | 
00052 | > **This is a General Availability (GA) release.** You are welcome to download the model, explore the codebase, and build on the stack, with full support and stability guarantees.
00053 | >
00054 | > **What's available:**
00055 | > - Pre-trained GR00T N1.7 model weights and reference code
00056 | > - Fine-tuning and inference with custom robot data or demonstrations
00057 | > - Experimentation, prototyping, and research use cases
00058 | > - Production deployment with commercial support
00059 | > - Complete benchmarks and a fully validated, stable feature set
00060 | > - Pull request contributions
00061 | >
00062 | > We welcome feedback - please feel free to raise issues and pull requests in this repository.
00063 | 
00064 | > Previous releases: [N1.6](https://github.com/NVIDIA/Isaac-GR00T/tree/n1d6) | [N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1d5)
00065 | 
00066 | NVIDIA Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills. This cross-embodiment model takes multimodal input, including language and images, to perform manipulation tasks in diverse environments.
00067 | 
00068 | GR00T N1.7 is trained on a diverse mixture of robot data including bimanual, semi-humanoid and an expansive humanoid dataset. It is adaptable through post-training for specific embodiments, tasks and environments.
00069 | 
00070 | GR00T N1.7 is fully commercially licensable under Apache 2.0. It delivers comparable performance to N1.6, with improved generalization and language-following capabilities driven by the inclusion of 20K hours of EgoScale human video data in pretraining.
00071 | 
00072 | The neural network architecture of GR00T N1.7 is a combination of vision-language foundation model and diffusion transformer head that denoises continuous actions. Here is a schematic diagram of the architecture:
00073 | 
00074 | <div align="center">
00075 | <img src="media/model-architecture.png" width="800" alt="model-architecture">
00076 | </div>
00077 | 
00078 | ### Workflow Overview
00079 | 
00080 | 1. **Prepare data** — Collect robot demonstrations (video, state, action) and convert them to the [GR00T LeRobot format](#data-format). Demo datasets are included for quick testing.
00081 | 2. **Run inference** — Try zero-shot inference with the base model on [pretrain embodiments](#embodiment-tags), or use a [finetuned checkpoint](#checkpoints) for benchmark tasks.
00082 | 3. **Fine-tune** — Adapt the model to your robot using [`launch_finetune.py`](#fine-tuning) with your own data and modality config.
00083 | 4. **Evaluate** — Validate with [open-loop evaluation](#open-loop-evaluation), then test in [simulation benchmarks](#benchmark-examples) or on real hardware via the [Policy API](getting_started/policy.md).
00084 | 5. **Deploy** — Connect `Gr00tPolicy` to your robot controller, optionally accelerated with [TensorRT](scripts/deployment/README.md).
00085 | 
00086 | ## What's New in GR00T N1.7
00087 | 
00088 | GR00T N1.7 builds on N1.6 with a new VLM backbone and code-level improvements.
00089 | 
00090 | 1. **Relative EEF Action Space** — N1.7 adopts a relative end-effector action space shared across robot and human embodiments. Representing actions as deltas from the current pose (rather than absolute targets) improves generalization and is a key factor in the model's cross-embodiment performance. See [`getting_started/finetune_new_embodiment.md`](getting_started/finetune_new_embodiment.md) for guidance on configuring relative EEF for your own robot.
00091 | 
00092 | 2. **Human Video Pretraining** — N1.7 is pretrained on 20K hours of EgoScale human video data alongside diverse robot demonstrations. Because the relative EEF action representation is consistent across both human and robot data, the model can transfer manipulation priors learned from human video directly to robot control.
00093 | 
00094 | ### Key Changes from N1.6
00095 | 
00096 | Compared with N1.6, N1.7 updates the model stack, training data interface,
00097 | evaluation coverage, deployment flow, fine-tuning workflow, and runtime behavior.
00098 | 
00099 | - **New VLM backbone:** Cosmos-Reason2-2B (Qwen3-VL architecture), replacing the Eagle backbone used in N1.6. Supports flexible resolution and encodes images in their native aspect ratio without padding.
00100 | - **Updated model interface:** N1.7 moves to the `gr00t_n1d7` model package, expands the state/action dimensions, and increases the model action horizon.
00101 | - **More flexible dataset handling:** Fine-tuning can use multiple dataset paths with mixture weighting, making multi-dataset training easier to configure.
00102 | - **Broader benchmark coverage:** N1.7 refreshes and expands documented results across RoboCasa, RoboCasa GR1 tabletop tasks, SimplerEnv, and real G1 evaluation.
00103 | - **More complete deployment path:** N1.7 adds full-pipeline ONNX and TensorRT export support and improves deployment consistency across desktop GPUs and edge platforms.
00104 | - **More predictable runtime behavior:** Policy serving, rollout recording, evaluation, and configuration validation have been hardened so errors are easier to diagnose.
00105 | 
00106 | <details>
00107 | <summary>Detailed changes from N1.6</summary>
00108 | 
00109 | These are the main code, model, training, evaluation, and deployment changes
00110 | that distinguish the current N1.7 main branch from the N1.6 / 1D6 code path.
00111 | Use the [`n1d6` branch](https://github.com/NVIDIA/Isaac-GR00T/tree/n1d6) when you need
00112 | the N1.6 model package and runtime behavior.
00113 | 
00114 | - Model package changed from `gr00t_n1d6` to `gr00t_n1d7`, so codepaths and processor metadata move to the N1.7 namespace.
00115 | - VLM backbone changed from vendored Eagle, `nvidia/Eagle-Block2A-2B-v2`, to `nvidia/Cosmos-Reason2-2B` via Qwen3-VL.
00116 | - Transformers changed from `4.51.3` to `4.57.3` to support the newer Qwen3-VL stack.
00117 | - Model defaults changed: `select_layer` `16` to `12`, `tune_top_llm_layers` `4` to `0`, and `load_bf16` `true` to `false`.
00118 | - State and action dimensions expanded from `29` to `132`, and `action_horizon` expanded from `16` to `40`.
00119 | - Action head remains flow-matching DiT, but changes from `32` to `16` diffusion layers and adds newer N1.7 behavior options.
00120 | - Dataset input handling now supports multiple dataset paths and `ds_weights_alpha` for dataset mixtures.
00121 | - The rollout CLI flag was renamed from `--action-horizon` to `--execution-horizon` to clarify how many predicted actions are executed per policy call.
00122 | - Server/client transport has stronger object-dtype ndarray serialization and cleaner socket timeout behavior.
00123 | 
00124 | </details>
00125 | 
00126 | ---
00127 | 
00128 | ## Installation
00129 | 
00130 | ### Hardware Requirements
00131 | 
00132 | **Inference:** 1 GPU with 16 GB+ VRAM (e.g., RTX 4090, L40, H100, Jetson AGX Thor/Orin, DGX Spark).
00133 | 
00134 | **Fine-tuning:** 1 or more GPUs with 40 GB+ VRAM recommended. We recommend H100 or L40 nodes for optimal performance. Other hardware (e.g., A6000) works but may require longer training time. See the [Hardware Recommendation Guide](getting_started/hardware_recommendation.md) for detailed specs.
00135 | 
00136 | **CUDA / Python per platform:** dGPU on CUDA 12.8 with Python 3.12; Jetson Orin on CUDA 12.6 with Python 3.10; Jetson Thor and DGX Spark on CUDA 13.0 with Python 3.12. The per-platform install scripts and Dockerfiles live under `scripts/deployment/`; see the [Deployment & Inference Guide](scripts/deployment/README.md) for the full matrix.
00137 | 
00138 | ### Clone the Repository
00139 | 
00140 | GR00T relies on submodules for certain dependencies. Include them when cloning:
00141 | 
00142 | **Note:** `git-lfs` is **required** to download parquet data files in `demo_data/`. Install it before cloning: `sudo apt install git-lfs && git lfs install`.
00143 | ```sh
00144 | git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
00145 | cd Isaac-GR00T
00146 | ```
00147 | 
00148 | If you've already cloned without submodules, initialize them separately:
00149 | 
00150 | ```sh
00151 | git submodule update --init --recursive
00152 | ```
00153 | 
00154 | ### Set Up the Environment
00155 | 
00156 | GR00T uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management. Install uv first:
00157 | 
00158 | ```sh
00159 | curl -LsSf https://astral.sh/uv/install.sh | sh
00160 | ```
00161 | 
00162 | #### dGPU (x86_64) — Default
00163 | 
00164 | Install FFmpeg (required by `torchcodec`, the only supported video backend):
00165 | ```sh
00166 | sudo apt-get update && sudo apt-get install -y ffmpeg
00167 | ```
00168 | > **FFmpeg version:** `torchcodec==0.8.0` supports **FFmpeg 4-7 only**. On Ubuntu 25.10+/26.04 the `ffmpeg` package is version 8, which `torchcodec` cannot load (`RuntimeError: Could not load libtorchcodec ... We support versions 4, 5, 6 and 7`). On those distros install an FFmpeg&lt;8 runtime instead, e.g. `conda install -c conda-forge 'ffmpeg<8'`, and make sure its libraries are on `LD_LIBRARY_PATH`.
00169 | 
00170 | Create the environment and install GR00T:
00171 | ```sh
00172 | uv sync --python 3.12
00173 | ```
00174 | GPU dependencies (flash-attn, TensorRT, etc.) are included in the default install.
00175 | 
00176 | Verify the installation:
00177 | ```sh
00178 | uv run python -c "import gr00t; print('GR00T installed successfully')"
00179 | ```
00180 | 
00181 | > **Hugging Face access (required):** GR00T's VLM backbone is [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B), a **gated** model that every GR00T checkpoint (including the base `nvidia/GR00T-N1.7-3B`) loads on first use. Before running inference or finetuning, request access on the model page and authenticate:
00182 | > ```sh
00183 | > uv run huggingface-cli login   # or: export HF_TOKEN=<your_token>
00184 | > ```
00185 | > Without access, model loading fails with a `GatedRepoError` / `401 Client Error`.
00186 | 
00187 | > **`flash-attn` message on every `uv run`:** You may see `Installing flash-attn...` each time you run `uv run`. This is a known `uv` behavior with URL-pinned wheel sources — `uv` re-validates the cached wheel against the source URL on each invocation. It is **not** rebuilding from source; the wheel is already cached locally and the operation takes 2-3 seconds. This affects platforms that use URL-pinned flash-attn wheels (x86_64 and aarch64). 
00188 | > To suppress it, remove the `flash-attn` entries under `[tool.uv.sources]` in your local `pyproject.toml` after the initial install. But that will break `uv lock` and cause flash-attn to build from source on next lock regeneration.
00189 | 
00190 | <details>
00191 | <summary><strong>Alternative: pip install (without uv)</strong></summary>
00192 | 
00193 | If you prefer pip/conda over uv, create a Python 3.12 virtualenv and install:
00194 | ```sh
00195 | python3.12 -m venv .venv && source .venv/bin/activate
00196 | pip install -e .
00197 | ```
00198 | Note: GPU dependencies (flash-attn, TensorRT) may require manual installation with pip. The `uv` workflow handles these automatically.
00199 | </details>
00200 | 
00201 | > **If fine-tuning fails with `CUDA_HOME is unset`:** Run `bash scripts/deployment/dgpu/install_deps.sh` once to configure CUDA paths, or manually `export CUDA_HOME=/usr/local/cuda`.
00202 | 
00203 | > **CUDA 13.x Users (Thor, Spark, and other CUDA 13+ platforms):** PyTorch 2.7 pins Triton to 3.3.1, which does not recognize CUDA major version 13+. This causes a `RuntimeError` in Triton's `ptx_get_version()`. Run `scripts/patch_triton_cuda13.sh` to fix:
00204 | > ```sh
00205 | > uv run bash scripts/patch_triton_cuda13.sh
00206 | > ```
00207 | 
00208 | > **GB300 (sm_103) Users:** Triton 3.3.1 (pinned by PyTorch 2.7) does not support the GB300 GPU architecture (sm_103). `torch.compile` will fail on GB300. Use PyTorch eager mode or TensorRT inference instead. Triton 3.5.1+ adds sm_103 support but is not yet compatible with the pinned PyTorch version.
00209 | 
00210 | > **Video Backend:** GR00T uses [`torchcodec`](https://github.com/pytorch/torchcodec) as its sole video decoding backend. Backends such as `decord` and `pyav` are no longer supported. `torchcodec` 0.8.0 requires **FFmpeg 4-7** (FFmpeg 8 is not supported — see the FFmpeg version note above) and supports H.264 on all platforms; AV1 decoding is not guaranteed (convert AV1 datasets to H.264 with `examples/SimplerEnv/convert_av1_to_h264.py`). On aarch64 platforms (Thor, Orin), `torchcodec` is built from source during `install_deps.sh` because pre-built wheels are not available — if you encounter a `NotImplementedError`, ensure the build completed successfully.
00211 | 
00212 | <details>
00213 | <summary><strong>DGX Spark</strong> (tested with DGX Spark GB10)</summary>
00214 | 
00215 | ```bash
00216 | bash scripts/deployment/spark/install_deps.sh
00217 | source .venv/bin/activate
00218 | source scripts/activate_spark.sh
00219 | ```
00220 | 
00221 | See the [Spark setup guide](scripts/deployment/README.md#dgx-spark-setup) for Docker and bare metal details.
00222 | </details>
00223 | 
00224 | <details>
00225 | <summary><strong>Jetson AGX Thor</strong> (tested with JetPack 7.1)</summary>
00226 | 
00227 | > **flash-attn on older systems (e.g., Ubuntu 20.04 with glibc < 2.35):** The pre-built `flash-attn` wheel may fail with `ImportError: glibc_compat.so: cannot open shared object file`. To fix this, build from source:
00228 | > ```sh
00229 | > uv pip install flash-attn==2.7.4.post1 --no-binary flash-attn --no-cache
00230 | > ```
00231 | > This compiles locally (~10-30 minutes) and avoids the glibc compatibility issue.
00232 | 
00233 | ```bash
00234 | bash scripts/deployment/thor/install_deps.sh
00235 | source .venv/bin/activate
00236 | source scripts/activate_thor.sh
00237 | ```
00238 | 
00239 | See the [Thor setup guide](scripts/deployment/README.md#jetson-thor-setup) for Docker and bare metal details.
00240 | </details>
00241 | 
00242 | <details>
00243 | <summary><strong>Jetson Orin</strong> (tested with JetPack 6.2)</summary>
00244 | 
00245 | ```bash
00246 | bash scripts/deployment/orin/install_deps.sh
00247 | source .venv/bin/activate
00248 | source scripts/activate_orin.sh
00249 | ```
00250 | 
00251 | See the [Orin setup guide](scripts/deployment/README.md#jetson-orin-setup) for Docker and bare metal details.
00252 | </details>
00253 | 
00254 | > ⚠️ **aarch64 users (Spark / Thor / Orin):** After running `install_deps.sh`, always
00255 | > activate the venv with `source .venv/bin/activate && source scripts/activate_<platform>.sh`
00256 | > (`activate_spark.sh`, `activate_thor.sh`, or `activate_orin.sh`) and run the example
00257 | > commands in this guide with **plain `python`** / `torchrun`, not `uv run python` /
00258 | > `uv run torchrun`. The latter will re-sync against the root `pyproject.toml` (which targets
00259 | > x86_64 Python 3.12) and destroy the platform-specific environment. See the
00260 | > [Deployment & Inference Guide](scripts/deployment/README.md#platform-specific-setup) for
00261 | > per-platform Docker and bare-metal setup.
00262 | 
00263 | 
00264 | For a containerized setup that avoids system-level dependency conflicts, see our [Docker Setup Guide](docker/README.md). The recommended container workflow is to start the image first, then clone or pull the repo inside the running container so your checkout uses the image's prebuilt dependency environment.
00265 | 
00266 | ---
00267 | 
00268 | ## LeRobot Integration
00269 | 
00270 | GR00T N1.7 is also available through Hugging Face LeRobot via the `groot` policy type. Use the [LeRobot GR00T documentation](https://github.com/huggingface/lerobot/blob/main/docs/source/groot.mdx) for LeRobot-native training, evaluation, and rollout workflows. Use this repository for the reference GR00T implementation, model internals, deployment tooling, and benchmark-specific examples.
00271 | 
00272 | ---
00273 | 
00274 | ## Model Checkpoints & Embodiment Tags
00275 | 
00276 | ### Checkpoints
00277 | 
00278 | | Checkpoint | Type | Embodiment Tag | Description |
00279 | |------------|------|---------------|-------------|
00280 | | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) | Base | See [pretrain tags](getting_started/policy.md#--embodiment-tag) | Base model (3B params) — zero-shot inference on pretrain embodiments, or finetune for new tasks |
00281 | | [`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) | Finetuned | `LIBERO_PANDA` | Finetuned on [LIBERO](https://libero-project.github.io/) benchmark (Franka Panda) |
00282 | | [`nvidia/GR00T-N1.7-DROID`](https://huggingface.co/nvidia/GR00T-N1.7-DROID) | Finetuned | `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` | Finetuned on [DROID](https://droid-dataset.github.io/) dataset |
00283 | | [`nvidia/GR00T-N1.7-SimplerEnv-Bridge`](https://huggingface.co/nvidia/GR00T-N1.7-SimplerEnv-Bridge) | Finetuned | `SIMPLER_ENV_WIDOWX` | Finetuned on SimplerEnv Bridge (WidowX) |
00284 | | [`nvidia/GR00T-N1.7-SimplerEnv-Fractal`](https://huggingface.co/nvidia/GR00T-N1.7-SimplerEnv-Fractal) | Finetuned | `SIMPLER_ENV_GOOGLE` | Finetuned on SimplerEnv Fractal (Google Robot) |
00285 | 
00286 | ### Embodiment Tags
00287 | 
00288 | Every inference or finetuning command requires an `--embodiment-tag`. The tag determines which modality config (state/action keys, normalization) the model uses. Tags are **case-insensitive**.
00289 | 
00290 | For the full list of pretrain and posttrain tags, see the [Policy API Guide — Embodiment Tags](getting_started/policy.md#--embodiment-tag).
00291 | 
00292 | ---
00293 | 
00294 | ## Data Format
00295 | 
00296 | GR00T uses a flavor of the [LeRobot v2 dataset format](https://github.com/huggingface/lerobot) with an additional `meta/modality.json` file that describes state/action/video structure. A dataset looks like:
00297 | 
00298 | ```
00299 | my_dataset/
00300 |   meta/
00301 |     info.json            # dataset metadata
00302 |     episodes.jsonl       # episode index and lengths
00303 |     tasks.jsonl          # language task descriptions
00304 |     modality.json        # state/action/video key mapping (GR00T-specific)
00305 |   data/chunk-000/        # parquet files (state, action per timestep)
00306 |   videos/chunk-000/      # mp4 video files per episode
00307 | ```
00308 | 
00309 | The `modality.json` maps how the concatenated state/action arrays split into named fields (e.g., `x`, `y`, `z`, `gripper`) and which video keys are available. This is what the embodiment tag uses to interpret the data.
00310 | 
00311 | **Included demo datasets** (ready to use, no download needed):
00312 | 
00313 | | Dataset | Robot | Embodiment Tag | Use Case |
00314 | |---------|-------|---------------|----------|
00315 | | `demo_data/droid_sample` | DROID (3 episodes) | `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` | Zero-shot or finetuned inference (DROID) |
00316 | | `demo_data/libero_demo` | LIBERO Panda (5 episodes) | `LIBERO_PANDA` | Inference with finetuned checkpoint |
00317 | | `demo_data/simplerenv_bridge_sample` | WidowX (SimplerEnv Bridge) | `SIMPLER_ENV_WIDOWX` | Inference with finetuned SimplerEnv Bridge checkpoint |
00318 | | `demo_data/simplerenv_fractal_sample` | Google Robot (SimplerEnv Fractal) | `SIMPLER_ENV_GOOGLE` | Inference with finetuned SimplerEnv Fractal checkpoint |
00319 | | `demo_data/cube_to_bowl_5` | SO100 arm (5 episodes) | `NEW_EMBODIMENT` | Fine-tuning custom embodiment example |
00320 | | `demo_data/cube_to_bowl_5_with_mask` | SO100 arm + per-frame masks | `NEW_EMBODIMENT` | [Mask-guided background suppression](examples/mask-guided-background-suppression/README.md) example |
00321 | 
00322 | > To generate more DROID episodes: `python scripts/download_droid_sample.py --num-episodes 10`
00323 | 
00324 | **Using your own data:** Convert your demonstrations to the format above. If coming from LeRobot v3, use the conversion helper in its own environment:
00325 | ```bash
00326 | cd scripts/lerobot_conversion
00327 | uv venv
00328 | source .venv/bin/activate
00329 | uv pip install -e . --verbose
00330 | python convert_v3_to_v2.py --repo-id <DATASET_REPO_ID>
00331 | ```
00332 | See the full [Data Preparation Guide](getting_started/data_preparation.md) for schema details and examples.
00333 | 
00334 | ---
00335 | 
00336 | ## Inference
00337 | 
00338 | > **Prefer an interactive walkthrough?** The [`getting_started/GR00T_inference.ipynb`](getting_started/GR00T_inference.ipynb) notebook steps through loading the model and predicting actions from observations on a sample dataset.
00339 | 
00340 | ### Zero-Shot Inference (Base Model)
00341 | 
00342 | The included `demo_data/droid_sample` dataset works with the base model out of the box — no finetuning or checkpoint download needed:
00343 | 
00344 | ```bash
00345 | uv run python scripts/deployment/standalone_inference_script.py \
00346 |     --model-path nvidia/GR00T-N1.7-3B \
00347 |     --dataset-path demo_data/droid_sample \
00348 |     --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
00349 |     --traj-ids 1 2 \
00350 |     --inference-mode pytorch \
00351 |     --execution-horizon 8
00352 | ```
00353 | 
00354 | This runs open-loop inference on 2 DROID episodes, comparing predicted actions against ground truth. The base model downloads automatically from HuggingFace on first run (~6 GB).
00355 | 
00356 | > **Note:** The base model loads the gated `nvidia/Cosmos-Reason2-2B` backbone, so this command requires Hugging Face access (see [Set Up the Environment](#set-up-the-environment)). Without it the run fails with a `GatedRepoError`.
00357 | 
00358 | ### Finetuned Inference
00359 | 
00360 | For posttrain embodiments, use a finetuned checkpoint. Most finetuned checkpoints (e.g., DROID, SimplerEnv) have a flat file structure and can be passed directly as a HuggingFace model ID — no manual download needed:
00361 | 
00362 | ```bash
00363 | uv run python scripts/deployment/standalone_inference_script.py \
00364 |     --model-path nvidia/GR00T-N1.7-DROID \
00365 |     --dataset-path demo_data/droid_sample \
00366 |     --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
00367 |     --traj-ids 1 2 \
00368 |     --inference-mode pytorch \
00369 |     --execution-horizon 8
00370 | ```
00371 | 
00372 | Some checkpoints (e.g., LIBERO) use a nested folder structure with model files under a subfolder. HuggingFace does not support nested repo paths in `--model-path`, so you must download first:
00373 | 
00374 | ```bash
00375 | uv run hf download nvidia/GR00T-N1.7-LIBERO \
00376 |     --include "libero_10/config.json" "libero_10/embodiment_id.json" \
00377 |     "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" \
00378 |     "libero_10/processor_config.json" "libero_10/statistics.json" \
00379 |     --local-dir checkpoints/GR00T-N1.7-LIBERO
00380 | ```
00381 | 
00382 | ```bash
00383 | uv run python scripts/deployment/standalone_inference_script.py \
00384 |     --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
00385 |     --dataset-path demo_data/libero_demo \
00386 |     --embodiment-tag LIBERO_PANDA \
00387 |     --traj-ids 0 1 2 \
00388 |     --inference-mode pytorch \
00389 |     --execution-horizon 8
00390 | ```
00391 | 
00392 | ### Server-Client Inference (for Deployment)
00393 | 
00394 | For real-world deployment or simulation evaluation, use the server-client architecture. The policy runs on a GPU server; a lightweight client sends observations and receives actions over ZMQ.
00395 | 
00396 | **Terminal 1 — Start the policy server:**
00397 | ```bash
00398 | uv run python gr00t/eval/run_gr00t_server.py \
00399 |     --model-path nvidia/GR00T-N1.7-3B \
00400 |     --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
00401 |     --device cuda:0
00402 | ```
00403 | 
00404 | **Terminal 2 — Run open-loop evaluation as a client:**
00405 | ```bash
00406 | uv run python gr00t/eval/open_loop_eval.py \
00407 |     --dataset-path demo_data/droid_sample \
00408 |     --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
00409 |     --host 127.0.0.1 \
00410 |     --port 5555 \
00411 |     --traj-ids 1 2 \
00412 |     --execution-horizon 8
00413 | ```
00414 | 
00415 | > **Tip:** If you get `ZMQError: Address already in use`, the default port 5555 is occupied. Use `--port <other_port>`.
00416 | 
00417 | For connecting to a real robot (e.g., DROID hardware), see [examples/DROID/README.md](examples/DROID/README.md). For faster inference with TensorRT, see the [Deployment & Inference Guide](scripts/deployment/README.md).
00418 | 
00419 | See the complete [Policy API Guide](getting_started/policy.md) for documentation on observation/action formats, batched inference, and troubleshooting.
00420 | 
00421 | ---
00422 | 
00423 | ## Fine-tuning
00424 | 
00425 | ### Reproducing Benchmark Results
00426 | 
00427 | Each benchmark has a self-contained README with dataset download, finetune, and evaluation commands:
00428 | 
00429 | | Benchmark | Embodiment | Guide |
00430 | |-----------|-----------|-------|
00431 | | LIBERO | `LIBERO_PANDA` | [examples/LIBERO/README.md](examples/LIBERO/README.md) |
00432 | | SimplerEnv (Fractal) | `SIMPLER_ENV_GOOGLE` | [examples/SimplerEnv/README.md](examples/SimplerEnv/README.md) |
00433 | | SimplerEnv (Bridge) | `SIMPLER_ENV_WIDOWX` | [examples/SimplerEnv/README.md](examples/SimplerEnv/README.md) |
00434 | | SO100 | `NEW_EMBODIMENT` | [examples/SO100/README.md](examples/SO100/README.md) |
00435 | 
00436 | For the optional 512-D latent Action-to-Action Flow Matching model variant,
00437 | including its strict canonical data contract and the reason raw LIBERO
00438 | state/action columns cannot be used directly, see
00439 | [`getting_started/latent_a2a.md`](getting_started/latent_a2a.md). The complete
00440 | file-by-file implementation and operating reference is available in
00441 | [`getting_started/潜空间A2A完整实现与使用手册.md`](getting_started/潜空间A2A完整实现与使用手册.md).
00442 | The optional temporal-token DiT comparison backend is documented in
00443 | [`getting_started/潜空间A2A_DiT对比实现与使用.md`](getting_started/潜空间A2A_DiT对比实现与使用.md).
00444 | 
00445 | ### Humanoid Whole-Body Control (SONIC)
00446 | 
00447 | GR00T N1.7 supports whole-body humanoid control via the `UNITREE_G1_SONIC` embodiment tag and the [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) controller. In this workflow, the VLA predicts compact latent action tokens that a learned whole-body controller decodes into full-body joint commands — including legs, arms, and hands. A single policy produces language-conditioned, coordinated manipulation and locomotion end-to-end. SONIC supports whole-body coordination with precise hand and foot placements.
00448 | 
00449 | The complete collect → finetune → deploy workflow is documented in the [GR00T-WholeBodyControl repository](https://github.com/NVlabs/GR00T-WholeBodyControl):
00450 | 
00451 | - [Data collection](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html) — VR teleoperation with SONIC for demonstration recording
00452 | - [VLA Workflow](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html) — finetuning Isaac-GR00T N1.7 on collected data and deploying the policy
00453 | - [VLA Inference](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html) — running the PolicyServer + SONIC decoder for real-time control
00454 | 
00455 | > **Note:** The `UNITREE_G1` embodiment tag is compatible with the [decoupled WBC](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/decoupled_wbc) controller, but the end-to-end collect-finetune-deploy workflow is only supported for GEAR-SONIC (`UNITREE_G1_SONIC`).
00456 | 
00457 | ### Fine-tune on Your Own Robot ("NEW_EMBODIMENT")
00458 | 
00459 | To finetune GR00T on your own robot data and configuration, follow the detailed tutorial at [`getting_started/finetune_new_embodiment.md`](getting_started/finetune_new_embodiment.md).
00460 | 
00461 | Ensure your input data follows the [GR00T LeRobot format](#data-format), and specify your modality configuration via `--modality-config-path`.
00462 | 
00463 | **Single GPU:**
00464 | ```bash
00465 | CUDA_VISIBLE_DEVICES=0 uv run python \
00466 |     gr00t/experiment/launch_finetune.py \
00467 |     --base-model-path nvidia/GR00T-N1.7-3B \
00468 |     --dataset-path demo_data/cube_to_bowl_5 \
00469 |     --embodiment-tag NEW_EMBODIMENT \
00470 |     --modality-config-path examples/SO100/so100_config.py \
00471 |     --num-gpus 1 \
00472 |     --output-dir /tmp/test_finetune \
00473 |     --max-steps 2000 \
00474 |     --global-batch-size 32 \
00475 |     --dataloader-num-workers 4
00476 | ```
00477 | 
00478 | **Multi-GPU (e.g., 8xH100):**
00479 | ```bash
00480 | uv run torchrun --nproc_per_node=8 --master_port=29500 \
00481 |     gr00t/experiment/launch_finetune.py \
00482 |     --base-model-path nvidia/GR00T-N1.7-3B \
00483 |     --dataset-path demo_data/cube_to_bowl_5 \
00484 |     --embodiment-tag NEW_EMBODIMENT \
00485 |     --modality-config-path examples/SO100/so100_config.py \
00486 |     --num-gpus 8 \
00487 |     --output-dir /tmp/test_finetune_8gpu \
00488 |     --max-steps 2000 \
00489 |     --global-batch-size 32 \
00490 |     --dataloader-num-workers 4
00491 | ```
00492 | 
00493 | Replace `demo_data/cube_to_bowl_5` and `examples/SO100/so100_config.py` with your own dataset and modality config. See [`examples/SO100`](examples/SO100/README.md) for a complete walkthrough.
00494 | 
00495 | > **Note:** Use `uv run torchrun` (not bare `torchrun`) to ensure the correct virtual environment is used. Add `--use-wandb` to enable Weights & Biases logging. For more extensive configuration, use `gr00t/experiment/launch_train.py`.
00496 | 
00497 | ### Training Tips
00498 | 
00499 | - Maximize batch size for your hardware and train for a few thousand steps.
00500 | - Users may observe 5-6% variance between runs due to non-deterministic image augmentations. Keep this in mind when comparing to reported benchmarks.
00501 | - **`--state_dropout_prob`** (model config default: 0.8; finetune CLI default: 0.2; see `gr00t/configs/finetune_config.py`): Randomly drops state inputs during training to improve generalization and reduce state-dependency. The shipped benchmark scripts override the CLI default per suite: LIBERO 10-Long uses 0.2 (the CLI default), SimplerEnv Bridge uses 0.8, SimplerEnv Fractal uses 0.5. If your task relies heavily on proprioceptive state, lower this value.
00502 | 
00503 | ---
00504 | 
00505 | ## Evaluation
00506 | 
00507 | ### Open-Loop Evaluation
00508 | 
00509 | Compare predicted actions against ground truth from your dataset:
00510 | 
00511 | ```bash
00512 | uv run python gr00t/eval/open_loop_eval.py \
00513 |     --dataset-path <DATASET_PATH> \
00514 |     --embodiment-tag NEW_EMBODIMENT \
00515 |     --model-path <CHECKPOINT_PATH> \
00516 |     --traj-ids 0 \
00517 |     --execution-horizon 16
00518 | ```
00519 | 
00520 | This generates a visualization at `/tmp/open_loop_eval/traj_{traj_id}.jpeg` with ground truth vs. predicted actions and MSE metrics. Use `--save-plot-path <dir>` to save plots to a custom location.
00521 | 
00522 | ### Closed-Loop Evaluation
00523 | 
00524 | Test your model in simulation or on real hardware using the server-client architecture:
00525 | 
00526 | ```bash
00527 | # Start the policy server
00528 | uv run python gr00t/eval/run_gr00t_server.py \
00529 |     --embodiment-tag NEW_EMBODIMENT \
00530 |     --model-path <CHECKPOINT_PATH> \
00531 |     --device cuda:0 \
00532 |     --host 0.0.0.0 --port 5555
00533 | ```
00534 | 
00535 | ```python
00536 | from gr00t.policy.server_client import PolicyClient
00537 | 
00538 | policy = PolicyClient(host="localhost", port=5555)
00539 | env = YourEnvironment()
00540 | obs, info = env.reset()
00541 | action, info = policy.get_action(obs)
00542 | obs, reward, done, truncated, info = env.step(action)
00543 | ```
00544 | 
00545 | **Debugging with ReplayPolicy:** To verify your environment setup without a trained model, start the server with `--dataset-path <DATASET_PATH>` (omit `--model-path`) to replay recorded actions from the dataset.
00546 | 
00547 | See the complete [Policy API Guide](getting_started/policy.md) for observation/action formats, batched inference, and troubleshooting.
00548 | 
00549 | ### Benchmark Examples
00550 | 
00551 | We support evaluation on public benchmarks using a server-client architecture. The policy server reuses the project root's uv environment; simulation clients have individual setup scripts.
00552 | 
00553 | You can use [the verification script](scripts/eval/check_sim_eval_ready.py) to verify that all dependencies are properly configured.
00554 | 
00555 | #### One-Time Simulation Environment Setup
00556 | 
00557 | Each simulation benchmark needs a one-time environment setup before its first run. First install the shared system libraries:
00558 | 
00559 | ```bash
00560 | sudo apt update
00561 | sudo apt install libegl1-mesa-dev libglu1-mesa
00562 | ```
00563 | 
00564 | Then run the benchmark's own `setup_*.sh` script, linked from each simulation benchmark's README (LIBERO, SimplerEnv, robocasa, and robocasa-gr1). This only needs to run once per benchmark; afterward you just launch the server and client. The real-hardware/custom-embodiment workflows (DROID, RoboLab, SO100) have no simulation setup script; follow their own READMEs instead.
00565 | 
00566 | **Zero-shot** (evaluate with the base model, no finetuning):
00567 | - [DROID](examples/DROID/README.md) — real-world DROID robot (also available as the finetuned `nvidia/GR00T-N1.7-DROID` checkpoint; `examples/DROID/README.md` covers both paths)
00568 | 
00569 | **Finetuned** (evaluate with finetuned checkpoints):
00570 | - [DROID](examples/DROID/README.md) — real-world DROID robot via `nvidia/GR00T-N1.7-DROID`
00571 | - [RoboLab](examples/RoboLab/README.md) — RoboLab simulation tasks via `nvidia/GR00T-N1.7-DROID`
00572 | - [LIBERO](examples/LIBERO/README.md) — LIBERO benchmark (Franka Panda)
00573 | - [SimplerEnv](examples/SimplerEnv/README.md) — Google Robot (Fractal) and WidowX (Bridge)
00574 | - [SO100](examples/SO100/README.md) — SO100 custom embodiment workflow
00575 | 
00576 | <details>
00577 | <summary><strong>Adding a New Sim Benchmark</strong></summary>
00578 | 
00579 | Each sim benchmark registers its environments under a gym env_name with the format `{prefix}/{task_name}` (e.g., `libero_sim/LIVING_ROOM_SCENE2_put_soup_in_basket`). The evaluation framework uses the prefix to look up the corresponding `EmbodimentTag` via a mapping in [`gr00t/eval/sim/env_utils.py`](gr00t/eval/sim/env_utils.py).
00580 | 
00581 | > **Important:** The env_name prefix and the `EmbodimentTag` **name** are often different. For example, the prefix `libero_sim` maps to `EmbodimentTag.LIBERO_PANDA` (whose value happens to be `"libero_sim"`). Do not assume the prefix matches the tag name.
00582 | 
00583 | To add a new benchmark:
00584 | 
00585 | 1. Add an entry to `ENV_PREFIX_TO_EMBODIMENT_TAG` in `gr00t/eval/sim/env_utils.py`:
00586 |    ```python
00587 |    ENV_PREFIX_TO_EMBODIMENT_TAG = {
00588 |        ...
00589 |        "my_new_benchmark": EmbodimentTag.MY_ROBOT,
00590 |    }
00591 |    ```
00592 | 2. If the benchmark has multiple env_name prefixes (e.g., `my_benchmark_v1`, `my_benchmark_v2`), all related prefixes **must** map to the same `EmbodimentTag`.
00593 | 3. Add corresponding test cases in `tests/gr00t/eval/sim/test_env_utils.py` and update the `test_all_known_prefixes_present` test.
00594 | </details>
00595 | 
00596 | 
00597 | 
00598 | ## Running Tests
00599 | 
00600 | Install the development dependencies before running the test suite:
00601 | ```bash
00602 | uv sync --python 3.12 --extra dev
00603 | uv run python -m pytest
00604 | ```
00605 | 
00606 | Use targeted test paths for faster local checks, and reserve GPU-marked tests for machines with the required CUDA hardware.
00607 | 
00608 | ---
00609 | 
00610 | ## Contributions
00611 | 
00612 | We welcome issues and pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute and for support details now that GR00T N1.7 has reached General Availability (GA).
00613 | 
00614 | ## License
00615 | 
00616 | - **Code:** Apache 2.0 — see [LICENSE](LICENSE)
00617 | - **Model weights:** [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
00618 | 
00619 | ```
00620 | # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
00621 | # SPDX-License-Identifier: Apache-2.0
00622 | #
00623 | # Licensed under the Apache License, Version 2.0 (the "License");
00624 | # you may not use this file except in compliance with the License.
00625 | # You may obtain a copy of the License at
00626 | #
00627 | # http://www.apache.org/licenses/LICENSE-2.0
00628 | #
00629 | # Unless required by applicable law or agreed to in writing, software
00630 | # distributed under the License is distributed on an "AS IS" BASIS,
00631 | # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
00632 | # See the License for the specific language governing permissions and
00633 | # limitations under the License.
00634 | ```
00635 | 
00636 | 
00637 | ## Citation
00638 | 
00639 | [Paper Site](https://research.nvidia.com/labs/lpr/publication/gr00tn1_2025/)
00640 | ```bibtex
00641 | @inproceedings{gr00tn1_2025,
00642 |   archivePrefix = {arxiv},
00643 |   eprint     = {2503.14734},
00644 |   title      = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
00645 |   author     = {NVIDIA and Johan Bjorck and Fernando Castañeda, Nikita Cherniadev and Xingye Da and Runyu Ding and Linxi "Jim" Fan and Yu Fang and Dieter Fox and Fengyuan Hu and Spencer Huang and Joel Jang and Zhenyu Jiang and Jan Kautz and Kaushil Kundalia and Lawrence Lao and Zhiqi Li and Zongyu Lin and Kevin Lin and Guilin Liu and Edith Llontop and Loic Magne and Ajay Mandlekar and Avnish Narayan and Soroush Nasiriany and Scott Reed and You Liang Tan and Guanzhi Wang and Zu Wang and Jing Wang and Qi Wang and Jiannan Xiang and Yuqi Xie and Yinzhen Xu and Zhenjia Xu and Seonghyeon Ye and Zhiding Yu and Ao Zhang and Hao Zhang and Yizhou Zhao and Ruijie Zheng and Yuke Zhu},
00646 |   month      = {March},
00647 |   year       = {2025},
00648 |   booktitle  = {ArXiv Preprint},
00649 | }
00650 | ```
