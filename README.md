<div align="center">

<img src="pics/mascot2.png" width="80" height="80" alt="mascot" align="left">

<h1>AURA: Always-On Understanding and Real-Time Assistance via Video Streams</h1>

<a href="https://aurateam2026.github.io/" target="_blank"><img alt="Home Page" src="https://img.shields.io/badge/Home-Page-0A66C2?logo=googlechrome&logoColor=white" height="22px"></a>
<a href="https://huggingface.co/aurateam/AURA/" target="_blank"><img src="https://img.shields.io/badge/Model-Hugging_Face-FFD21E?logo=huggingface&logoColor=black" height="22px"></a>
<a href="https://arxiv.org/abs/2604.04184" target="_blank"><img src="https://img.shields.io/badge/Technical_Report-arXiv-EC1C24?logo=arxiv&logoColor=white" height="22px"></a>

</div>

<p>A real-time multimodal streaming system powered by <a href="https://huggingface.co/aurateam/AURA/">our AURA model</a>, supporting continuous video understanding with speech interaction.</p>


## Demo Deployment and Usage

### Highlights

- **Real-Time Streaming**: Continuously processes live video at 2 FPS with sub-second response latency
- **Full Pipeline**: Integrated ASR, Vision-Language Model, and Streaming TTS, all running locally
- **Context Management**: Sliding-window history with automatic pruning and prefix KV cache reuse for bounded latency
- **One-Click Launch**: Single script (`start_all.sh`) to start all services with automatic GPU allocation


### Demo Videos

| Demo | Video |
|------|-------|
| *The pens on the desk* | [The pens on the desk.mp4](https://1drv.ms/v/c/7be8ecfc440137f7/IQBNwUIKXpxURIa1XcuP_KTvARn9UXmd7lclOyZsJan39os?e=QTyEIM) |
| *Watch the kettle for me* | [Watch the kettle for me.mp4](https://1drv.ms/v/c/7be8ecfc440137f7/IQAhAe9bOh_eToP02gnvbOFJAVYCb6MkJTozOamQhJYupwQ?e=tpHejG) |
| *史迪仔放在哪？* | [史迪仔放在哪？.mp4](https://1drv.ms/v/c/7be8ecfc440137f7/IQCYhlYg5oG1TZnZnVufE7FkATfPMzfAbrpW69HliohXPWk?e=MMnAos) |
| *小馋猫偷吃冻干* | [小馋猫偷吃冻干.mp4](https://1drv.ms/v/c/7be8ecfc440137f7/IQAkcpo66JgpSLjaIFrIk6InAWgIYIoQuqEVckk_kFqozi0?e=5dnnba) |
| *工作时不准摸鱼* | [工作时不准摸鱼.mp4](https://1drv.ms/v/c/7be8ecfc440137f7/IQBcGeeawTQ7TIlPM8B1DUtFAT3HA4U_JUuAY-0w4xMI4io?e=mcjk4I) |
| *帮我找找鼠标* | [帮我找找鼠标.mp4](https://1drv.ms/v/c/7be8ecfc440137f7/IQCbfvFX05k-Top9BLsQL-qAAQbW2vgExyML8hQnn0EI6E8?e=rt19XP) |
| *帮我盯着烧水壶* | [帮我盯着烧水壶.mp4](https://1drv.ms/v/c/7be8ecfc440137f7/IQCFpfzKJl_DQJG3gbaPnPEfAaNhWj0RThbmuoTh-y6X5M8?e=dU91oM) |
| *我刚才关灯了吗* | [我刚才关灯了吗.mp4](https://1drv.ms/v/c/7be8ecfc440137f7/IQCjfTRY1KwkRJ0Xzo9J-RlbAYMu9qaNfpPt3N_tzfri-XA?e=0NRQWo) |
| *桌面上的笔* | [桌面上的笔.mp4](https://1drv.ms/v/c/7be8ecfc440137f7/IQCNAwcGx6iQT5qIOXBy_gYOATismEjWA0nHG6djk4-kvkU?e=ZeYJYN) |
| *绿灯才能过马路哦* | [绿灯才能过马路哦.mp4](https://1drv.ms/v/c/7be8ecfc440137f7/IQAJInfVpuT2ToNqGbJ8iY87AY8SUyORWpN20ws6qYMlI5I) |

> Click a link above to download and watch the demo video. For more demo videos, please visit our [Home Page](https://aurateam2026.github.io/).

### Requirements

| Category | Requirement |
|----------|-------------|
| Python | 3.12 |
| PyTorch | 2.10+ with CUDA 12.8 |
| vLLM | >= 0.17.1 (V1 engine with Automatic Prefix Caching) |
| GPU | 1× A800 80GB (FP8 Qwen3-Omni + TTS co-resident). Multi-GPU also supported via `--tensor-parallel-size`. |
| System | `ffmpeg`, `numactl` |
| OS | Linux (tested on Ubuntu 22.04) |
| Browser | Google Chrome (desktop or mobile) |

### Installation

We use [**uv**](https://docs.astral.sh/uv/) for fast, reproducible environment management.

#### 1. Install uv

If you do not have `uv` installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Set Up the Environment

```bash
git clone https://github.com/aurateam2026/AURA.git && cd AURA

# Install system dependencies
sudo apt install -y ffmpeg numactl

# Create a Python 3.12 virtual environment and install all packages
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# Install flash-attn (requires a platform-specific .whl matching your CUDA/PyTorch/arch)
# Download the correct wheel from https://github.com/Dao-AILab/flash-attention/releases
# Example for CUDA 12 + PyTorch 2.10 + x86_64:
uv pip install flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

> **Note:** `flash-attn` is **not** included in `requirements.txt` because it requires a platform-specific `.whl` file. You must download the correct wheel that matches your CUDA version, PyTorch version, and CPU architecture, then install it manually.

> **Note:** `qwen-asr` is installed from a [patched fork](https://github.com/yangbo0926/Qwen3-ASR) instead of PyPI. The upstream `qwen-asr==0.0.6` targets vLLM 0.14.0, while AURA requires vLLM 0.17.1+. Our fork applies a [minimal patch](https://github.com/yangbo0926/Qwen3-ASR/commit/37422f4) (1 file, ~10 lines) to fix three deprecated API calls. Once the upstream package releases a vLLM 0.17+ compatible version, we will switch back to PyPI.

> **Note:** The `Qwen3-TTS-streaming/` subdirectory is a local library loaded at runtime via `sys.path`. It does **not** need a separate install.

#### 3. Verify Installation

```bash
source .venv/bin/activate
python -c "
import torch, vllm, flask, qwen_omni_utils
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'vLLM:     {vllm.__version__}')
print(f'GPUs:     {torch.cuda.device_count()}')
"
```

Expected output:
```
PyTorch:  2.10.0
CUDA:     12.8
vLLM:     0.17.1
GPUs:     2  (or more)
```

### Quick Start

#### 1. Download Models

Download the following models from [Hugging Face](https://huggingface.co/):

| Model | Purpose | Size |
|-------|---------|------|
| [Qwen3-Omni-30B-A3B-FP8](https://huggingface.co/marksverdhei/Qwen3-Omni-30B-A3B-FP8) | End-to-end multimodal (audio + video + text) | ~30 GB |
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/tree/main) | Text-to-Speech synthesis | ~4 GB |

> **Why no ASR?** Qwen3-Omni ingests the user's audio directly. The transcription is recovered from a `<query>...</query>` prefix that the model emits at the start of its response — see `aura/query_extractor.py` and `arch.md` §5. The standalone ASR service that AURA-8B required has been removed.

#### 2. Configure (optional)

All ports and the model path are read from environment variables — see `.env.example` at the repo root. The only required variable is `AURA_MODEL_PATH`:

```bash
cp .env.example .env
# Edit .env: set AURA_MODEL_PATH to your downloaded model directory.
# Then load it into your shell before launching:
set -a; source .env; set +a
```

Defaults (when env is unset): Flask `5003`, TTS `8002`, inference `12345`, infer host `127.0.0.1`.

#### 3. One-Click Launch

```bash
# Default: GPU 0 for TTS + Qwen3-Omni inference (single-A800 co-resident)
bash start_all.sh
```

The script automatically:
- Cleans up any leftover processes on the configured ports (defaults: 8002, 12345)
- Starts TTS and vLLM inference server in order
- Waits for each service to be healthy before proceeding
- Logs to `logs/tts.log`, `logs/vllm.log`
- `Ctrl+C` cleanly shuts down all services

**Custom GPU allocation:**

```bash
# Separate cards for TTS and inference
GPU_TTS=0 GPU_INFERENCE=1 bash start_all.sh

# Multi-GPU inference (tensor parallel)
GPU_TTS=0 GPU_INFERENCE=2,3 bash start_all.sh
```

#### 4. Launch Web Frontend

The web frontend connects to the backend inference server via a TCP socket. The connection is configured via environment variables (see `.env.example`) or CLI flags — defaults are `127.0.0.1:12345`, suitable when frontend and backend run on the same machine.

```bash
source .venv/bin/activate

# Same machine (defaults are fine):
python realtime_capture_video_audio_streaming.py

# Different machine — point at the backend host:
python realtime_capture_video_audio_streaming.py --infer-host <backend-host> --infer-port 12345

# Or via environment variables:
AURA_INFER_HOST=<backend-host> AURA_INFER_PORT=12345 \
    python realtime_capture_video_audio_streaming.py
```

| Mode | Command |
|------|---------|
| HTTP (default) | `python realtime_capture_video_audio_streaming.py` |
| HTTPS | `python realtime_capture_video_audio_streaming.py --https` |
| Cloudflare Tunnel | `python realtime_capture_video_audio_streaming.py --tunnel` |

#### 5. Access from a Browser

> **Required:** You must use **Google Chrome** to access the demo. Chrome is the only browser that fully supports the camera, microphone, MediaRecorder, and Web Audio APIs used by AURA. **Safari and Firefox are not supported** and may fail silently.

**Local access (desktop):**

Open `http://localhost:5003` in **Chrome**.

**Remote access from a phone:**

Open the demo in **Chrome for Android** or **Chrome for iOS**. The phone must be able to reach the frontend server. There are several ways:

1. **Same LAN**: If the phone and the server are on the same network, open `http://<server-ip>:5003` in Chrome on your phone. Note that Chrome **requires HTTPS** to access the camera and microphone from a non-localhost address.

2. **HTTPS mode** (recommended for LAN access from phone):
   ```bash
   # Generate a self-signed certificate (one-time setup)
   openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes

   # Start the frontend with HTTPS
   python realtime_capture_video_audio_streaming.py --https
   ```
   Then open `https://<server-ip>:5003` in Chrome on your phone. You will need to accept the self-signed certificate warning.

3. **Cloudflare Tunnel** (recommended for public/cross-network access):
   ```bash
   python realtime_capture_video_audio_streaming.py --tunnel
   ```
   This creates a public HTTPS URL that you can open in Chrome on any device without network restrictions.

#### 6. Using the Demo (Chrome only)

The interface has three buttons at the bottom of the screen:

| Button | Icon | Action |
|--------|------|--------|
| **Start** | Camera | Tap to start/stop the video stream. The camera feed is sent to the backend for real-time understanding. |
| **Record** | Microphone | **Press and hold** to record your voice. **Release** to stop recording and send the audio to the server for ASR. Do not tap -- you must hold the button down while speaking. |
| **Flip** | Camera Rotate | Tap to switch between the front and rear cameras (useful on phones). |

**Typical workflow:**

1. Open the URL in **Google Chrome** (desktop or mobile).
2. Tap **Start** to activate the camera. Grant camera permission when prompted by Chrome.
3. Point the camera at something you want AURA to understand.
4. **Press and hold** the **Record** button while asking your question out loud. Release when done speaking. Grant microphone permission when prompted.
5. Watch the streaming text response appear on screen in real-time.
6. The TTS audio response will play automatically through your speaker.
7. Tap **Flip** to switch cameras if needed (front/rear).
8. Tap **Start** again to stop the video stream.

> **Tip:** On mobile devices, make sure to grant both camera and microphone permissions when Chrome prompts you. If permissions were previously denied, reset them in Chrome Settings > Site Settings.

### Manual Service Launch

If you prefer to start services individually:

<details>
<summary><b>Step 1: TTS Service (Port 8002)</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0 bash tts_service.sh
```

Verify: `curl http://localhost:8002/v1/tts/health`

</details>

<details>
<summary><b>Step 2: Main Inference Server (Port 12345)</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0 bash Qwen3_VL_online_streaming_v2_CM.sh
```

Wait for: `Server listening on port 12345`

</details>

<details>
<summary><b>Step 3: Web Frontend (Port 5003)</b></summary>

```bash
python realtime_capture_video_audio_streaming.py
```

Open: `http://localhost:5003`

</details>

### GPU Allocation Reference

Single A800 80GB (recommended):

| GPU | Service | VRAM |
|-----|---------|------|
| GPU 0 | TTS (Qwen3-TTS-1.7B) + Qwen3-Omni-30B-A3B-FP8 inference (vLLM, TP=1) | ~38 GB weights + ~35-40 GB KV/activation budget |

Multi-GPU (optional, for BF16 weights or larger context):

```bash
GPU_TTS=0 GPU_INFERENCE=1,2 bash start_all.sh   # TP=2 for inference
```

### Key Configuration

Main inference parameters as passed by `Qwen3_VL_online_streaming_v2_CM.sh`.
These are the values shipped in the launch script — the argparse defaults
in `Qwen3_VL_online_streaming_v2_ContextManaged.py` are more conservative
and are overridden here.

| Parameter | Script value | Description |
|-----------|-------------|-------------|
| `--max-model-len` | 131072 | Maximum context length (128K tokens, sized for FP8 Omni on a single A800) |
| `--gpu-memory-utilization` | 0.80 | Leaves headroom for co-resident TTS + encoder activations |
| `--enable-expert-parallel` | on | MoE expert parallelism (required for Qwen3-Omni-30B-A3B) |
| `--temperature` | 0.5 | Sampling temperature |
| `--max-tokens` | 128 | Max tokens per response |
| `--cross-turn-penalty` | 1 | Cross-turn repetition penalty strength |
| `--cross-turn-lookback` | 10 | Number of recent turns to penalize |
| `--enable-pruning` | on | Enable sliding-window context pruning |
| `--max-rounds` | 45 | Trigger pruning when rounds exceed this |
| `--num-rounds-keep` | 30 | Rounds to keep after pruning |
| `--kv-offloading-size` | 20 | KV cache CPU offload size (GB) |
| `--query-echo-max-chars` | 80 | Fallback threshold for `<query>...</query>` parser |

### Project Structure

```
.
├── .env.example                              # Configuration template (ports, model path)
├── start_all.sh                              # One-click launch script for TTS + inference
├── Qwen3_VL_online_streaming_v2_CM.sh        # Main inference launch script (argparse wiring)
├── Qwen3_VL_online_streaming_v2_ContextManaged.py  # Core: vLLM engine driver + TCP server
├── tts_service.py / tts_service.sh           # TTS service (streaming synthesis)
├── realtime_capture_video_audio_streaming.py # Flask bridge between browser and inference (SSE + TCP)
│
├── aura/                                     # Extracted modules (all unit-tested)
│   ├── protocol.py                           #   Wire format: 9-byte header + message types
│   ├── server_io.py                          #   Outbound token / audio-chunk / query-echo senders
│   ├── session.py                            #   StreamingSession dataclass (per-connection state)
│   ├── session_history.py                    #   Sliding-window chat history with pruning, audio rewrite
│   ├── cross_turn_penalty.py                 #   n-gram penalty + logit bias against prior turns
│   ├── media.py                              #   WebM → numpy video frame decoding
│   ├── audio_media.py                        #   WebM/Opus → mono float32 @ 16 kHz waveform
│   ├── query_extractor.py                    #   Streaming <query>...</query> prefix parser
│   ├── tts.py                                #   TTSController: sentence queue + worker thread
│   ├── sse.py                                #   Server-Sent Events framing for /api/events
│   └── text_utils.py                         #   remove_markdown() for TTS-safe text
│
├── templates/index_streaming.html            # Browser UI (layout + CSS + <script src>)
├── static/                                   # Frontend JS modules, loaded by index_streaming.html
│   ├── state.js                              #   Shared globals + APP_CONFIG reader + DOM refs
│   ├── ui.js                                 #   log(), updateStats()
│   ├── playback.js                           #   Web Audio API streaming PCM
│   ├── sse.js                                #   /api/events subscription + streaming-token handler
│   ├── video.js                              #   Camera capture + MediaRecorder + frame send
│   ├── session.js                            #   /api/force_release_session
│   ├── audio.js                              #   Mic capture + audio send + mic button wiring
│   └── main.js                               #   onload + audio unlock
│
├── tests/                                    # ~165 pytest cases (pure unit tests, no GPU needed)
│   ├── test_protocol.py / test_session.py / test_text_utils.py
│   ├── test_session_history.py / test_cross_turn_penalty.py
│   ├── test_media.py / test_audio_media.py / test_query_extractor.py
│   ├── test_server_io.py / test_sse.py / test_tts.py
│   └── conftest.py
│
├── requirements.txt                          # Python dependencies for the demo server
├── AURA_bench_eval/                          # Benchmark evaluation (separate env — see below)
├── shuhan.mp3                                # TTS reference audio for voice cloning
└── Qwen3-TTS-streaming/                      # Vendored TTS model inference library
```

### Running Tests

The `aura/` modules and the frontend's Python plumbing are covered by a
pytest suite that does **not** require a GPU, the vLLM engine, or any of
the heavy model downloads:

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

All 165+ tests should pass in a few seconds (7 audio-decode tests
skip if `soundfile` is absent from your dev environment — they run
in the deployment environment). They cover the wire protocol, session
state, TTS controller state machine, SSE framing, video/audio decoding,
cross-turn penalty, history pruning (including audio rewrite), the
`<query>` prefix parser, and server_io error isolation. This makes it
safe to iterate on the Python side without spinning up an inference node.

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `sched_setaffinity: Invalid argument` | Remove `numactl` from the launch script |
| `<query>` fallback rate is high | Increase `--query-echo-max-chars`, or check that the system prompt reaches the model (inspect `debug_context.jsonl`) |
| TTS voice clone fails | Verify the reference audio file exists in the working directory |
| OOM on main GPU | Reduce `--gpu-memory-utilization` or `--max-model-len`; confirm you're using the FP8 weights, not BF16 |
| vLLM version error | Requires vLLM >= 0.17.1 with V1 engine support |
| vLLM can't load Qwen3-Omni-FP8 | Ensure `trust_remote_code=True` is honored (it is when model path contains "omni"); check `config.json` has `quantization_config` |
| Phone cannot access camera/mic | Use HTTPS mode or Cloudflare Tunnel (browsers require HTTPS for media on non-localhost) |
| Frontend cannot reach backend | Set `AURA_INFER_HOST` (env or `--infer-host` flag) to the backend host; default is `127.0.0.1` |

## Benchmark Evaluation

We provide benchmark evaluation code for AURA in [`AURA_bench_eval/`](AURA_bench_eval/), including evaluation pipelines for `OVO-Bench`, `StreamingBench`, and `OmniMMI`. For environment setup, dataset preparation, deployment, and evaluation commands, please refer to [`AURA_bench_eval/README.md`](AURA_bench_eval/README.md).

> **Note:** `AURA_bench_eval/` uses its own `requirements.txt` with a different `torch`/`vllm`/`transformers` pin than the demo server. Install it into a **separate virtualenv** inside `AURA_bench_eval/` — do not mix the two.

## Other

### License

This project is released under the [Apache-2.0 License](LICENSE).

### Citation

```bibtex
@article{aura2026,
  title={AURA: Always-On Understanding and Real-Time Assistance via Video Streams},
  author={Lu, Xudong and Bo, Yang and Chen, Jinpeng and Li, Shuhan and Guo, Xintong and Guan, Huankang and Liu, Fang and Xu, Dunyuan and Sun, Peiwen and Sun, Heyang and Liu, Rui and Li, Hongsheng},
  journal={arXiv preprint arXiv:2604.04184},
  year={2026}
}
```
