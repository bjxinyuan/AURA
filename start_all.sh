#!/bin/bash
# 一键启动: TTS + vLLM 主推理 (Qwen3-Omni 端到端模式)
#
# Qwen3-Omni 原生处理音频，不再需要独立的 ASR 服务。
# 日志输出到 logs/ 目录。Ctrl+C 会自动终止所有后台服务。

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── GPU 分配（按需修改）──
# 单 A800 部署：TTS 与主推理共卡。
GPU_TTS=${GPU_TTS:-0}
GPU_INFERENCE=${GPU_INFERENCE:-0}

# ── 服务端口（可通过环境变量覆盖；见 .env.example）──
export AURA_FLASK_PORT="${AURA_FLASK_PORT:-5003}"
export AURA_INFER_PORT="${AURA_INFER_PORT:-12345}"
export AURA_TTS_PORT="${AURA_TTS_PORT:-8002}"

# ── 自动计算 Tensor Parallel 大小 ──
IFS=',' read -ra _GPU_LIST <<< "$GPU_INFERENCE"
TP_SIZE=${TP_SIZE:-${#_GPU_LIST[@]}}

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── 启动前清理残留进程 ──
kill_port() {
    local port=$1
    local pids
    pids=$(ss -tlnp "sport = :$port" 2>/dev/null | awk 'NR>1{match($0,/pid=([0-9]+)/,a); if(a[1]) print a[1]}' | sort -u)
    if [ -n "$pids" ]; then
        echo "⚠️  Port $port is occupied by PID(s): $pids — killing..."
        echo "$pids" | xargs kill 2>/dev/null
        sleep 2
        echo "$pids" | xargs kill -9 2>/dev/null
        sleep 1
    fi
}

echo "🧹 Checking for leftover processes on ports $AURA_TTS_PORT, $AURA_INFER_PORT..."
kill_port "$AURA_TTS_PORT"
kill_port "$AURA_INFER_PORT"

PIDS=()

cleanup() {
    echo ""
    echo "🛑 Shutting down all services..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            echo "  Stopped PID $pid"
        fi
    done
    wait 2>/dev/null
    echo "👋 All services stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

# ── 1. TTS 服务 ──
echo "🔊 Starting TTS service (GPU $GPU_TTS, port $AURA_TTS_PORT)..."
CUDA_VISIBLE_DEVICES=$GPU_TTS bash tts_service.sh > "$LOG_DIR/tts.log" 2>&1 &
PIDS+=($!)
echo "    PID=${PIDS[-1]}, log: logs/tts.log"

echo "    Waiting for TTS to be ready..."
for i in $(seq 1 180); do
    if curl -s "http://localhost:$AURA_TTS_PORT/v1/tts/health" 2>/dev/null | grep -q '"status":"ok"'; then
        echo "    ✓ TTS service ready"
        break
    fi
    if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
        echo "    ✗ TTS process exited unexpectedly, check logs/tts.log"
        cleanup
    fi
    sleep 2
done

# ── 2. 主推理服务 (Qwen3-Omni) ──
echo "🚀 Starting vLLM inference server (GPU $GPU_INFERENCE, TP=$TP_SIZE, port $AURA_INFER_PORT)..."
CUDA_VISIBLE_DEVICES=$GPU_INFERENCE TP_SIZE=$TP_SIZE bash Qwen3_VL_online_streaming_v2_CM.sh > "$LOG_DIR/vllm.log" 2>&1 &
PIDS+=($!)
echo "    PID=${PIDS[-1]}, log: logs/vllm.log"

echo ""
echo "============================================"
echo "  Services launched (Qwen3-Omni E2E mode)"
echo "  TTS:  http://localhost:$AURA_TTS_PORT  (GPU $GPU_TTS)"
echo "  vLLM: port $AURA_INFER_PORT            (GPU $GPU_INFERENCE, TP=$TP_SIZE)"
echo ""
echo "  Logs: $LOG_DIR/"
echo "  Press Ctrl+C to stop all services"
echo "============================================"

wait
