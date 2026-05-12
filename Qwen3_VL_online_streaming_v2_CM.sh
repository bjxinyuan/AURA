#!/bin/bash
#
# Main inference launch script for Qwen3-Omni end-to-end mode.
#
# Qwen3-Omni takes audio and video natively, so there is no separate
# ASR service — the transcription is extracted from a <query>...</query>
# prefix that the model emits. See aura/query_extractor.py.
#
# Defaults below are tuned for a single A800 (80 GB) running the FP8
# weights of Qwen3-Omni-30B-A3B co-resident with the TTS service.

MODEL_PATH="${AURA_MODEL_PATH:?Please set AURA_MODEL_PATH to your Qwen3-Omni model directory (see .env.example)}"
echo "MODEL_PATH: $MODEL_PATH"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TP=${TP_SIZE:-1}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, tensor-parallel-size=$TP"

python -u Qwen3_VL_online_streaming_v2_ContextManaged.py \
    --listen-port "${AURA_INFER_PORT:-12345}" \
    --model $MODEL_PATH \
    --tensor-parallel-size $TP \
    --max-model-len 131072 \
    --max-seq-len 131072 \
    --gpu-memory-utilization 0.80 \
    --enable-expert-parallel \
    --kv-offloading-size 20 \
    --disable-hybrid-kv-cache-manager \
    --block-size 16 \
    --prefix-caching-hash-algo xxhash \
    --mm-encoder-attn-backend FLASH_ATTN \
    --mm-encoder-tp-mode data \
    --max-num-batched-tokens 15360 \
    --temperature 0.5 \
    --max-tokens 128 \
    --enable-tts \
    --tts-service-url "http://localhost:${AURA_TTS_PORT:-8002}" \
    --tts-output-dir tts_results \
    --cross-turn-penalty 1 \
    --cross-turn-lookback 10 \
    --cross-turn-ngram-sizes \
    --enable-pruning \
    --max-rounds 45 \
    --num-rounds-keep 30 \
    --max-context-qas 10 \
    --query-echo-max-chars 80 \
    --debug-context-file debug_context.jsonl \
    --debug-context
