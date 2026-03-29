#!/bin/bash
# AniML LoRA Training — Pass 1 (I2V + Plücker, zero GLD)
# Edit COMMON / TRAINING args below, then run:
#   bash scripts/wan2.2_fun/train_animl_lora.sh

# ── Common paths ──────────────────────────────────────────────────────────────
MODEL_PATH="/home/ubuntu/dev/models/Wan2.2-Fun-A14B-Control-Camera"
JSONL_PATH="/home/ubuntu/dev/finetrainers_geomgen/dataset_da3_out.jsonl"
OUTPUT_DIR="/home/ubuntu/dev/animl_videox/checkpoints/lora_pass1"

# ── Resolution (must be divisible by 16, frames must be 4k+1) ────────────────
HEIGHT=256
WIDTH=384
N_FRAMES=21

# ── LoRA ─────────────────────────────────────────────────────────────────────
RANK=32
NETWORK_ALPHA=16

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE=1e-4
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4   # effective batch = 4
MAX_TRAIN_STEPS=2000
CHECKPOINTING_STEPS=200
CHECKPOINTS_TOTAL_LIMIT=5
LR_SCHEDULER="constant_with_warmup"
LR_WARMUP_STEPS=100
BOUNDARY_TYPE="both"             # train both low+high noise transformers

# ── Launch ────────────────────────────────────────────────────────────────────
python scripts/wan2.2_fun/train_animl_lora.py \
  --model_path       "$MODEL_PATH" \
  --jsonl_path       "$JSONL_PATH" \
  --output_dir       "$OUTPUT_DIR" \
  --height           $HEIGHT \
  --width            $WIDTH \
  --n_frames         $N_FRAMES \
  --rank             $RANK \
  --network_alpha    $NETWORK_ALPHA \
  --learning_rate    $LEARNING_RATE \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_train_steps  $MAX_TRAIN_STEPS \
  --checkpointing_steps $CHECKPOINTING_STEPS \
  --checkpoints_total_limit $CHECKPOINTS_TOTAL_LIMIT \
  --lr_scheduler     $LR_SCHEDULER \
  --lr_warmup_steps  $LR_WARMUP_STEPS \
  --boundary_type    $BOUNDARY_TYPE \
  --gradient_checkpointing \
  --num_workers 4 \
  2>&1 | tee "$OUTPUT_DIR/train.log"