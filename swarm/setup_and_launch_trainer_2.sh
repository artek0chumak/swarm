#!/usr/bin/env bash
set -e
ulimit -n 8192
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=-1 python train_pipeline.py \
  --grid_size 128 --output_dir temp  --no_cuda --max_steps 125000000 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --dataloader_num_workers 0 --logging_steps 1 --experiment_prefix test_swarm \
  --initial_peers '/ip4/10.128.0.2/tcp/46015/p2p/QmcvLHH6ZSDBNmbZs3rLPvcWKhz45XJQrDMnfpmTTpYMWz' 2>&1 | tee -a trainer_stderr_2.log
