#!/usr/bin/env bash
set -e
ulimit -n 8192
WANDB_DISABLED=true OMP_THREAD_LIMIT=16 python train_pipeline.py \
  --grid_size 128 --output_dir temp  --no_cuda --max_steps 125000000 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --dataloader_num_workers 1 --logging_steps 1 --experiment_prefix test_swarm \
  --initial_peers '/ip4/127.0.0.1/tcp/45485/p2p/QmbKrXmsqbNgRrb6xVUW8Dw1TwTwsjNETV8vixTMQ8MQHc' 2>&1 | tee -a trainer_stderr_$ind.log
