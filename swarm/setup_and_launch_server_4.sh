#!/usr/bin/env bash
set -e
ip addr show ens6 | grep "inet .* scope global" | awk '{{print $2}}' | cut -d/ -f1 > ip_address
export IPADDR=$(cat ip_address)
ulimit -n 8192

src-server --num_experts 1 \
--expert_pattern tail.0.[0:127] --expert_cls lm_tail --hidden_dim 1024 --num_handlers 64 \
--scheduler linear --fp16 --stats_report_interval 60 \
--num_warmup_steps 3125 --num_total_steps 15000 --clip_grad_norm 1.0 --compression BLOCKWISE_8BIT \
--averaging_target_batch_size 4096 --averaging_expiration 60 --averaging_timeout 700 --metadata_expiration 700 \
--min_batch_size 1 --max_batch_size 1 --offload \
--device cuda:0 --listen_on 127.0.0.1:* --dht_listen_on ip4/127.0.0.1 \
--initial_peers "/ip4/127.0.0.1/tcp/45485/p2p/QmbKrXmsqbNgRrb6xVUW8Dw1TwTwsjNETV8vixTMQ8MQHc" 2>&1 | tee -a server_stderr_tail_1.log;
