#!/usr/bin/env bash
set -e
ip addr show ens6 | grep "inet .* scope global" | awk '{{print $2}}' | cut -d/ -f1 > ip_address
export IPADDR=$(cat ip_address)
ulimit -n 8192

src-server --num_experts 1 \
--expert_pattern body2.0.[0:127] --expert_cls lm_body --hidden_dim 512 --num_handlers 64 \
--scheduler linear --fp16 --stats_report_interval 60 \
--num_warmup_steps 128 --num_total_steps 15000 --clip_grad_norm 1.0 --compression BLOCKWISE_8BIT \
--averaging_target_batch_size 128 --averaging_expiration 60 --averaging_timeout 700 --metadata_expiration 700 \
--min_batch_size 1 --max_batch_size 1 --offload \
--device cuda:1 --listen_on 127.0.0.1:* --dht_listen_on ip4/127.0.0.1 \
--initial_peers "/ip4/10.128.0.2/tcp/46015/p2p/QmcvLHH6ZSDBNmbZs3rLPvcWKhz45XJQrDMnfpmTTpYMWz" 2>&1 | tee -a server_stderr_body_2.log;
