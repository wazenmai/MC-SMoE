# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0,1
# export TOKENIZERS_PARALLELISM="false"

# dominant:         random, frequency, knowledge
# similarity_base:  weight, router-weight, router-logits, expert-output
# zipit merge mode: normal (default), activation-with-router-logits, input-weight, learnable weight
# merge:            zipit (default), freq
# task: winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte,


# model: AIChenKai/TinyLlama-1.1B-Chat-v1.0-x2-MoE. s3nh/TinyLLama-4x1.1B-MoE, mistralai/Mixtral-8x7B-v0.1
# /app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=6 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/mixtral8x7b/freq-dom-router-logits-group-freq-merge-32" |& tee results/log_freq-dom-router-logits-group-freq-merge-32

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="weight" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=6 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/mixtral8x7b/freq-dom-weight-group-freq-merge-32" |& tee results/log_freq-dom-weight-group-freq-merge-32

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=6 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/mixtral8x7b/freq-dom-expert-output-group-freq-merge-32" |& tee results/log_freq-dom-expert-output-group-freq-merge-32

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=6 \
  --n_sentences=128 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/mixtral8x7b/freq-dom-router-logits-group-freq-merge-128" |& tee results/log_freq-dom-router-logits-group-freq-merge-128

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="weight" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=6 \
  --n_sentences=128 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/mixtral8x7b/freq-dom-weight-group-freq-merge-128" |& tee results/log_freq-dom-weight-group-freq-merge-128

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=6 \
  --n_sentences=128 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/mixtral8x7b/freq-dom-expert-output-group-freq-merge-128" |& tee results/log_freq-dom-expert-output-group-freq-merge-128
