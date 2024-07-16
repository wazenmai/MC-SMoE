export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/mnt/nfs/wazenmai/huggingface"

# dominant:         random, frequency, knowledge
# similarity_base:  weight, router-weight, router-logits, expert-output
# zipit merge mode: normal (default), activation-with-router-logits, input-weight, learnable weight
# merge:            zipit (default), freq
# task: winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte,


# model: AIChenKai/TinyLlama-1.1B-Chat-v1.0-x2-MoE. s3nh/TinyLLama-4x1.1B-MoE, mistralai/Mixtral-8x7B-v0.1
# /mnt/nfs/wazenmai/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/mnt/nfs/wazenmai/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="weight" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=4 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-4e/freq-dom-weight-group-freq-merge-32" |& tee results/log_4e_freq-dom-weight-group-freq-merge-32

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/mnt/nfs/wazenmai/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=4 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-4e/freq-dom-expert-output-group-freq-merge-32" |& tee results/log_4e_freq-dom-expert-output-group-freq-merge-32

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/mnt/nfs/wazenmai/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="weight" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=4 \
  --n_sentences=128 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-4e/freq-dom-weight-group-freq-merge-128" |& tee results/log_4e_freq-dom-weight-group-freq-merge-128

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/mnt/nfs/wazenmai/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="normal" \
  --num_average_groups=4 \
  --n_sentences=128 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-4e/freq-dom-expert-output-group-freq-merge-128" |& tee results/log_4e_freq-dom-expert-output-group-freq-merge-128
