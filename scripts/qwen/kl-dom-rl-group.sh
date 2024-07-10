# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0,1
# export TOKENIZERS_PARALLELISM="false"

# dominant:         random, frequency, knowledge
# similarity_base:  weight, router-weight, router-logits, expert-output
# merge:            no, freq, zipit, update, fix-dom, unmerge, fix-dom-same
# zipit merge mode: normal, activation-with-router-logits, input-weight, all
# task: winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte,


# model: AIChenKai/TinyLlama-1.1B-Chat-v1.0-x2-MoE. s3nh/TinyLLama-4x1.1B-MoE, mistralai/Mixtral-8x7B-v0.1
# /app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e45ea1341823841

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="router-logits" \
  --merge="zipit" \
  --mode="normal" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_45e_kl-dom-router-logits-group-zipit-merge.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-router-logits-group-zipit-merge" |& tee results/log_45e_kl-dom-router-logits-group-zipit-merge

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="router-logits" \
  --merge="zipit" \
  --mode="activation-with-router-logits" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_45e_kl-dom-router-logits-group-zipit-merge-activation.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-router-logits-group-zipit-activation-merge" |& tee results/log_45e_kl-dom-router-logits-group-zipit-activation-merge

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="router-logits" \
  --merge="zipit" \
  --mode="input-weight" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_45e_kl-dom-router-logits-group-zipit-merge-input-weight.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-router-logits-group-zipit-input-weight-merge" |& tee results/log_45e_kl-dom-router-logits-group-zipit-input-weight-merge

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="router-logits" \
  --merge="zipit" \
  --mode="all" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_45e_kl-dom-router-logits-group-zipit-merge-all.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-router-logits-group-zipit-all-merge" |& tee results/log_45e_kl-dom-router-logits-group-zipit-all-merge

# kl-weight
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="router-logits" \
  --merge="kl-weight" \
  --mode="normal" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_45e_kl-dom-router-logits-group-kl-weight-merge.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-router-logits-group-kl-weight-merge" |& tee results/log_45e_kl-dom-router-logits-group-kl-weight-merge
