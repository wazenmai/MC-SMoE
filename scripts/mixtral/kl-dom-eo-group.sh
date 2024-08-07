# dominant:         random, frequency, knowledge
# similarity_base:  weight, router-weight, router-logits, expert-output
export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/mnt/nfs/wazenmai/huggingface"

# merge:            no, freq, zipit, update, fix-dom, unmerge, fix-dom-same
# zipit merge mode: normal, activation-with-router-logits, input-weight, all
# task: winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte,

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29504 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/mnt/nfs/wazenmai/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="expert-output" \
  --merge="zipit" \
  --mode="all" \
  --num_average_groups=6 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_6e_kl-dom-expert-output-group-zipit-merge-all-50k.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-6e/kl-dom-expert-output-group-zipit-all-merge-50k" |& tee results/log_6e_kl-dom-expert-output-group-zipit-all-merge-50k

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29504 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/mnt/nfs/wazenmai/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="expert-output" \
  --merge="kl-weight" \
  --mode="normal" \
  --num_average_groups=6 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_6e_kl-dom-expert-output-group-kl-weight-merge-50k.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-6e/kl-dom-expert-output-group-kl-weight-merge-50k" |& tee results/log_6e_kl-dom-expert-output-group-kl-weight-merge-50k
