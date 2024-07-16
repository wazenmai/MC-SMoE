export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/mnt/nfs/wazenmai/huggingface"

# fix-dom-same
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --merge="fix-dom-same" \
  --mode="activation-with-router-logits" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --result_path="/app/results/results_qwen_45e_freq-dom-expert-output-group-fix-dom-same-merge-activation.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-expert-output-group-fix-dom-same-merge-activation" |& tee results/log_45e_freq-dom-expert-output-group-fix-dom-same-merge-activation

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="expert-output" \
  --merge="fix-dom-same" \
  --mode="input-weight" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=1 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_qwen_45e_kl-dom-expert-output-group-fix-dom-same-merge-input-weight.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-expert-output-group-fix-dom-same-merge-input-weight" |& tee results/log_45e_kl-dom-expert-output-group-fix-dom-same-merge-input-weight

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="router-logits" \
  --merge="fix-dom-same" \
  --mode="input-weight" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=1 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_qwen_45e_kl-dom-router-logits-group-fix-dom-same-merge-input-weight.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-45e/kl-dom-router-logits-group-fix-dom-same-merge-input-weight" |& tee results/log_45e_kl-dom-router-logits-group-fix-dom-same-merge-input-weight

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="weight" \
  --merge="zipit" \
  --mode="input-weight" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=1 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_qwen_45e_45e_kl-dom-weight-group-zipit-merge-input-weight.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-weight-group-zipit-input-weight-merge" |& tee results/log_45e_kl-dom-weight-group-zipit-input-weight-merge
