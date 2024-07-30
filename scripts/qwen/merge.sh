export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/mnt/nfs/wazenmai/huggingface"

# zipit
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --merge="zipit" \
  --mode="all" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_qwen_45e_freq-dom-expert-output-group-zipit-merge-all.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-expert-output-group-zipit-merge-all" |& tee results/log_45e_freq-dom-expert-output-group-zipit-merge-all

# fix-dom
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --merge="fix-dom" \
  --mode="all" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_qwen_45e_freq-dom-expert-output-group-fix-dom-merge-all.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-expert-output-group-fix-dom-merge-all" |& tee results/log_45e_freq-dom-expert-output-group-fix-dom-merge-all

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --merge="fix-dom" \
  --mode="input-weight" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_qwen_45e_freq-dom-expert-output-group-fix-dom-merge-input-weight.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-expert-output-group-fix-dom-merge-input-weight" |& tee results/log_45e_freq-dom-expert-output-group-fix-dom-merge-input-weight

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --merge="fix-dom" \
  --mode="activation-with-router-logits" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_qwen_45e_freq-dom-expert-output-group-fix-dom-merge-activation.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-expert-output-group-fix-dom-merge-activation" |& tee results/log_45e_freq-dom-expert-output-group-fix-dom-merge-activation

