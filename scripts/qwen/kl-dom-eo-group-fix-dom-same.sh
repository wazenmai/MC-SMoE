# fix-dom-same
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="expert-output" \
  --merge="fix-dom-same" \
  --mode="all" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=8 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_kl-dom-expert-output-group-fix-dom-same-merge-all.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-expert-output-group-fix-dom-same-merge-all" |& tee results/log_45e_kl-dom-expert-output-group-fix-dom-same-merge-all

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
  --train_batch_size=8 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_kl-dom-expert-output-group-fix-dom-same-merge-input-weight.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-expert-output-group-fix-dom-same-merge-input-weight" |& tee results/log_45e_kl-dom-expert-output-group-fix-dom-same-merge-input-weight

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="expert-output" \
  --merge="fix-dom-same" \
  --mode="activation-with-router-logits" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=8 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_kl-dom-expert-output-group-fix-dom-same-merge-activation.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-expert-output-group-fix-dom-same-merge-activation" |& tee results/log_45e_kl-dom-expert-output-group-fix-dom-same-merge-activation

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="expert-output" \
  --merge="fix-dom-same" \
  --mode="normal" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=8 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_kl-dom-expert-output-group-fix-dom-same-merge.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-expert-output-group-fix-dom-same-merge" |& tee results/log_45e_kl-dom-expert-output-group-fix-dom-same-merge
