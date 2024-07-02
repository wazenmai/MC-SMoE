accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="normal" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-expert-output-group-zipit-merge" |& tee results/log_qwen_45e_freq-dom-expert-output-group-zipit-merge

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="activation-with-router-logits" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-expert-output-group-zipit-activation-with-router-logits-merge" |& tee results/log_qwen_45e_freq-dom-expert-output-group-zipit-activation-with-router-logits-merge

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="input-weight" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-expert-output-group-zipit-input-weight-merge" |& tee results/log_qwen_45e_freq-dom-expert-output-group-zipit-input-weight-merge
