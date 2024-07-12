# dominant:         random, frequency, knowledge
# similarity_base:  weight, router-weight, router-logits, expert-output
# merge:            no, freq, zipit, update, fix-dom, unmerge, fix-dom-same
# zipit merge mode: normal, activation-with-router-logits, input-weight, all
# task: winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte,

# zipit
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --merge="zipit" \
  --mode="all" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_freq-dom-router-logits-group-zipit-merge-all.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-router-logits-group-zipit-merge-all" |& tee results/log_45e_freq-dom-router-logits-group-zipit-merge-all

# fix-dom
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --merge="fix-dom" \
  --mode="all" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_freq-dom-router-logits-group-fix-dom-merge-all.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-router-logits-group-fix-dom-merge-all" |& tee results/log_45e_freq-dom-router-logits-group-fix-dom-merge-all

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --merge="fix-dom" \
  --mode="input-weight" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_freq-dom-router-logits-group-fix-dom-merge-input-weight.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-router-logits-group-fix-dom-merge-input-weight" |& tee results/log_45e_freq-dom-router-logits-group-fix-dom-merge-input-weight

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --merge="fix-dom" \
  --mode="activation-with-router-logits" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_freq-dom-router-logits-group-fix-dom-merge-activation.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-router-logits-group-fix-dom-merge-activation" |& tee results/log_45e_freq-dom-router-logits-group-fix-dom-merge-activation

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --merge="fix-dom" \
  --mode="normal" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_freq-dom-router-logits-group-fix-dom-merge.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-router-logits-group-fix-dom-merge" |& tee results/log_45e_freq-dom-router-logits-group-fix-dom-merge
