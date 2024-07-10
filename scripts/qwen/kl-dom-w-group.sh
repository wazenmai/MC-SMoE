# dominant:         random, frequency, knowledge
# similarity_base:  weight, router-weight, router-logits, expert-output
# merge:            no, freq, zipit, update, fix-dom, unmerge, fix-dom-same
# zipit merge mode: normal, activation-with-router-logits, input-weight, all
# task: winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte,

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="weight" \
  --merge="zipit" \
  --mode="normal" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_45e_kl-dom-weight-group-zipit-merge.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-weight-group-zipit-merge" |& tee results/log_45e_kl-dom-weight-group-zipit-merge

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="weight" \
  --merge="zipit" \
  --mode="activation-with-router-logits" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_45e_kl-dom-weight-group-zipit-merge-activation.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-weight-group-zipit-activation-merge" |& tee results/log_45e_kl-dom-weight-group-zipit-activation-merge

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
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_45e_kl-dom-weight-group-zipit-merge-input-weight.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-weight-group-zipit-input-weight-merge" |& tee results/log_45e_kl-dom-weight-group-zipit-input-weight-merge

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="weight" \
  --merge="zipit" \
  --mode="all" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_45e_kl-dom-weight-group-zipit-merge-all.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-weight-group-zipit-all-merge" |& tee results/log_45e_kl-dom-weight-group-zipit-all-merge

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="weight" \
  --merge="kl-weight" \
  --mode="normal" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_45e_kl-dom-weight-group-kl-weight-merge.txt" \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/kl-dom-weight-group-kl-weight-merge" |& tee results/log_45e_kl-dom-weight-group-kl-weight-merge
