# fix-dom-same
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29505 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/mnt/nfs/wazenmai/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="router-logits" \
  --merge="fix-dom-same" \
  --mode="all" \
  --num_average_groups=6 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_kl-dom-router-logits-group-fix-dom-same-merge-all-50k.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-6e/kl-dom-router-logits-group-fix-dom-same-merge-all-50k" |& tee results/log_6e_kl-dom-router-logits-group-fix-dom-same-merge-all-50k

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29505 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/mnt/nfs/wazenmai/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="router-logits" \
  --merge="fix-dom-same" \
  --mode="input-weight" \
  --num_average_groups=6 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_kl-dom-router-logits-group-fix-dom-same-merge-input-weight-50k.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-6e/kl-dom-router-logits-group-fix-dom-same-merge-input-weight-50k" |& tee results/log_6e_kl-dom-router-logits-group-fix-dom-same-merge-input-weight-50k

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29505 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/mnt/nfs/wazenmai/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="router-logits" \
  --merge="fix-dom-same" \
  --mode="activation-with-router-logits" \
  --num_average_groups=6 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_kl-dom-router-logits-group-fix-dom-same-merge-activation-50k.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-6e/kl-dom-router-logits-group-fix-dom-same-merge-activation-50k" |& tee results/log_6e_kl-dom-router-logits-group-fix-dom-same-merge-activation-50k

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29505 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/mnt/nfs/wazenmai/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="router-logits" \
  --merge="fix-dom-same" \
  --mode="normal" \
  --num_average_groups=6 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --partition=1 \
  --result_path="/app/results/results_kl-dom-router-logits-group-fix-dom-same-merge-50k.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-6e/kl-dom-router-logits-group-fix-dom-same-merge-50k" |& tee results/log_6e_kl-dom-router-logits-group-fix-dom-same-merge-50k
