# accelerate launch --config_file static/finetune_config.yaml \
#   --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
#   --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
#   --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
#   --dominant="knowledge" \
#   --similarity_base="weight" \
#   --mode="update" \
#   --num_average_groups=6 \
#   --n_sentences=32 \
#   --train_batch_size=8 \
#   --eval_batch_size=32 \
#   --partition=1 \
#   --output_path="/app/results/mc-smoe/mixtral8x7b/merge-6e/kl-dom-weight-group-zipit-update-merge-50k" |& tee results/log_6e_kl-dom-weight-group-zipit-update-merge-50k

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="knowledge" \
  --similarity_base="expert-output" \
  --mode="dom-group" \
  --num_average_groups=6 \
  --n_sentences=32 \
  --train_batch_size=8 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/mixtral8x7b/merge-6e/kl-dom-expert-output-group-zipit-dom-group-merge-50k" |& tee results/log_6e_kl-dom-expert-output-group-zipit-dom-group-merge-50k

# accelerate launch --config_file static/finetune_config.yaml \
#   --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
#   --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
#   --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
#   --dominant="knowledge" \
#   --similarity_base="expert-output" \
#   --mode="update" \
#   --num_average_groups=6 \
#   --n_sentences=32 \
#   --train_batch_size=8 \
#   --eval_batch_size=32 \
#   --partition=1 \
#   --output_path="/app/results/mc-smoe/mixtral8x7b/merge-6e/kl-dom-expert-output-group-zipit-update-merge-50k" |& tee results/log_6e_kl-dom-expert-output-group-zipit-update-merge-50k
