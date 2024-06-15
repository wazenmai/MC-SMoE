# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export TOKENIZERS_PARALLELISM="false"

# dominant:         random, frequency, knowledge
# similarity_base:  weight, router-weight, router-logits, expert-output
# zipit merge mode: normal, activation-with-router-logits, input-weight, learnable weight

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-router-logits-group-freq-merge-32" |& tee results/log_qwen_45e_freq-dom-router-logits-group-freq-merge-32

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=45 \
  --n_sentences=128 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-router-logits-group-freq-merge-128" |& tee results/log_qwen_45e_freq-dom-router-logits-group-freq-merge-128

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="weight" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-weight-group-freq-merge-32" |& tee results/log_qwen_45e_freq-dom-weight-group-freq-merge-32

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="weight" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=45 \
  --n_sentences=128 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-weight-group-freq-merge-128" |& tee results/log_qwen_45e_freq-dom-weight-group-freq-merge-128

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-expert-output-group-freq-merge-32" |& tee results/log_qwen_45e_freq-dom-expert-output-group-freq-merge-32

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=45 \
  --n_sentences=128 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-45e/freq-dom-expert-output-group-freq-merge-128" |& tee results/log_qwen_45e_freq-dom-expert-output-group-freq-merge-128

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=30 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-30e/freq-dom-router-logits-group-freq-merge-32" |& tee results/log_qwen_30e_freq-dom-router-logits-group-freq-merge-32

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=30 \
  --n_sentences=128 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-30e/freq-dom-router-logits-group-freq-merge-128" |& tee results/log_qwen_30e_freq-dom-router-logits-group-freq-merge-128

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="weight" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=30 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-30e/freq-dom-weight-group-freq-merge-32" |& tee results/log_qwen_30e_freq-dom-weight-group-freq-merge-32

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="weight" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=30 \
  --n_sentences=128 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-30e/freq-dom-weight-group-freq-merge-128" |& tee results/log_qwen_30e_freq-dom-weight-group-freq-merge-128

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=30 \
  --n_sentences=32 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-30e/freq-dom-expert-output-group-freq-merge-32" |& tee results/log_qwen_30e_freq-dom-expert-output-group-freq-merge-32

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="normal" \
  --merge="freq" \
  --num_average_groups=30 \
  --n_sentences=128 \
  --train_batch_size=4 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/app/results/mc-smoe/qwen/merge-30e/freq-dom-expert-output-group-freq-merge-128" |& tee results/log_qwen_30e_freq-dom-expert-output-group-freq-merge-128
