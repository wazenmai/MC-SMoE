export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM="false"

# dominant:         random, frequency, knowledge
# similarity_base:  weight, router-weight, router-logits, expert-output
# zipit merge mode: normal, activation-with-router-logits, input-weight, learnable weight

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="router-logits" \
  --mode="normal" \
  --num_average_groups=30 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/home/wazenmai/Warehouse/NLP/checkpoints/mc-smoe/qwen/merge-30e/freq-dom-router-logits-group-zipit-merge"