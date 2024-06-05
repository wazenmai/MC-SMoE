export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM="false"

# dominant:         random, frequency, knowledge
# similarity_base:  weight, router-weight, router-logits, expert-output
# zipit merge mode: normal, activation-with-router-logits, input-weight, learnable weight


# model: AIChenKai/TinyLlama-1.1B-Chat-v1.0-x2-MoE. s3nh/TinyLLama-4x1.1B-MoE

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="s3nh/TinyLLama-4x1.1B-MoE" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --mode="input-weight" \
  --num_average_groups=2 \
  --eval_batch_size=32 \
  --partition=1 \
  --output_path="/home/wazenmai/Warehouse/NLP/checkpoints/mc-smoe/s3nh-tinyllama-4e/merge-2e/freq-dom-expert-output-group-zipit-input-weight-merge-100000"