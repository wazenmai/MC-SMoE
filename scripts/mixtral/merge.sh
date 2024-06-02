export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM="false"

# mode: normal, activation-with-router-logits, input-weight, learnable weight

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29513 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="s3nh/TinyLLama-4x1.1B-MoE" \
  --task="arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte,winogrande" \
  --similarity_base="expert-output" \
  --mode="activation-with-router-logits" \
  --num_average_groups=2 \
  --eval_batch_size=8 \
  --partition=2 \
  --output_path="/home/wazenmai/Warehouse/NLP/checkpoints/mc-smoe/s3nh-tinyllama-4e/merge-2e/expert-output-group-zipit-activation-with-router-logits-merge-10000"