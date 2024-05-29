export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM="false"



accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29518 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="Corianas/Tiny-Moe" \
  --task="winogrande" \
  --similarity_base="expert-output" \
  --mode="activation-with-router-logits" \
  --num_average_groups=1 \
  --eval_batch_size=1 \
  --output_path="/home/wazenmai/Warehouse/NLP/checkpoints/mc-smoe/corianas-tiny-moe/merge/4e-freq-dominant-expert-output-group-zipit-merge-activation-with-router-logits"