export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM="false"



accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29518 mcsmoe/msmoe-merging-mixtral.py \
  --model_name="eastwind/tinymix-8x1b" \
  --task="winogrande" \
  --similarity_base="router-logits" \
  --mode="normal" \
  --num_average_groups=4 \
  --eval_batch_size=8 \
  --output_path="/home/u2139934/Warehouse/models/mc-smoe/eastwind-tinymix-8x1b/merge-4e/test"