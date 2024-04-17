accelerate launch --config_file static/finetune_config.yaml \
  mcsmoe/finetune-switch-transformers.py \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --num_epochs=1 \
  --report=False \
  --no_eval_until_epochs=1 \
  --save_each_epoch=False \
  --preprocessing_num_workers=8 \
  --num_experts=32 \
  --task="mrpc" \
  --learning_rate=3e-5 \
  --warmup_steps=16 \
  --output_dir="/home/u2139934/Warehouse/models/mc-smoe/mrpc/switch-32e-mrpc" |& tee log1
  # --checkpoint="/home/u2139934/Warehouse/models/mc-smoe/copa/switch-32e-copa/best" \

# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0

# accelerate launch --config_file static/evaluation_config.yaml \
#   --main_process_port 29513 mcsmoe/evaluate-fsgpt-zero-shot.py \
#   --checkpoint="/home/u2139934/Warehouse/models/mc-smoe/copa/switch-32e-copa/best" \
#   --tasks="copa" \
#   --eval_batch_size=32