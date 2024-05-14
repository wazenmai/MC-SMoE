export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29514 mcsmoe/finetune-switch-transformers.py \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --num_epochs=20 \
  --report=False \
  --no_eval_until_epochs=1 \
  --save_each_epoch=True \
  --preprocessing_num_workers=8 \
  --num_experts=8 \
  --task="mrpc" \
  --learning_rate=3e-5 \
  --warmup_steps=16 \
  --checkpoint="/home/wazenmai/Warehouse/NLP/cache/huggingface/hub/models--google--switch-base-8/snapshots/92fe2d22b024d9937146fe097ba3d3a7ba146e1b" \
  --output_dir="/home/wazenmai/Warehouse/NLP/checkpoints/mc-smoe/test/switch-8e-mrpc-test1-5" |& tee log_mrpc
