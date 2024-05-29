export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

accelerate launch --config_file static/evaluation_config.yaml \
  --main_process_port 29515 mcsmoe/evaluate_zero_shot.py \
  --checkpoint="/home/wazenmai/Warehouse/NLP/checkpoints/mc-smoe/switch-32e/mrpc-fine/best" \
  --tasks="mrpc" \
  --eval_batch_size=32 |& tee log_eval
