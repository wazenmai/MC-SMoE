export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=3

accelerate launch --config_file static/evaluation_config.yaml \
  --main_process_port 29513 mcsmoe/profile-flops.py \
  --task="squad" \
  --checkpoint="results/squad/t5-bs32-lr-1e-4/best" \
  --batch_size=512

