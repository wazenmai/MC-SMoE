export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=2

accelerate launch --config_file static/evaluation_config.yaml \
  --main_process_port 29512 mcsmoe/evaluate-fsgpt-zero-shot.py \
  --checkpoint="Phando/fairseq-dense-125m" \
  --tasks="sst2,mrpc,copa,openbookqa" \
  --eval_batch_size=32