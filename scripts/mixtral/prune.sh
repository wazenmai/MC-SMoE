export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM="false"

# task: winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29513 mcsmoe/pruning-mixtral.py \
  --model_name="s3nh/TinyLLama-4x1.1B-MoE" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --train_batch_size=1 \
  --eval_batch_size=16 \
  --reconstruct_batch_size=1024 \
  --n_sentences=16 \
  --constraint=0.5 \
  --output_path="/home/wazenmai/Warehouse/NLP/checkpoints/kprune/s3nh-tinyllama-4x1b/test_0.5_data_16" |& tee log