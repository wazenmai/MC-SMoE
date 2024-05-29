export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM="false"



accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29518 mcsmoe/random-merge.py \
  --task="mrpc" \
  --strategy="zipit" \
  --similarity_base="output" \
  --num_epochs=0 \
  --per_device_train_batch_size=32 \
  --per_device_eval_batch_size=32 \
  --num_groups=4 \
  --encoder_merging_layers="1,3,5,7,9,11" \
  --decoder_merging_layers="1,3,5,7,9,11" \
  --output_dir="/home/wazenmai/Warehouse/NLP/checkpoints/mc-smoe/switch-8e/merge/zipit-withoutkd-use-mrpc-data-output-group-kprune-dominant" \
  --teacher_checkpoint="/home/wazenmai/Warehouse/NLP/checkpoints/mc-smoe/switch-8e/mrpc-fine/best" \
  --student_checkpoint="/home/wazenmai/Warehouse/NLP/checkpoints/mc-smoe/switch-8e/mrpc-fine/best"