# data=8
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/pruning-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="openbookqa,rte,winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu" \
  --start_layer=24 \
  --constraint=0.7 \
  --n_sentences=8 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --result_path="/app/results/result_con_0.7_data_8_layer24to31_nor.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/kprune/con_0.7_data_8_layer24to31_nor" |& tee results/log_0.7_data_8_layer24to31_nor

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/pruning-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="openbookqa,rte,winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu" \
  --start_layer=16 \
  --constraint=0.7 \
  --n_sentences=8 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --result_path="/app/results/result_con_0.7_data_8_layer16to31.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/kprune/con_0.7_data_8_layer16to31" |& tee results/log_0.7_data_8_layer16to31

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/pruning-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="openbookqa,rte,winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu" \
  --start_layer=0 \
  --constraint=0.7 \
  --n_sentences=8 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --result_path="/app/results/result_con_0.7_data_8_all.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/kprune/con_0.7_data_8_all" |& tee results/log_0.7_data_8_all

## con=0.9
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/pruning-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="openbookqa,rte,winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu" \
  --start_layer=24 \
  --constraint=0.9 \
  --n_sentences=8 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --result_path="/app/results/result_con_0.9_data_8_layer24to31_nor.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/kprune/con_0.9_data_8_layer24to31_nor" |& tee results/log_0.9_data_8_layer24to31_nor

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/pruning-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="openbookqa,rte,winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu" \
  --start_layer=16 \
  --constraint=0.9 \
  --n_sentences=8 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --result_path="/app/results/result_con_0.9_data_8_layer16to31.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/kprune/con_0.9_data_8_layer16to31" |& tee results/log_0.9_data_8_layer16to31

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/pruning-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="openbookqa,rte,winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu" \
  --start_layer=0 \
  --constraint=0.9 \
  --n_sentences=8 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --result_path="/app/results/result_con_0.9_data_8_all.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/kprune/con_0.9_data_8_all" |& tee results/log_0.9_data_8_all

### con=0.5
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/pruning-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="openbookqa,rte,winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu" \
  --start_layer=24 \
  --constraint=0.5 \
  --n_sentences=8 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --result_path="/app/results/result_con_0.5_data_8_layer24to31_nor.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/kprune/con_0.5_data_8_layer24to31_nor" |& tee results/log_0.5_data_8_layer24to31_nor

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/pruning-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="openbookqa,rte,winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu" \
  --start_layer=16 \
  --constraint=0.5 \
  --n_sentences=8 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --result_path="/app/results/result_con_0.5_data_8_layer16to31.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/kprune/con_0.5_data_8_layer16to31" |& tee results/log_0.5_data_8_layer16to31

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 mcsmoe/pruning-mixtral.py \
  --model_name="/app/warehouse/huggingface/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841" \
  --task="openbookqa,rte,winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu" \
  --start_layer=0 \
  --constraint=0.5 \
  --n_sentences=8 \
  --train_batch_size=4 \
  --eval_batch_size=16 \
  --result_path="/app/results/result_con_0.5_data_8_all.txt" \
  --output_path="/app/results/mc-smoe/mixtral8x7b/kprune/con_0.5_data_8_all" |& tee results/log_0.5_data_8_all
