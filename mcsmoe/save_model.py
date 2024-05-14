import torch
from datasets import load_dataset
from transformers import T5TokenizerFast, get_scheduler, SwitchTransformersForConditionalGeneration
from torch.utils.data import DataLoader

from mcsmoe.data import (
    Seq2SeqDataPreProcessor,
    tokenize_seq2seq,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForSeq2Seq,
    get_evaluate_fn,
    EXTRA_KEYS_FOR_EVAL,
    keep_only_supporting_facts_in_context_for_hotpotqa
)

# checkpoint = "google/switch-base-8"
# checkpoint="/home/wazenmai/Warehouse/NLP/cache/huggingface/hub/models--google--switch-base-32/snapshots/2018338b8dad760fa7a35a754d532486ef3942f9"
# checkpoint = "/home/wazenmai/Warehouse/NLP/checkpoints/mc-smoe/test/switch-8e-mrpc-test1-5/best"
# checkpoint = "/home/wazenmai/Warehouse/NLP/checkpoints/sky_ckpts/switch-32e-mrpc-fine/best" # {'accuracy': 0.8553921568627451, 'f1': 0.8984509466437176, 'loss': 2.0424006496156966}
# num_experts = 32
task = "mrpc"
preprocessing_num_workers = 8
per_device_eval_batch_size = 64



def eval_data_process(task, tokenizer):
    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])
    eval_dataset = raw_dataset["validation"] if task != "mnli" else (
        raw_dataset["validation_matched"], raw_dataset["validation_mismatched"]
    )
    if task == "hotpotqa":
        eval_dataset = eval_dataset.map(
            keep_only_supporting_facts_in_context_for_hotpotqa,
            batched=False,
            num_proc=preprocessing_num_workers
        )
   
    eval_dataset = eval_dataset.map(
        Seq2SeqDataPreProcessor(benchmark=task, keep_specific_keys=EXTRA_KEYS_FOR_EVAL),
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=eval_dataset.column_names,
    )

    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=True),
        num_proc=preprocessing_num_workers,
        batched=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           max_length=tokenizer.model_max_length,
                                           return_tensors='pt',
                                           keys_to_ignore=EXTRA_KEYS_FOR_EVAL)
    print(f"Number of validation examples: {len(tokenized_eval_dataset)}")
    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=per_device_eval_batch_size,
        num_workers=4
    )

    evaluate_fn = get_evaluate_fn(
        task=task,
        tokenizer=tokenizer,
        raw_eval_dataset=raw_dataset['validation']
    )

    return eval_dataloader, evaluate_fn


# model = SwitchTransformersForConditionalGeneration.from_pretrained(checkpoint)
# for n, p in model.named_parameters():
#     print(f"{n} {p.shape} {p[:5]}")
# tokenizer = T5TokenizerFast.from_pretrained(f"google/switch-base-{num_experts}")
# eval_dataloader, evaluate_fn = eval_data_process(task, tokenizer)



# model.eval()
# losses = []
# output_labels = []
# output_predictions = []
# output_ids = [] if task in ["squad", "copa", "multirc", "squad_v2", "hotpotqa"] else None
# for eval_step, eval_batch in enumerate(eval_dataloader):
#     extra_keys_eval_batch = {}
#     for key in list(eval_batch.keys()):
#         if key in EXTRA_KEYS_FOR_EVAL:
#             extra_keys_eval_batch[key] = eval_batch.pop(key)
#     with torch.no_grad():
#         outputs = model(**eval_batch)
#     eval_labels = eval_batch['labels'].clone()
#     output_labels += torch.cat([
#         eval_labels,
#         torch.ones(eval_labels.shape[0], tokenizer.model_max_length - eval_labels.shape[1],
#                     dtype=eval_labels.dtype,
#                     device=eval_labels.device) * -100
#     ], dim=-1)
#     output_predictions.extend(outputs.logits.argmax(dim=-1).tolist())
#     if task in ["squad", "squad_v2", "hotpotqa"]:
#         output_ids += extra_keys_eval_batch["id"]
#     elif task == "copa" or task == "multirc":
#         output_ids += extra_keys_eval_batch["idx"]
#     losses.append(outputs["loss"].mean().item())

# # losses = torch.cat(losses)
# # eval_loss = torch.mean(losses)
# output_labels = torch.stack(output_labels, dim=0)
# eval_res = evaluate_fn(predictions=output_predictions, labels=output_labels, ids=output_ids)
# metric_key = list(eval_res.keys())[0]
# # eval_res["loss"] = eval_loss.item()
# eval_res["loss"] = sum(losses) / len(losses)
# print(f"Eval results: {metric_key} {eval_res[metric_key]}, loss: {eval_res['loss']}")
# print(eval_res)