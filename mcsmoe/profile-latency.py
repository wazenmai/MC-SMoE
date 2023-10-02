import time
from typing import Optional, Dict

import torch
from accelerate.utils import set_seed
from datasets import load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    SwitchTransformersForConditionalGeneration
)

from mcsmoe.data import (
    Seq2SeqDataPreProcessor,
    tokenize_seq2seq,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForSeq2Seq,
)
from mcsmoe.merging import (
    ExpertsGrouperForSwitch,
    merge_by_groups_with_usage_frequency_weighting
)

set_seed(233)


def prepare_dataloader(task: str, tokenizer: T5TokenizerFast, batch_size: int):
    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])
    dataset = raw_dataset["train"]
    dataset = dataset.map(
        Seq2SeqDataPreProcessor(benchmark=task),
        batched=True,
        num_proc=6,
        remove_columns=dataset.column_names
    )
    dataset = dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=False),
        num_proc=6,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           max_length=tokenizer.model_max_length,
                                           return_tensors='pt')
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=8
    )
    return dataloader


def merge_switch_by_usage_weighted_averaging(
        model: SwitchTransformersForConditionalGeneration,
        batch: Dict[str, torch.Tensor],
        prune_router_non_core: bool = False,
) -> SwitchTransformersForConditionalGeneration:
    grouper = ExpertsGrouperForSwitch(
        config=model.config,
        similarity_fn="cosine",
        similarity_base="router-logits"
    )
    grouper.compute_all_similarities(
        model=model,
        batch=batch,
    )
    grouper.compute_all_usages(
        model=model,
        batch=batch,
    )
    core_experts = grouper.group_experts_into_clusters_by_routing_guided_globally(
        average_num_groups=8,
        merging_encoder_layers=[3, 5, 7, 9, 11],
        merging_decoder_layers=[1, 3, 5, 7, 9, 11]
    )
    model = merge_by_groups_with_usage_frequency_weighting(
        model=model,
        grouper=grouper,
        encoder_merging_layers=[3, 5, 7, 9, 11],
        decoder_merging_layers=[1, 3, 5, 7, 9, 11],
    )

    if prune_router_non_core:
        # Prune the fake experts and router classifier rows
        num_experts = model.config.num_experts
        router_bias = model.config.router_bias
        # Encoder
        for layer_idx in range(3, 12, 2):
            encoder_mlp = model.encoder.block[layer_idx].layer[-1].mlp
            encoder_name = f"encoder.block.{layer_idx}.layer.1.mlp"
            core_expert_indices = torch.tensor(core_experts[encoder_name], dtype=torch.int).sort().values
            # prune router
            encoder_mlp.router.num_experts = len(core_expert_indices)
            with torch.no_grad():
                pruned_router_cls = encoder_mlp.router.classifier.weight.data[core_expert_indices]
                encoder_mlp.router.classifier = torch.nn.Linear(
                    encoder_mlp.router.classifier.in_features,
                    encoder_mlp.router.num_experts,
                    bias=router_bias
                )
                encoder_mlp.router.classifier.weight.data.copy_(pruned_router_cls)
                if router_bias:
                    pruned_router_bias = encoder_mlp.router.classifier.bias.data[core_expert_indices]
                    encoder_mlp.router.classifier.bias.data.copy_(pruned_router_bias)
            # prune experts
            for expert_idx in range(num_experts):
                if expert_idx not in core_expert_indices:
                    del encoder_mlp.experts[f"expert_{expert_idx}"]
            assert len(encoder_mlp.experts) == encoder_mlp.router.classifier.out_features
        # Decoder
        for layer_idx in range(1, 12, 2):
            decoder_mlp = model.decoder.block[layer_idx].layer[-1].mlp
            decoder_name = f"decoder.block.{layer_idx}.layer.2.mlp"
            core_expert_indices = torch.tensor(core_experts[decoder_name], dtype=torch.int).sort().values
            # prune router
            decoder_mlp.router.num_experts = len(core_expert_indices)
            with torch.no_grad():
                pruned_router_cls = decoder_mlp.router.classifier.weight.data[core_expert_indices]
                decoder_mlp.router.classifier = torch.nn.Linear(
                    decoder_mlp.router.classifier.in_features,
                    decoder_mlp.router.num_experts,
                    bias=router_bias
                )
                decoder_mlp.router.classifier.weight.data.copy_(pruned_router_cls)
                if router_bias:
                    pruned_router_bias = decoder_mlp.router.classifier.bias.data[core_expert_indices]
                    decoder_mlp.router.classifier.bias.data.copy_(pruned_router_bias)
            # prune experts
            for expert_idx in range(num_experts):
                if expert_idx not in core_expert_indices:
                    del decoder_mlp.experts[f"expert_{expert_idx}"]
            assert len(decoder_mlp.experts) == decoder_mlp.router.classifier.out_features

    return model


def profile_inference_latency(
        checkpoint: str,
        task: str,
        batch_size: Optional[int] = 256,
        merging_strategy: Optional[str] = None,
):
    if "t5" in checkpoint:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    else:
        model = SwitchTransformersForConditionalGeneration.from_pretrained(checkpoint)
    tokenizer = T5TokenizerFast.from_pretrained(checkpoint)
    dataloader = prepare_dataloader(task, tokenizer, batch_size)
    batch = next(iter(dataloader))
    batch = {k: v.cuda() for k, v in batch.items()}

    if merging_strategy == "redirect":
        model = merge_switch_by_usage_weighted_averaging(model, batch, prune_router_non_core=False)
    elif merging_strategy == "fuse":
        model = merge_switch_by_usage_weighted_averaging(model, batch, prune_router_non_core=True)

    if "t5" not in checkpoint:
        batch["output_router_logits"] = False

    model = model.cuda().eval()
    print("Start profiling...")
    num_runs = 100
    # 1. Test BF16
    model = model.half()
    # warmup
    for i in range(10):
        with torch.no_grad():
            model(**batch)

    # test
    torch.cuda.synchronize()
    bf16_start = time.time()
    for i in range(num_runs):
        with torch.no_grad():
            model(**batch)
    torch.cuda.synchronize()
    bf16_end = time.time()

    # 2. Test FP32
    model = model.float()
    # warmup
    for i in range(10):
        with torch.no_grad():
            model(**batch)

    # test
    torch.cuda.synchronize()
    fp32_start = time.time()
    for i in range(num_runs):
        with torch.no_grad():
            model(**batch)
    torch.cuda.synchronize()
    fp32_end = time.time()

    print(f"********************************************************")
    print(f"****** Latency profile for {checkpoint} on {task} ******")
    if merging_strategy is not None:
        print(f"Merging strategy: {merging_strategy} to 8 experts")
    print(f"Model number of parameters: {model.num_parameters()}")
    print(f"Number of runs: {num_runs}")
    print(f"Batch size: {batch_size}")
    print(f"Input sequence length: {batch['input_ids'].shape[1]}, target sequence length: {batch['labels'].shape[1]}")
    print(f"****** Results ******")
    print(f"FP32 Latency: {(fp32_end - fp32_start) / num_runs * 1000} ms")
    print(f"BF16 Latency: {(bf16_end - bf16_start) / num_runs * 1000} ms")


if __name__ == '__main__':
    Fire(profile_inference_latency)
