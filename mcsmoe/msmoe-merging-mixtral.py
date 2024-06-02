# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18
from typing import Optional

import logging
import torch
from fire import Fire
from transformers import MixtralForCausalLM, AutoTokenizer

from mcsmoe.evaluation import get_minipile_dataloder, evaluate_minipile_perplexity, evaluate_fewshot, get_calib_dataloder
from mcsmoe.merging.grouping_mixtral import ExpertsGrouperForMixtral, merge_by_groups_with_usage_weighted, merge_by_groups_within_and_across_models

logger = logging.getLogger(__name__)

def evaluate_mcsmoe(
        task: str,
        num_average_groups: int,
        model_name: Optional[str] = "mistralai/Mixtral-8x7B-v0.1",
        similarity_base: Optional[str] = "router-logits",
        mode: Optional[str] = "normal", 
        num_fewshot: Optional[int] = 0,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        output_path: Optional[str] = None,
):
    # torch.cuda.memory._record_memory_history(True)
    eval_ppl = (task == "minipile")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = MixtralForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, device_map="auto"
    )

    # dataloader_for_merging = get_minipile_dataloder(
    #     tokenizer=tokenizer,
    #     block_size=512,
    #     batch_size=1,
    #     subset_ratio=0.1,
    # )

    print("[MC-SMoE] Number of parameters before merging:", model.num_parameters())

    dataloader_for_merging = get_calib_dataloder(
        dataset="c4",
        tokenizer=tokenizer,
        max_block_size=2048,
        n_blocks_for_stat=128,
        batch_size=2,
        num_workers=4,
    )

    # MC-SMoE!
    print(f"[MC-SMoE] Merging into average {num_average_groups} groups...")

    grouper = ExpertsGrouperForMixtral(config=model.config, similarity_base=similarity_base)
    grouper.compute_all_similarities(model, dataloader_for_merging)
    # grouper.compute_all_usages(model, dataloader_for_merging)
    # dom_experts = grouper.group_experts_globally_from_dominant_experts(
    #     num_average_groups=num_average_groups, merging_layers=list(range(0, model.config.num_hidden_layers))
    # )

    # print(grouper._usage_frequency_state_dict)

    grouper.group_experts_randomly(num_groups=num_average_groups)

    # model = merge_by_groups_with_usage_weighted(
    #     model, grouper=grouper, merging_layers=list(range(0, model.config.num_hidden_layers))
    # )

    model.student = merge_by_groups_within_and_across_models(
        mixtral_model=model,
        grouper=grouper,
        dataloader=dataloader_for_merging,
        mode=mode,
        partition=partition,
        dominant_alone=False,
        usage_weighted=False
    )

    print(f"[MC-SMoE] ========= Grouping results ========= ")
    for name, state in grouper.group_state_dict().items():
        # print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]})")
        print(f"Group {name}: {state.tolist()}")
    
    model = model.cuda()

    print("[MC-SMoE] Number of parameters after merging:", model.num_parameters())

    if eval_ppl:
        evaluate_minipile_perplexity(
            model, tokenizer=tokenizer, batch_size=eval_batch_size, log=True
        )
    elif "," in task:
        tasks = task.split(",")
        for t in tasks:
            evaluate_fewshot(
                model, tokenizer=tokenizer, task=task, num_fewshot=num_fewshot, output_path=output_path+f"_{t}", log=True
            )
    else:
        evaluate_fewshot(
            model, tokenizer=tokenizer, task=task, num_fewshot=num_fewshot, output_path=output_path, log=True
        )


if __name__ == "__main__":
    Fire(evaluate_mcsmoe)
